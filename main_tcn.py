import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.TCN.tcn import TemporalConvNet, CustomDataset, NegLogLikelihood
from src.simulate_data import DataAnalyzer
import numpy as np
from src.TCN.general_functions import int_or_str, create_second_diff_matrix, plot_spikes, \
    plot_intensity_and_latents, plot_latent_coupling, create_relevant_files, \
    load_model_checkpoint
from src.TCN.main_learn_initial_outputs import learn_initial_outputs
from src.TCN.main_learn_initial_maps import learn_initial_maps
from src.TCN.main_finetune_maps import finetune_maps
from scipy.interpolate import BSpline

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_false', default=False, help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.25, help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2, help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--ksize', type=int, default=5, help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4, help='# of levels (default: 4)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150, help='number of hidden units per layer (default: 150)')
parser.add_argument('--batch_size', type=int, default='10', help='the batch size for training')
parser.add_argument('--plot_lkhd', type=int, default=0, help='')
parser.add_argument('--load_only', type=int, default=0, help='')
parser.add_argument('--load_and_train', type=int, default=0, help='')
parser.add_argument('--tau_psi', type=int, default=1, help='Value for tau_psi')
parser.add_argument('--tau_beta', type=int, default=100, help='Value for tau_beta')
parser.add_argument('--tau_s', type=int, default=10, help='Value for tau_s')
parser.add_argument('--num_epochs', type=int, default=1500, help='Number of training epochs')
parser.add_argument('--notes', type=str, default='empty', help='Run notes')
parser.add_argument('--K', type=int, default=50, help='Number of neurons')
parser.add_argument('--R', type=int, default=5, help='Number of trials')
parser.add_argument('--L', type=int, default=3, help='Number of latent factors')
parser.add_argument('--intensity_mltply', type=float, default=25, help='Latent factor intensity multiplier')
parser.add_argument('--intensity_bias', type=float, default=1, help='Latent factor intensity bias')
parser.add_argument('--param_seed', type=int_or_str, default='Truth', help='options are: seed (int), Truth (str), Learned (str)')
parser.add_argument('--stage', type=str, default='finetune', help='options are: initialize_output, initialize_map, finetune')

args = parser.parse_args()

args.stage = 'initialize_map'
folder_name = 'paramSeed192600038_dataSeed4223320140_L3_K50_R5_int.mltply25_int.add1_tauBeta3000_tauS10000_iters25000_notes-initialize_states_heavy_penalty'
args.num_epochs = 25000
args.lr = 1e-3
args.tau_beta = 3000
args.tau_s = 10000
load_epoch = args.num_epochs - 1


if args.param_seed == '':
    args.param_seed = np.random.randint(0, 2 ** 32 - 1)
data_seed = np.random.randint(0, 2 ** 32 - 1)
sub_folder_name = args.stage

if args.stage == 'initialize_map':
    args.load_and_train = 1
    args.param_seed = 'learned'
    sub_folder_name = 'initialize_output'

start_epoch = 0
if args.load_only or args.load_and_train:
    load_dir = os.path.join(os.getcwd(), 'outputs', folder_name, sub_folder_name)
    model, loss_function, start_epoch = load_model_checkpoint(load_dir, load_epoch)
else:
    folder_name = (f'paramSeed{args.param_seed}_dataSeed{data_seed}_L{args.L}_K{args.K}_R{args.R}'
                   f'_int.mltply{args.intensity_mltply}_int.add{args.intensity_bias}_tauBeta{args.tau_beta}'
                   f'_tauS{args.tau_s}_iters{args.num_epochs}_notes-{args.notes}')

if args.load_only:
    sys.exit()

print(f'folder_name: {folder_name}')
output_dir = os.path.join(os.getcwd(), 'outputs', folder_name, args.stage)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Set the random seed manually for reproducibility.
torch.manual_seed(data_seed)
data_train = DataAnalyzer().initialize(K=args.K, R=args.R, intensity_mltply=args.intensity_mltply,
                                 intensity_bias=args.intensity_bias, max_offset=0)
true_likelihood_train = data_train.likelihood()
print(f"True likelihood Training: {true_likelihood_train}")
binned, stim_time = data_train.sample_data()
dims = binned.shape
X_train_array = binned.reshape(dims[0], args.R, int(dims[1] / args.R)).transpose(0, 2, 1)
X_train = [torch.Tensor(X_train_array[i].astype(np.float64)) for i in range(dims[0])]
degree = 3
n_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
input_size = args.R
output_size = args.nhid
bin_kernel_size = 2
dt = torch.round(torch.tensor(stim_time[1] - stim_time[0]) * 1000) / 1000
knots = np.concatenate([np.repeat(stim_time[0], degree), stim_time, np.repeat(stim_time[-1], degree)])
Bspline_matrix = torch.tensor(BSpline.design_matrix(stim_time, knots, degree).transpose().toarray()).float()
state_size = (args.L, Bspline_matrix.shape[0])
Delta2 = create_second_diff_matrix(stim_time.shape[0])
Delta2TDelta2 = torch.tensor(Delta2.T @ Delta2).float()

# Instantiate the dataset and dataloader
dataset = CustomDataset(X_train)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
# create some validation data
data_test = DataAnalyzer().initialize(K=args.K, R=args.R, intensity_mltply=args.intensity_mltply,
                                     intensity_bias=args.intensity_bias, max_offset=0)
true_likelihood_test = data_test.likelihood()
print(f"True likelihood Test: {true_likelihood_test}")
binned_test, stim_time = data_test.sample_data()
dims = binned_test.shape
X_test_array = binned_test.reshape(dims[0], args.R, int(dims[1] / args.R)).transpose(0, 2, 1)
X_test = [torch.Tensor(X_test_array[i].astype(np.float64)) for i in range(dims[0])]

if not args.load_and_train:
    os.makedirs(output_dir)
    plot_spikes(binned, output_dir)
    plot_intensity_and_latents(data_train.time, data_train.latent_factors, data_train.intensity, output_dir)
    plot_latent_coupling(data_train.latent_coupling, output_dir)
    create_relevant_files(output_dir, args, data_seed)

    if isinstance(args.param_seed, int):
        torch.manual_seed(args.param_seed)
    model = TemporalConvNet(state_size, input_size, output_size, n_channels, output_size, bin_kernel_size,
                            kernel_size, args.dropout)
    # Attach hooks
    # model.register_hooks()
    loss_function = NegLogLikelihood(state_size, len(X_train), Bspline_matrix, Delta2TDelta2, dt)
    if args.param_seed.lower() == 'truth':
        model.init_states(torch.tensor(data_train.latent_factors).float(), Bspline_matrix)
    elif args.param_seed.lower() == 'learned':
        model.init_states(loss_function.state.clone(), Bspline_matrix)

if args.cuda:
    model.cuda()
    loss_function.cuda()

model_optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
loss_optimizer = getattr(optim, args.optim)(loss_function.parameters(), lr=args.lr)

# # Visualize networks
# cluster_attn = F.softmax(torch.randn(1, state_size[0]), dim=-1)
# firing_attn = F.softplus(torch.randn(1, state_size[0]))
# latent_coeffs = F.softplus(torch.randn(state_size))
# torch.onnx.export(model, X_train[0].unsqueeze(0), 'runs/model/model.onnx', input_names=["features"], output_names=["outputs"])
# torch.onnx.export(loss_function, (X_train[0].unsqueeze(0), latent_coeffs, cluster_attn, firing_attn),
#                   'runs/loss/loss.onnx', input_names=["features"], output_names=["outputs"])
# # Initialize a writer
# # For the model graph
# model_writer = SummaryWriter('runs/model')
# model_writer.add_graph(model, X_train[0].unsqueeze(0))
# model_writer.close()
# # For the loss graph
# loss_writer = SummaryWriter('runs/loss')
# loss_writer.add_graph(loss_function, [X_train[0].unsqueeze(0), latent_coeffs, cluster_attn, firing_attn])
# loss_writer.close()
# print('DONE')

if __name__ == "__main__":

    if args.stage == 'initialize_output':
        learn_initial_outputs(globals())
        sys.exit()

    if args.stage == 'initialize_map':
        learn_initial_maps(globals())
        sys.exit()

    if args.stage == 'finetune':
        finetune_maps(globals())
        sys.exit()

        # if epoch > args.log_interval and len(log_likelihoods_test) > 5 and log_likelihoods_test[-1] > max(log_likelihoods_test[-6:-1]):
        #     lr /= 10
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # # Plot activations
        # act = model.activations['conv1'].squeeze()
        # plt.figure(figsize=(15, 5))
        # for i in range(act.size(0)):  # Iterate over all features/neurons
        #     plt.plot(act[i].cpu().numpy(), label=f'Feature {i}')
        # plt.title('Feature Activations Over Sequence')
        # plt.xlabel('Sequence Position')
        # plt.ylabel('Activation')
        # plt.legend()
        # plt.show()
        #
        # # Using seaborn for heatmap
        # plt.figure(figsize=(10, 5))
        # sns.heatmap(act.cpu().numpy(), cmap='viridis')
        # plt.title('Activation Heatmap')
        # plt.xlabel('Sequence Position')
        # plt.ylabel('Feature/Neuron')
        # plt.show()
