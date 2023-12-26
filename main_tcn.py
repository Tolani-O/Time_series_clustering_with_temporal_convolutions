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
from src.TCN.general_functions import int_or_str, create_second_diff_matrix, plot_outputs, write_losses, plot_spikes, \
    plot_intensity_and_latents, plot_latent_coupling, plot_losses, write_log_and_model, create_relevant_files, \
    load_model_checkpoint
from src.TCN.main_learn_initial_outputs import learn_initial_outputs
from src.TCN.main_learn_initial_maps import learn_initial_maps
from scipy.interpolate import BSpline
import time

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
parser.add_argument('--tau_beta', type=int, default=1, help='Value for tau_beta')
parser.add_argument('--tau_s', type=int, default=1, help='Value for tau_s')
parser.add_argument('--num_epochs', type=int, default=1500, help='Number of training epochs')
parser.add_argument('--notes', type=str, default='empty', help='Run notes')
parser.add_argument('--K', type=int, default=50, help='Number of neurons')
parser.add_argument('--R', type=int, default=5, help='Number of trials')
parser.add_argument('--L', type=int, default=3, help='Number of latent factors')
parser.add_argument('--intensity_mltply', type=float, default=25, help='Latent factor intensity multiplier')
parser.add_argument('--intensity_bias', type=float, default=1, help='Latent factor intensity bias')
parser.add_argument('--param_seed', type=int_or_str, default='TRUTH', help='')
parser.add_argument('--stage', type=str, default='finetune', help='options are: initialize_output, initialize_map, finetune')

args = parser.parse_args()

args.notes = 'Random_States'
args.param_seed = ''

if args.param_seed == '':
    args.param_seed = np.random.randint(0, 2 ** 32 - 1)
data_seed = np.random.randint(0, 2 ** 32 - 1)

# if args.plot_lkhd:
#     np.random.seed(data_seed)
#     true_data = DataAnalyzer().initialize(K=args.K, R=args.R, intensity_mltply=args.intensity_mltply, intensity_bias=args.intensity_bias,
#                                           max_offset=0)
#     plot_losses(true_data, args.K, args.R, args.L, args.intensity_mltply, args.intensity_bias, data_seed)
#     sys.exit()

folder_name = (f'paramSeed{args.param_seed}_dataSeed{data_seed}_L{args.L}_K{args.K}_R{args.R}'
               f'_int.mltply{args.intensity_mltply}_int.add{args.intensity_bias}_tauBeta{args.tau_beta}'
               f'_tauS{args.tau_s}_iters{args.num_epochs}_notes-{args.notes}')
print(f'folder_name: {folder_name}')
output_dir = os.path.join(os.getcwd(), 'outputs', folder_name, args.stage)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
plot_spikes(binned, output_dir)
plot_intensity_and_latents(data_train.time, data_train.latent_factors, data_train.intensity, output_dir)
plot_latent_coupling(data_train.latent_coupling, output_dir)
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
tau_beta = torch.tensor(args.tau_beta)
tau_s = torch.tensor(args.tau_s)

# Instantiate the dataset and dataloader
dataset = CustomDataset(X_train)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
# create some validation data
data_test = DataAnalyzer().initialize(K=args.K, R=args.R, intensity_mltply=args.intensity_mltply,
                                     intensity_bias=args.intensity_bias, max_offset=0)
true_likelihood_test = data_test.likelihood()
print(f"True likelihood Test: {true_likelihood_test}")
binned, stim_time = data_test.sample_data()
dims = binned.shape
X_test_array = binned.reshape(dims[0], args.R, int(dims[1] / args.R)).transpose(0, 2, 1)
X_test = [torch.Tensor(X_test_array[i].astype(np.float64)) for i in range(dims[0])]

start_epoch = 0
if args.load_only or args.load_and_train:
    model, loss_function, start_epoch = load_model_checkpoint(output_dir, args.num_epochs)

if args.load_only:
        sys.exit()

if not args.load_and_train:
    create_relevant_files(output_dir, args, data_seed)
    if isinstance(args.param_seed, int):
        torch.manual_seed(args.param_seed)
    model = TemporalConvNet(state_size, input_size, output_size, n_channels, output_size, bin_kernel_size,
                            kernel_size, args.dropout)
    # Attach hooks
    # model.register_hooks()
    loss_function = NegLogLikelihood(state_size, len(X_train), Bspline_matrix, Delta2TDelta2, dt, tau_beta, tau_s)
    if args.param_seed == 'TRUTH':
        model.init_ground_truth(torch.tensor(data_train.latent_factors).float(), Bspline_matrix)

if args.cuda:
    model.cuda()
    loss_function.cuda()

lr = 1e-4 #args.lr
model_optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
loss_optimizer = getattr(optim, args.optim)(loss_function.parameters(), lr=lr)

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


def train(log_likelihoods, losses):
    model.train()
    loss_function.train()
    for binned in dataloader:
        if args.cuda: binned = binned.cuda()
        # Zero out the gradients for both optimizers
        model_optimizer.zero_grad()
        loss_optimizer.zero_grad()
        latent_coeffs, cluster_attn, firing_attn = model(binned)
        loss, LogLikelihood, smoothness_budget_constrained = loss_function(binned, latent_coeffs, cluster_attn, firing_attn)
        log_likelihoods.append(-LogLikelihood.item())
        losses.append(-loss.item())
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # Backpropagate the loss through both the model and the loss_function
        loss.backward()
        # Update the parameters for both the model and the loss_function
        model_optimizer.step()
        loss_optimizer.step()

    return log_likelihoods, losses, smoothness_budget_constrained


def evaluate(data, log_likelihoods, losses):
    model.eval()
    loss_function.eval()
    with torch.no_grad():
        data = torch.stack(data)
        if args.cuda: data = data.cuda()
        latent_coeffs, cluster_attn, firing_attn = model(data)
        loss, LogLikelihood, _ = loss_function(data, latent_coeffs, cluster_attn, firing_attn)
    log_likelihoods.append(-LogLikelihood.item())
    losses.append(-loss.item())
    return log_likelihoods, losses


if __name__ == "__main__":

    if args.stage == 'initialize_output':
        learn_initial_outputs()
        sys.exit()

    if args.stage == 'initialize_map':
        learn_initial_maps()
        sys.exit()

    total_time = 0
    log_likelihoods = []
    losses = []
    log_likelihoods_train = []
    losses_train = []
    log_likelihoods_test = []
    losses_test = []
    start_time = time.time()  # Record the start time of the epoch
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        log_likelihoods, losses, smoothness_budget_constrained = train(log_likelihoods, losses)
        log_likelihoods_train, losses_train = evaluate(X_train, log_likelihoods_train, losses_train)
        log_likelihoods_test, losses_test = evaluate(X_test, log_likelihoods_test, losses_test)
        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_loss_train = losses_train[-1]
            cur_log_likelihood_test = log_likelihoods_test[-1]
            cur_loss_test = losses_test[-1]
            smoothness_budget_constrained = smoothness_budget_constrained.detach().numpy()
            output_str = (f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                          f"Loss train: {cur_loss_train:.5f}, Log Likelihood train: {cur_log_likelihood_train:.5f},\n"
                          f"Loss test: {cur_loss_test:.5f}, Log Likelihood test: {cur_log_likelihood_test:.5f},\n"
                          f"lr: {lr:.5f}, smoothness_budget: {smoothness_budget_constrained.T}\n")
            plot_outputs(model, X_train, data_train.intensity, stim_time, Bspline_matrix, output_dir, 'Train', epoch)
            plot_outputs(model, X_test, data_test.intensity, stim_time, Bspline_matrix, output_dir, 'Test', epoch)
            write_log_and_model(model, loss_function, output_str, output_dir, epoch)
            is_empty = start_epoch==0 and epoch==0
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            write_losses(losses_train, 'Train', 'Loss', output_dir, is_empty)
            write_losses(log_likelihoods_test, 'Test', 'Likelihood', output_dir, is_empty)
            write_losses(losses_test, 'Test', 'Loss', output_dir, is_empty)
            write_losses(log_likelihoods, 'Batch', 'Likelihood', output_dir, is_empty)
            write_losses(losses, 'Batch', 'Loss', output_dir, is_empty)
            plot_losses(data_train.likelihood(), output_dir, 'Train', 'Likelihood', 20)
            plot_losses(0, output_dir, 'Train', 'Loss', 20)
            plot_losses(data_test.likelihood(), output_dir, 'Test', 'Likelihood', 20)
            plot_losses(0, output_dir, 'Test', 'Loss', 20)
            plot_losses(data_train.likelihood(), output_dir, 'Batch', 'Likelihood', 100)
            plot_losses(0, output_dir, 'Batch', 'Loss', 100)
            # if cur_log_likelihood_test > max(log_likelihoods_test[-6:-1]):
            #     lr /= 10
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            log_likelihoods = []
            losses = []
            log_likelihoods_test = []
            losses_test = []
            start_time = time.time()  # Record the start time of the epoch
            print(output_str)
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
