import time
import torch
import torch.nn.functional as F
from src.TCN.general_functions import write_log_and_model, plot_outputs, write_losses, plot_losses


def define_global_vars(global_vars):
    global model, loss_function, model_optimizer, data_train, data_test, X_train, X_test, stim_time, args, output_dir, start_epoch, Bspline_matrix
    model = global_vars['model']
    loss_function = global_vars['loss_function']
    model_optimizer = global_vars['model_optimizer']
    data_train = global_vars['data_train']
    data_test = global_vars['data_test']
    X_train = global_vars['X_train']
    X_test = global_vars['X_test']
    stim_time = global_vars['stim_time']
    args = global_vars['args']
    output_dir = global_vars['output_dir']
    start_epoch = global_vars['start_epoch']
    Bspline_matrix = global_vars['Bspline_matrix']


def initialization_training_epoch(losses):
    model.train()
    model_optimizer.zero_grad()
    data = torch.stack(X_train)
    latent_coeffs, cluster_attn, firing_attn = model(data)
    loss = loss_function(cluster_attn=cluster_attn, firing_attn=firing_attn, mode='initialize_map')
    losses.append(-loss.item())
    if args.clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    # Backpropagate the loss through both the model and the loss_function
    loss.backward()
    # Update the parameters for both the model and the loss_function
    model_optimizer.step()
    return losses, cluster_attn, firing_attn


def initialization_evaluation(losses):
    model.eval()
    with torch.no_grad():
        data = torch.stack(X_test)
        if args.cuda: data = data.cuda()
        latent_coeffs, cluster_attn, firing_attn = model(data)
        loss = loss_function(cluster_attn=cluster_attn, firing_attn=firing_attn, mode='initialize_map')
    losses.append(-loss.item())
    return losses, cluster_attn, firing_attn


def learn_initial_maps(global_vars):
    define_global_vars(global_vars)
    total_time = 0
    losses_train = []
    losses_test = []
    start_time = time.time()  # Record the start time of the epoch
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        losses_train, cluster_attn_train, firing_attn_train = initialization_training_epoch(losses_train)
        losses_test, cluster_attn_test, firing_attn_test = initialization_evaluation(losses_test)
        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_loss_train = losses_train[-1]
            cur_loss_test = losses_test[-1]
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Loss train: {cur_loss_train:.5f}, Loss test: {cur_loss_test:.5f}, lr: {args.lr:.5f}\n")
            write_log_and_model(output_str, output_dir, epoch, model=model)
            latent_coeffs = F.softplus(loss_function.state.detach())
            latent_factors = torch.matmul(latent_coeffs, Bspline_matrix).numpy()
            cluster_attn = cluster_attn_train.detach().numpy()
            firing_attn = firing_attn_train.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, output_dir, 'Train', epoch)
            cluster_attn = cluster_attn_test.detach().numpy()
            firing_attn = firing_attn_test.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, output_dir, 'Test', epoch)
            is_empty = start_epoch == 0 and epoch == 0
            write_losses(losses_train, 'Train', 'Loss', output_dir, is_empty)
            write_losses(losses_test, 'Test', 'Loss', output_dir, is_empty)
            plot_losses(0, output_dir, 'Train', 'Loss', 20)
            plot_losses(0, output_dir, 'Test', 'Loss', 20)
            losses_train = []
            losses_test = []
            start_time = time.time()  # Record the start time of the epoch
            print(output_str)
