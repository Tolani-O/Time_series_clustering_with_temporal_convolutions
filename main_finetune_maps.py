import time
import torch
from src.TCN.general_functions import plot_outputs, write_log_and_model, write_losses, plot_losses


def define_global_vars(global_vars):
    global model, loss_function, model_optimizer, loss_optimizer, dataloader, data_train, data_test, X_train, X_test, \
        stim_time, args, output_dir, start_epoch, Bspline_matrix
    model = global_vars['model']
    loss_function = global_vars['loss_function']
    model_optimizer = global_vars['model_optimizer']
    loss_optimizer = global_vars['loss_optimizer']
    dataloader = global_vars['dataloader']
    data_train = global_vars['data_train']
    data_test = global_vars['data_test']
    X_train = global_vars['X_train']
    X_test = global_vars['X_test']
    stim_time = global_vars['stim_time']
    args = global_vars['args']
    output_dir = global_vars['output_dir']
    start_epoch = global_vars['start_epoch']
    Bspline_matrix = global_vars['Bspline_matrix']


def initialization_training_epoch(log_likelihoods, losses):
    model.train()
    loss_function.train()
    for binned in dataloader:
        if args.cuda: binned = binned.cuda()
        # Zero out the gradients for both optimizers
        model_optimizer.zero_grad()
        loss_optimizer.zero_grad()
        latent_coeffs, cluster_attn, firing_attn = model(binned)
        loss, negLogLikelihood, latent_factors, smoothness_budget_constrained = (
            loss_function(binned, latent_coeffs, cluster_attn, firing_attn, args.tau_beta, args.tau_s))
        log_likelihoods.append(-negLogLikelihood.item())
        losses.append(-loss.item())
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # Backpropagate the loss through both the model and the loss_function
        loss.backward()
        # Update the parameters for both the model and the loss_function
        model_optimizer.step()
        loss_optimizer.step()
    return log_likelihoods, losses, smoothness_budget_constrained


def initialization_evaluation(data, log_likelihoods, losses):
    model.eval()
    loss_function.eval()
    with (torch.no_grad()):
        data = torch.stack(data)
        if args.cuda: data = data.cuda()
        latent_coeffs, cluster_attn, firing_attn = model(data)
        loss, negLogLikelihood, latent_factors, smoothness_budget_constrained = (
            loss_function(data, latent_coeffs, cluster_attn, firing_attn, args.tau_beta, args.tau_s))
    log_likelihoods.append(-negLogLikelihood.item())
    losses.append(-loss.item())
    return log_likelihoods, losses, latent_factors, cluster_attn, firing_attn


def finetune_maps(global_vars):
    define_global_vars(global_vars)
    total_time = 0
    log_likelihoods = []
    losses = []
    log_likelihoods_train = []
    losses_train = []
    log_likelihoods_test = []
    losses_test = []
    start_time = time.time()  # Record the start time of the epoch
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        log_likelihoods, losses, smoothness_budget_constrained = initialization_training_epoch(log_likelihoods, losses)
        log_likelihoods_train, losses_train, latent_factors_train, cluster_attn_train, firing_attn_train = (
            initialization_evaluation(X_train, log_likelihoods_train, losses_train))
        log_likelihoods_test, losses_test, latent_factors_test, cluster_attn_test, firing_attn_test = (
            initialization_evaluation(X_test, log_likelihoods_test, losses_test))
        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            cur_loss_train = losses_train[-1]
            cur_log_likelihood_test = log_likelihoods_test[-1]
            cur_loss_test = losses_test[-1]
            smoothness_budget_constrained = smoothness_budget_constrained.detach().numpy()
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Loss train: {cur_loss_train:.5f}, Log Likelihood train: {cur_log_likelihood_train:.5f},\n"
                f"Loss test: {cur_loss_test:.5f}, Log Likelihood test: {cur_log_likelihood_test:.5f},\n"
                f"lr: {args.lr:.5f}, smoothness_budget: {smoothness_budget_constrained.T}\n")
            write_log_and_model(output_str, output_dir, epoch, model, loss_function)
            latent_factors = latent_factors_train.detach().numpy()
            cluster_attn = cluster_attn_train.detach().numpy()
            firing_attn = firing_attn_train.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, output_dir,
                         'Train', epoch)
            latent_factors = latent_factors_test.detach().numpy()
            cluster_attn = cluster_attn_test.detach().numpy()
            firing_attn = firing_attn_test.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, output_dir,
                         'Test', epoch)
            is_empty = start_epoch == 0 and epoch == 0
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
            log_likelihoods = []
            losses = []
            log_likelihoods_test = []
            losses_test = []
            start_time = time.time()  # Record the start time of the epoch
            print(output_str)
