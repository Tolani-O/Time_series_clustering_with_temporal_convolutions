import time
import torch
from src.TCN.general_functions import write_log_and_model, plot_outputs, write_losses, plot_losses

global loss_function, loss_optimizer, data_train, data_test, X_train, X_test, stim_time, args, output_dir, start_epoch, lr, Bspline_matrix


def initialization_training_epoch(log_likelihoods):
    loss_function.train()
    loss_optimizer.zero_grad()
    data = torch.stack(X_train)
    negLogLikelihood, latent_factors, cluster_attn, firing_attn = loss_function(data, mode='initialize_output')
    log_likelihoods.append(-negLogLikelihood.item())
    if args.clip > 0:
        torch.nn.utils.clip_grad_norm_(loss_function.parameters(), args.clip)
    # Backpropagate the loss through both the model and the loss_function
    negLogLikelihood.backward()
    # Update the parameters for both the model and the loss_function
    loss_optimizer.step()
    return log_likelihoods, latent_factors, cluster_attn, firing_attn


def learn_initial_outputs():
    # Initialize outputs
    total_time = 0
    log_likelihoods_train = []
    start_time = time.time()  # Record the start time of the epoch
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        log_likelihoods_train, latent_factors, cluster_attn, firing_attn = initialization_training_epoch(log_likelihoods_train)
        if epoch % args.log_interval == 0 or epoch == start_epoch + args.num_epochs - 1:
            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
            total_time += elapsed_time  # Calculate the total time for training
            cur_log_likelihood_train = log_likelihoods_train[-1]
            output_str = (
                f"Epoch: {epoch:2d}, Elapsed Time: {elapsed_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs,\n"
                f"Log Likelihood train: {cur_log_likelihood_train:.5f}, lr: {lr:.5f}\n")
            write_log_and_model(output_str, output_dir, epoch, loss_function=loss_function)
            latent_factors = latent_factors.detach().numpy()
            cluster_attn = cluster_attn.detach().numpy()
            firing_attn = firing_attn.detach().numpy()
            plot_outputs(latent_factors, cluster_attn, firing_attn, data_train.intensity, stim_time, output_dir, 'Train', epoch)
            is_empty = start_epoch == 0 and epoch == 0
            write_losses(log_likelihoods_train, 'Train', 'Likelihood', output_dir, is_empty)
            plot_losses(data_train.likelihood(), output_dir, 'Train', 'Likelihood', 20)
            log_likelihoods_train = []
            start_time = time.time()  # Record the start time of the epoch
            print(output_str)

