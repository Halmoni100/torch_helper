import math

import torch
import matplotlib.pyplot as plt

from progress_bar import ProgressBar

def train(train_dataloader, val_dataloader, train_size, val_size, inputs_labels_func,
          model, criterion, optimizer, metric=None, metric_name=None, scheduler=None,
          device=torch.device("cpu"), num_epochs=100, pbar_len=80, do_carriage_return=True):
    history = {'loss': [], 'val_loss': []}
    if metric is not None and metric_name is not None:
        history[metric_name] = []
        history['val_' + metric_name] = []

    train_ratio = train_size / (train_size + val_size)
    val_ratio = val_size / (train_size + val_size)
    pbar_train_len = pbar_len if do_carriage_return else math.floor(pbar_len * train_ratio)
    pbar_val_len = pbar_len if do_carriage_return else math.floor(pbar_len * val_ratio)
    if not do_carriage_return:
        ProgressBar.print_total_line(pbar_train_len)
        ProgressBar.print_total_line(pbar_val_len)
        print()

    epoch = 0
    while epoch < num_epochs:
        # Train
        pbar = ProgressBar(len(train_dataloader), length=pbar_train_len,
                           do_carriage_return=do_carriage_return)
        pbar.start(front_msg="Train ")

        optimizer.zero_grad()
        train_metric_sum = torch.zeros(1, device=device)
        train_loss_sum = torch.zeros(1, device=device)
        for data in train_dataloader:
            optimizer.zero_grad()
            inputs, labels = inputs_labels_func(data)
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss_sum += train_loss.detach()
            if metric is not None:
                with torch.no_grad():
                    train_metric_sum += metric(outputs, labels)
            train_loss.backward()
            optimizer.step()

            pbar.update(front_msg="Train ")
        pbar.reset()

        # Validation
        torch.set_grad_enabled(False)

        pbar = ProgressBar(len(val_dataloader), lenth=pbar_val_len,
                           do_carriage_return=do_carriage_return)
        pbar.start(front_msg="Val ")

        val_metric_sum = torch.zeros(1, device=device)
        val_loss_sum = torch.zeros(1, device=device)
        for data in val_dataloader:
            inputs, labels = inputs_labels_func(data)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_sum += loss
            if metric is not None:
                val_metric_sum += metric(outputs, labels)

            pbar.update(front_msg="Val ")
        pbar.reset()

        # Print and record stastistics

        avg_train_loss = (train_loss_sum.cpu() / train_size).item()
        avg_train_metric = (train_metric_sum.cpu() / train_size).item()
        avg_val_loss = (val_loss_sum.cpu() / val_size).item()
        avg_val_metric = (val_metric_sum.cpu() / val_size).item()

        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        if metric is None:
            print("[epoch %2d]  train loss: %.5f  val loss: %.5f" % (
                  epoch + 1, avg_train_loss, avg_val_loss))
        else:
            print("[epoch %2d]  train loss: %.5f  train %s: %.3f val loss: %.5f  val %s: %.3f" %
                  (epoch + 1, avg_train_loss, metric_name, avg_train_metric,
                   avg_val_loss, metric_name, avg_val_metric))
            history[metric_name].append(avg_train_metric)
            history['val_' + metric_name].append(avg_val_metric)

        torch.set_grad_enabled(True)

        if scheduler is not None:
            scheduler.step()

        epoch += 1

def plot_training_history(history, metric_name=None, metric_full_name=None, save_path=None):
    loss = history['loss']
    val_loss = history['val_loss']
    train_metric, val_metric = None, None
    if metric_name is not None:
        train_metric = history[metric_name]
        val_metric = history['val_' + metric_name]

    epochs = range(len(loss))

    plt.figure(figsize=(8, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    if metric_name is not None:
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_metric, 'bo', label='Training ' + metric_name)
        plt.plot(epochs, val_metric, 'b', label='Validation ' + metric_name)
        plt.title('Training and validation ' + metric_full_name)
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()