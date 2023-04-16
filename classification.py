import numpy as np
import argparse
import os
import time
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse

from lib.data import load_feature
from lib.nn_model import EdgeDetectionDataset, EdgeDetectionNet

"""Train a classification model that maps the weak detector's intermediate feature map to the offloading reward
class."""


@dataclass
class SaveOpt:
    """Options for loading/saving model weights."""
    model_dir: str = ''  # Directory to save the model weights.
    load: bool = False  # If model is loaded from pre-trained weights.
    save: bool = True  # If model weights need to be saved after training.


_SaveOPT = SaveOpt()


def fit_model(model, name, data, save_opts=_SaveOPT):
    """
    Fit a (non-CNN based) classification model that predicts offloading reward class based on weak detector feature map.
    :param model: the classification model.
    :param name: name of the classification model.
    :param data: features (inputs) and rewards (labels) for training and validation.
    :param save_opts: model saving options.
    :return: the estimated offloading reward class for the training and validation dataset,
             and the average inference time for each image.
    """
    # Retrieve data and flatten the feature maps.
    train_feature, val_feature, train_reward, val_reward = data
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    # Load model if specified.
    if save_opts.load and save_opts.model_dir != '':
        cls, scaler = pickle.load(open(os.path.join(save_opts.model_dir, 'wts.pickle'), 'rb'))
        # Normalize the data.
        train_feature = scaler.transform(train_feature)
        val_feature = scaler.transform(val_feature)
    else:
        scaler = StandardScaler().fit(train_feature)
        # Normalize the data.
        train_feature = scaler.transform(train_feature)
        val_feature = scaler.transform(val_feature)
        cls = model.fit(train_feature, train_reward)
    # Make predictions for both the training and test set.
    time1 = time.perf_counter()
    train_est = cls.predict(train_feature)
    time2 = time.perf_counter()
    val_est = cls.predict(val_feature)
    time3 = time.perf_counter()
    train_time, val_time = (time2 - time1) / len(train_reward), (time3 - time2) / len(val_reward)
    # Todo: replace the error metric with cross entropy.
    train_mse, val_mse = mse(train_reward, train_est), mse(val_reward, val_est)
    print(f"Trained {name} model with training MSE: {train_mse:.3f}, validation MSE: {val_mse:.3f}")
    # Save model if specified.
    if save_opts.save and save_opts.model_dir != '':
        Path(save_opts.model_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump((cls, scaler), open(os.path.join(save_opts.model_dir, 'wts.pickle'), 'wb'))
    return {"train_est": train_est, "val_est": val_est, "train_time": train_time, "val_time": val_time}


@dataclass
class CNNOpt:
    """Options for the Convolutional Neural Network model."""
    resize: bool = True  # Whether the inputs (feature maps extracted from the weak detector) have the same shape.
    learning_rate: float = 1e-3  # Initial learning rate.
    gamma: float = 0.1  # Scale for updating learning rate at each milestone.
    weight_decay: float = 1e-5  # Weight decay parameter for optimizer.
    milestones: List = field(default_factory=lambda: [50, 65, 75])  # Epochs to update the learning rate.
    max_epoch: int = 30  #80  # Maximum number of epochs for training.
    batch_size: int = 64  # Batch size for model training.
    channels: List = field(default_factory=lambda: [])  # Number of channels in each conv layer.
    kernels: List = field(default_factory=lambda: [])  # Kernel size for each conv layer.
    pools: List = field(default_factory=lambda: [])  # Whether max-pooling each conv layer.
    linear: List = field(
        default_factory=lambda: [145, 64, 64, 64, 50])  # Number of features in each linear after the conv layers.
    test_epoch: int = 1  # Number of epochs for periodic test using the validation set.


_CNNOPT = CNNOpt()


def fit_CNN(data, opts=_CNNOPT, save_opts=_SaveOPT, plot=True):
    """Fit a Convolutional Neural Network to predict offloading reward."""
    # Import pytorch.
    import torch
    from torch.utils.data import DataLoader
    # Prepare the dataset.
    train_feature, val_feature, train_reward, val_reward = data
    train_data = EdgeDetectionDataset(train_feature, train_reward, False)
    val_data = EdgeDetectionDataset(val_feature, val_reward, False)
    train_dataloader = DataLoader(train_data, batch_size=opts.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=opts.batch_size)
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # Build the CNN model.
    model = EdgeDetectionNet(opts.channels, opts.kernels, opts.pools, opts.linear, opts.resize).to(device)
    print(model)
    # Load weights if specified.
    if save_opts.load and save_opts.model_dir != '':
        model.load_state_dict(torch.load(os.path.join(save_opts.model_dir, 'wts.pth')))
    # Declare loss function, optimizer, and scheduler.
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=opts.gamma)

    # Define the training and test function.
    def train(dataloader, model, loss_fn, optimizer):
        num_batches, size = len(dataloader), len(dataloader.dataset)
        model.train()
        train_loss, process = 0, 0
        for batch, (X, y) in enumerate(dataloader):
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            processed = (batch + 1) * len(X)
            if processed / size >= process:
                process += 0.2
                print(f"loss: {train_loss / (batch + 1):>7f}  [{processed:>5d}/{size:>5d}]")
        return train_loss / num_batches

    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Avg Test Loss: {test_loss:>8f} \n")
        return test_loss

    # The training loop.
    train_loss, test_loss = list(), list()
    for t in range(opts.max_epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss.append(train(train_dataloader, model, loss_fn, optimizer))
        if t % opts.test_epoch == 0:
            test_loss.append(test(val_dataloader, model, loss_fn))
        scheduler.step()
    # Create a plot to visualize the training and test loss as a function of epoch number.
    if plot:
        CNN_plot(train_loss, test_loss, opts.test_epoch, opts.milestones)
    # Estimate the offloading reward for both training and validation set.
    with torch.no_grad():
        train_est, val_est = list(), list()
        time1 = time.perf_counter()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            train_est.append(model(X).cpu().numpy())
        train_est = np.concatenate(train_est)
        time2 = time.perf_counter()
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            val_est.append(model(X).cpu().numpy())
        val_est = np.concatenate(val_est)
        time3 = time.perf_counter()
    train_time = (time2 - time1) / len(train_dataloader.dataset)
    val_time = (time3 - time2) / len(val_dataloader.dataset)
    # Save model if specified.
    if save_opts.save and save_opts.model_dir != '':
        Path(save_opts.model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_opts.model_dir, 'wts.pth'))
    train_est = np.argmax(train_est, axis=1)
    val_est = np.argmax(val_est, axis=1)
    return {"train_est": train_est, "val_est": val_est, "train_time": train_time, "val_time": val_time}


def CNN_plot(train_loss, test_loss, test_epoch, lr_schedule):
    """Visualize the training of CNN model."""
    # Create the plot
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams['figure.figsize'] = (15, 10)
    fig, ax = plt.subplots()
    # Configure the subplot setting
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)
    ax.yaxis.grid(True, color='#C0C0C0')
    ax.xaxis.grid(True, color='#C0C0C0')
    ax.spines['top'].set_color('#606060')
    ax.spines['bottom'].set_color('#606060')
    ax.spines['left'].set_color('#606060')
    ax.spines['right'].set_color('#606060')
    ax.set_xlabel("Number of Epochs", labelpad=25, color='#333333', size=40)
    ax.set_ylabel("Model Loss", labelpad=30, color='#333333', size=35)
    # Plot the loss.
    ax.plot(np.arange(len(train_loss)) + 1, train_loss, linewidth=3, color='red', marker='o', markersize=15,
            label="train error")
    ax.plot(np.arange(1, len(train_loss) + 1, test_epoch), test_loss, linewidth=3, color='blue', marker='o',
            markersize=15, label="test error")
    # Plot the scheduled learning rate drop epochs.
    for idx, sched in enumerate(lr_schedule):
        line, = ax.plot([sched, sched],
                        [min(np.amin(train_loss), np.amin(test_loss)), max(np.amax(train_loss), np.amax(test_loss))],
                        linewidth=3, color='black')
        if idx == 0:
            line.set_label('lr schedule')
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    ax.legend(ax_handles, ax_labels, fontsize=20)
    plt.tight_layout()
    plt.savefig('./cnn_training.pdf', bbox_inches='tight')
    plt.show()
    return


def main(opts):
    # Load the weak detector feature maps for the training and validation dataset.
    ifpool = opts.pool_size > 0 and opts.stage != 24
    train_feature = load_feature(opts.train_dir, opts.stage, pool=ifpool, size=opts.pool_size)
    val_feature = load_feature(opts.val_dir, opts.stage, pool=ifpool, size=opts.pool_size)
    # Load the offloading rewards for the training dataset.
    train_reward = np.load(opts.train_label)['mapi']
    val_reward = np.load(opts.val_label)['mapi']
    # Split the rewards into classes.
    assert opts.num_class >= 2, "Please pick at least 2 classes to split offloading rewards."
    train_reward_sorted = np.sort(train_reward)
    threshold = [train_reward_sorted[int(i / opts.num_class * (len(train_reward) - 1))] for i in
                 range(opts.num_class + 1)]
    train_reward_classed = np.zeros_like(train_reward, dtype=int)
    val_reward_classed = np.zeros_like(val_reward, dtype=int)
    for idx, (thresh_low, thresh_high) in enumerate(zip(threshold[:-1], threshold[1:])):
        train_reward_classed[np.logical_and(train_reward >= thresh_low, train_reward <= thresh_high)] = idx
        val_reward_classed[np.logical_and(val_reward >= thresh_low, val_reward <= thresh_high)] = idx
    train_reward = train_reward_classed
    val_reward = val_reward_classed
    assert len(train_feature) == len(
        train_reward), "Inconsistent number of training feature maps and offloading rewards."
    assert len(val_feature) == len(
        val_reward), "Inconsistent number of validation feature maps and offloading rewards."
    # Select and fit the classification model.
    # TODO: register the classification models.
    model_names = ['CNN']
    models = [fit_CNN]
    try:
        model_idx = model_names.index(opts.model)
        model = models[model_idx]
    except ValueError:
        print("Please select a classification model from 'CNN' (Convolutional Neural Network).")
    if opts.pool_size == 0 and opts.stage != 24:
        # Check if model selection is consistent with pooling decision.
        assert opts.model == 'CNN', "Only fully convolutional NN can take input with different shapes. " + \
                                    "Please set model to 'CNN' if you choose to skip the RoI pooling step."
        # Force batch size to 1 when input feature maps have different shapes.
        _CNNOPT.resize = False
        _CNNOPT.batch_size = 1
    if opts.model == 'CNN':
        # Make sure CNN has at least one linear layer, and the output size should be set to the number of classes.
        assert len(_CNNOPT.linear) > 0
        _CNNOPT.linear[-1] = opts.num_class
    _SaveOPT.model_dir = opts.model_dir
    result = model((train_feature, val_feature, train_reward, val_reward))
    # Save the estimated offloading reward.
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(opts.save_dir, 'estimate.npz'), **result)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('train_dir', help="Directory that saves the weak detector feature maps for the training set.")
    args.add_argument('val_dir', help="Directory that saves the weak detector feature maps for the validation set.")
    args.add_argument('train_label', help="Path to the offloading reward for the training set.")
    args.add_argument('val_label', help="Path to the offloading reward for the validation set.")
    args.add_argument('save_dir', help="Directory to save the estimated offloading reward.")
    args.add_argument('--num_class', type=int, default=50,
                      help="The number of classes to split the dataset into. Images are put to different classes " +
                           "on the rank of their offloading reward. A larger number of classes provides finer " +
                           "granularity, at the cost of higher model complexity and lower inference accuracy.")
    args.add_argument('--stage', type=int, default=23,
                      help="Stage number of the selected feature map. For yolov5 detectors, this should be a number " +
                           "between [0, 24]. Value between 0-23 stands for intermediate feature map from one of the " +
                           "hidden layer. 24 stands for feature extracted from detection output.")
    args.add_argument('--pool_size', type=int, default=8,
                      help="Size (H,W) of the feature maps after using RoI pooling. If 0, skip RoI pooling.")
    args.add_argument('--model', type=str, default='LR',
                      help="Type of the classification model. Available choices include 'CNN' (Convolutional Neural " +
                           "Network).")
    args.add_argument('--model_dir', type=str, default='', help="Directory to save the model weights.")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
