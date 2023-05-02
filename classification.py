import numpy as np
import argparse
import os
import time
import copy
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import lib.utils as ut
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
    model_idx: int = 1  # The index of model in cross validation.


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
        cls, scaler = pickle.load(open(os.path.join(save_opts.model_dir, f'wts{save_opts.model_idx}.pickle'), 'rb'))
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
    train_acc = np.sum(train_reward == train_est) / len(train_reward)
    val_acc = np.sum(val_reward == val_est) / len(val_reward)
    print(f"Trained {name} model with training accuracy: {train_acc:.3f}, validation accuracy: {val_acc:.3f}")
    # Save model if specified.
    if save_opts.save and save_opts.model_dir != '':
        Path(save_opts.model_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump((cls, scaler), open(os.path.join(save_opts.model_dir, f'wts{save_opts.model_idx}.pickle'), 'wb'))
    return {"train_est": train_est, "val_est": val_est, "train_time": train_time, "val_time": val_time}


@dataclass
class LROpt:
    """Options for the Logistic Regression model."""
    C: float = 1.0  # Inverse of regularization strength.


_LROPT = LROpt()


def fit_LR(data, opts=_LROPT):
    """Fit a Logistic Regression model."""
    model = LogisticRegression(C=opts.C, multi_class="multinomial")
    return fit_model(model, "Logistic Regression", data)


@dataclass
class RCOpt:
    """Options for the Ridge Classifier model."""
    alpha: float = 1.0  # Regularization strength.


_RCOPT = RCOpt()


def fit_RC(data, opts=_RCOPT):
    """Fit a Ridge Classifier model."""
    model = RidgeClassifier(alpha=opts.alpha)
    return fit_model(model, "Ridge Classifier", data)


@dataclass
class BNBOpt:
    """Options for the Naive Bayes classifier for multivariate Bernoulli models."""
    alpha: float = 1.0  # Additive (Laplace/Lidstone) smoothing parameter.


_BNBOPT = BNBOpt()


def fit_BNB(data, opts=_BNBOPT):
    """Fit a Bernoulli Naive Bayes model."""
    model = BernoulliNB(alpha=opts.alpha)
    return fit_model(model, "Bernoulli Naive Bayes", data)


def fit_GNB(data):
    """Fit a Gaussian Naive Bayes model."""
    model = GaussianNB()
    return fit_model(model, "Gaussian Naive Bayes", data)


@dataclass
class LSVCOpt:
    """Options for the Linear Support Vector Classification model."""
    C: float = 1.0  # Inverse of regularization strength.


_LSVCOPT = LSVCOpt()


def fit_LSVC(data, opts=_LSVCOPT):
    """Fit a Linear Support Vector Classification model."""
    model = LinearSVC(C=opts.C, multi_class="crammer_singer")
    return fit_model(model, "Linear Support Vector Classification", data)


@dataclass
class RFCOpt:
    """Options for the Random Forest Classifier model."""
    n_estimators: int = 100  # The number of trees in the forest.
    max_depth: int = 10  # The maximum depth of the tree.
    min_samples_split: int = 100  # The minimum number of samples required to split an internal node.


_RFCOPT = RFCOpt()


def fit_RFC(data, opts=_RFCOPT):
    """Fit a Random Forest Classifier model."""
    model = RandomForestClassifier(n_estimators=opts.n_estimators, max_depth=opts.max_depth,
                                   min_samples_split=opts.min_samples_split)
    return fit_model(model, "Random Forest Classifier", data)


@dataclass
class KNCOpt:
    """Options for the K-Neighbors Classifier model."""
    n_neighbors: int = 100  # Number of neighbors to use.


_KNCOPT = KNCOpt()


def fit_KNC(data, opts=_KNCOPT):
    """Fit a K-Neighbors Classifier model."""
    model = KNeighborsClassifier(n_neighbors=opts.n_neighbors)
    return fit_model(model, "K-Neighbors Classifier", data)


@dataclass
class CNNOpt:
    """Options for the Convolutional Neural Network model."""
    resize: bool = True  # Whether the inputs (feature maps extracted from the weak detector) have the same shape.
    learning_rate: float = 5e-3  # Initial learning rate.
    gamma: float = 0.1  # Scale for updating learning rate at each milestone.
    weight_decay: float = 1e-5  # Weight decay parameter for optimizer.
    milestones: List = field(default_factory=lambda: [])  # Epochs to update the learning rate.
    max_epoch: int = 100  # Maximum number of epochs for training.
    batch_size: int = 64  # Batch size for model training.
    channels: List = field(default_factory=lambda: [])  # Number of channels in each conv layer.
    kernels: List = field(default_factory=lambda: [])  # Kernel size for each conv layer.
    pools: List = field(default_factory=lambda: [])  # Whether max-pooling each conv layer.
    weight: List = field(default_factory=lambda: [])  # A manual rescaling weight given to each class.
    linear: List = field(
        default_factory=lambda: [145, 16, 16, 16, 16, 10])  # Number of features in each linear after the conv layers.
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
    best_model = copy.deepcopy(model)
    print(model)
    # Parse the model saving path.
    model_best_dir, model_last_dir = ut.parse_path(save_opts.model_dir)
    # Load weights if specified.
    if save_opts.load and save_opts.model_dir != '':
        model.load_state_dict(torch.load(os.path.join(model_last_dir, f'wts{save_opts.model_idx}.pth')))
    # Declare loss function, optimizer, and scheduler.
    if len(opts.weight) == 0:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(opts.weight, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=opts.gamma)
    # Save model if specified.
    model_save = (save_opts.save and save_opts.model_dir != '')
    if model_save:
        Path(model_best_dir).mkdir(parents=True, exist_ok=True)
        Path(model_last_dir).mkdir(parents=True, exist_ok=True)

    # Define the training and test function.
    def train(dataloader, model, loss_fn, optimizer):
        num_batches, size = len(dataloader), len(dataloader.dataset)
        model.train()
        train_loss, process = 0, 0
        for batch, (X, y) in enumerate(dataloader):
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

    # Function for estimating the offloading reward of both the training and validation set.
    def estimate(model, train_dl, val_dl):
        with torch.no_grad():
            train_est, val_est = list(), list()
            time1 = time.perf_counter()
            for X, y in train_dl:
                X, y = X.to(device), y.to(device)
                train_est.append(model(X).cpu().numpy())
            train_est = np.concatenate(train_est)
            time2 = time.perf_counter()
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                val_est.append(model(X).cpu().numpy())
            val_est = np.concatenate(val_est)
            time3 = time.perf_counter()
        train_time = (time2 - time1) / len(train_dl.dataset)
        val_time = (time3 - time2) / len(val_dl.dataset)
        train_est = np.argmax(train_est, axis=1)
        val_est = np.argmax(val_est, axis=1)
        return train_est, val_est, train_time, val_time

    # The training loop.
    best_test_err = np.inf
    train_loss, test_loss = list(), list()
    for t in range(opts.max_epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss.append(train(train_dataloader, model, loss_fn, optimizer))
        if t % opts.test_epoch == 0:
            test_loss_value = test(val_dataloader, model, loss_fn)
            # Save the current best version of the model.
            if test_loss_value < best_test_err:
                best_test_err = test_loss_value
                best_model = copy.deepcopy(model)
            test_loss.append(test_loss_value)
        scheduler.step()
    # Create a plot to visualize the training and test loss as a function of epoch number.
    if plot:
        CNN_plot(train_loss, test_loss, opts.test_epoch, opts.milestones, save_opts.model_idx)
    train_best_est, val_best_est, train_best_time, val_best_time = estimate(best_model, train_dataloader,
                                                                            val_dataloader)
    train_last_est, val_last_est, train_last_time, val_last_time = estimate(model, train_dataloader, val_dataloader)
    if model_save:
        torch.save(best_model.state_dict(), os.path.join(model_best_dir, f'wts{save_opts.model_idx}.pth'))
        torch.save(model.state_dict(), os.path.join(model_last_dir, f'wts{save_opts.model_idx}.pth'))
    train_acc = np.sum(train_reward == train_last_est) / len(train_reward)
    val_acc = np.sum(val_reward == val_last_est) / len(val_reward)
    print(f"Trained CNN model with training accuracy: {train_acc:.3f}, validation accuracy: {val_acc:.3f}")
    return {"train_est": train_best_est, "val_est": val_best_est, "train_time": train_best_time,
            "val_time": val_best_time}, {"train_est": train_last_est, "val_est": val_last_est,
                                         "train_time": train_last_time, "val_time": val_last_time}


def CNN_plot(train_loss, test_loss, test_epoch, lr_schedule, index):
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
    # Plot the smallest validation loss.
    min_idx = np.argmin(test_loss)
    ax.scatter(test_epoch * min_idx + 1, test_loss[min_idx], c='orange', s=200, zorder=3, label="min test error")
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
    plt.savefig(f'./cnn_training{index}.pdf', bbox_inches='tight')
    plt.show()
    return


def main(opts):
    # Load the weak detector feature maps.
    ifpool = opts.pool_size > 0 and opts.stage != 24
    feature_data = load_feature(opts.data_dir, opts.stage, pool=ifpool, size=opts.pool_size)
    # Load the offloading rewards.
    mapi_data = np.load(opts.mapi_path)['mapi']
    assert len(feature_data) == len(mapi_data), "Inconsistent number of feature maps and offloading rewards."
    # Load the dataset split.
    data_split = np.load(opts.split_path)
    assert len(mapi_data) == data_split.shape[1], "Inconsistent number of data points from the dataset and the split."
    # Select and fit the classification model.
    model_names = ['LR', 'RC', 'BNB', 'GNB', 'LSVC', 'RFC', 'KNC', 'CNN']
    models = [fit_LR, fit_RC, fit_BNB, fit_GNB, fit_LSVC, fit_RFC, fit_KNC, fit_CNN]
    try:
        model_idx = model_names.index(opts.model)
        model = models[model_idx]
    except ValueError:
        print("Please select a classification model from 'LR' (Logistic Regression), 'RC' (Ridge Classifier), " +
              "'BNB' (Bernoulli Naive Bayes), 'GNB' (Gaussian Naive Bayes), " +
              "'LSVC' (Linear Support Vector Classification), 'RFC' (Random Forest Classifier), " +
              "'KNC' (K-Neighbors Classifier), and 'CNN' (Convolutional Neural Network).")
    if opts.stage != 24:
        # Check if model and feature map selections are consistent.
        assert opts.model == 'CNN', "Only fully convolutional NN can take feature maps from hidden layers as inputs."
        if opts.pool_size == 0:
            # Force batch size to 1 when input feature maps have different shapes.
            _CNNOPT.resize = False
            _CNNOPT.batch_size = 1
    if opts.model == 'CNN':
        # Make sure CNN has at least one linear layer, and the output size should be set to the number of classes.
        assert len(_CNNOPT.linear) > 0
        _CNNOPT.linear[-1] = opts.num_class
        if opts.weight:
            _CNNOPT.weight = [x + 1 for x in range(opts.num_class)]
    _SaveOPT.model_dir = opts.model_dir
    # Cross validation.
    save_best_dir, save_last_dir = ut.parse_path(opts.save_dir)
    assert opts.num_class >= 2, "Please pick at least 2 classes to split offloading rewards."
    for cv_idx, val_mask in enumerate(data_split):
        # Split the dataset.
        train_feature = [f for f, v in zip(feature_data, val_mask) if not v]
        val_feature = [f for f, v in zip(feature_data, val_mask) if v]
        train_mapi = mapi_data[np.logical_not(val_mask)]
        val_mapi = mapi_data[val_mask]
        # Split the offloading rewards into classes.
        train_mapi_sorted = np.sort(train_mapi)
        threshold = [train_mapi_sorted[int(i / opts.num_class * (len(train_mapi) - 1))] for i in
                     range(opts.num_class + 1)]
        train_mapi_classed = np.zeros_like(train_mapi, dtype=int)
        val_mapi_classed = np.zeros_like(val_mapi, dtype=int)
        for idx, (thresh_low, thresh_high) in enumerate(zip(threshold[:-1], threshold[1:])):
            train_mapi_classed[np.logical_and(train_mapi >= thresh_low, train_mapi <= thresh_high)] = idx
            val_mapi_classed[np.logical_and(val_mapi >= thresh_low, val_mapi <= thresh_high)] = idx
        train_mapi = train_mapi_classed
        val_mapi = val_mapi_classed
        # Train the model.
        print(f"==============================Cross Validation Fold {cv_idx + 1}==============================")
        _SaveOPT.model_idx = cv_idx + 1
        result = model((train_feature, val_feature, train_mapi, val_mapi))
        # Save the estimated offloading reward.
        if opts.model != 'CNN':
            ut.save_result(opts.save_dir, result, cv_idx)
        else:
            ut.save_result(save_best_dir, result[0], cv_idx)
            ut.save_result(save_last_dir, result[1], cv_idx)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('data_dir', help="Directory that saves the weak detector feature maps.")
    args.add_argument('mapi_path', help="Path to the offloading reward (precomputed mAPI+).")
    args.add_argument('split_path', help="Path to the dataset split (for cross validation).")
    args.add_argument('save_dir', help="Directory to save the estimated offloading reward.")
    args.add_argument('--num_class', type=int, default=50,
                      help="The number of classes to split the dataset into. Images are put to different classes " +
                           "on the rank of their offloading reward. A larger number of classes provides finer " +
                           "granularity, at the cost of higher model complexity and lower inference accuracy.")
    args.add_argument('--weight', action='store_true',
                      help="Whether to apply a rescaling weight to each class when computing cross-entropy loss " +
                           "during training. Only active when the classification model is 'CNN'.")
    args.add_argument('--stage', type=int, default=23,
                      help="Stage number of the selected feature map. For yolov5 detectors, this should be a number " +
                           "between [0, 24]. Value between 0-23 stands for intermediate feature map from one of the " +
                           "hidden layer. 24 stands for feature extracted from detection output.")
    args.add_argument('--pool_size', type=int, default=8,
                      help="Size (H,W) of the feature maps after using RoI pooling. If 0, skip RoI pooling.")
    args.add_argument('--model', type=str, default='LR',
                      help="Type of the classification model. Available choices include 'LR' (Logistic Regression), " +
                           "'RC' (Ridge Classifier), 'BNB' (Bernoulli Naive Bayes), 'GNB' (Gaussian Naive Bayes), " +
                           "'LSVC' (Linear Support Vector Classification), 'RFC' (Random Forest Classifier), " +
                           "'KNC' (K-Neighbors Classifier), 'CNN' (Convolutional Neural Network).")
    args.add_argument('--model_dir', type=str, default='', help="Directory to save the model weights.")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
