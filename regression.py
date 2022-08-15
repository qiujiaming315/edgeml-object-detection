import numpy as np
import argparse
import os
import time
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as mse

from lib.data import load_feature
from lib.nn_model import EdgeDetectionDataset, EdgeDetectionNet

"""Train a regression model that maps the weak detector's intermediate feature map to the offloading reward."""


@dataclass
class SaveOpt:
    """Options for loading/saving model weights."""
    model_dir: str = ''  # Directory to save the model weights.
    load: bool = False  # If model is loaded from pre-trained weights.
    save: bool = True  # If model weights need to be saved after training.


_SaveOPT = SaveOpt()


def fit_model(model, name, data, save_opts=_SaveOPT):
    """
    Fit a (non-CNN based) regression model that predicts offloading reward based on weak detector feature map.
    :param model: the regression model.
    :param name: name of the regression model.
    :param data: features (inputs) and rewards (labels) for training and validation.
    :param save_opts: model saving options.
    :return: the estimated offloading reward for the training and validation dataset,
             and the average inference time for each image.
    """
    # Retrieve data and flatten the feature maps.
    train_feature, val_feature, train_reward, val_reward = data
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    # Load model if specified.
    if save_opts.load and save_opts.model_dir != '':
        reg, scaler = pickle.load(open(os.path.join(save_opts.model_dir, 'wts.pickle'), 'rb'))
    else:
        reg = model.fit(train_feature, train_reward)
        scaler = StandardScaler().fit(train_feature)
    # Normalize the data.
    train_feature = scaler.transform(train_feature)
    val_feature = scaler.transform(val_feature)
    # Make predictions for both the training and test set.
    time1 = time.perf_counter()
    train_est = reg.predict(train_feature)
    time2 = time.perf_counter()
    val_est = reg.predict(val_feature)
    time3 = time.perf_counter()
    train_time, val_time = (time2 - time1) / len(train_reward), (time3 - time2) / len(val_reward)
    train_mse, val_mse = mse(train_reward, train_est), mse(val_reward, val_est)
    print(f"Trained {name} model with training MSE:{train_mse}, validation MSE:{val_mse}")
    # Save model if specified.
    if save_opts.save and save_opts.model_dir != '':
        Path(save_opts.model_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump((reg, scaler), open(os.path.join(save_opts.model_dir, 'wts.pickle'), 'wb'))
    return {"train_est": train_est, "val_est": val_est, "train_time": train_time, "val_time": val_time}


def fit_LR(data):
    """Fit a linear regression model."""
    model = LinearRegression()
    return fit_model(model, "Linear Regression", data)


@dataclass
class ENOpt:
    """Options for the Elastic net regression model."""
    alpha: float = 1.0  # Constant that multiplies the penalty terms.
    l1_ratio: float = 0.5  # The ElasticNet mixing parameter.


_ENOPT = ENOpt()


def fit_EN(data, opts=_ENOPT):
    """Fit an elastic net model."""
    model = ElasticNet(alpha=opts.alpha, l1_ratio=opts.l1_ratio)
    return fit_model(model, "Elastic Net", data)


@dataclass
class BROpt:
    """Options for the Bayesian ridge regression model."""
    alpha_1: float = 1e-6  # Shape parameter for the Gamma distribution prior over the alpha parameter.
    alpha_2: float = 1e-6  # Rate parameter for the Gamma distribution prior over the alpha parameter.
    lambda_1: float = 1e-6  # Shape parameter for the Gamma distribution prior over the lambda parameter.
    lambda_2: float = 1e-6  # Rate parameter for the Gamma distribution prior over the lambda parameter.


_BROPT = BROpt()


def fit_BR(data, opts=_BROPT):
    """Fit a Bayesian ridge regression model."""
    model = BayesianRidge(alpha_1=opts.alpha_1, alpha_2=opts.alpha_2, lambda_1=opts.lambda_1, lambda_2=opts.lambda_2)
    return fit_model(model, "Bayesian Ridge", data)


@dataclass
class SVROpt:
    """Options for the support vector regression model."""
    kernel: str = 'rbf'  # Specifies the kernel type to be used in the algorithm. Choose from 'linear', 'poly', 'rbf',
    # 'sigmoid', 'precomputed'.
    C: float = 1.0  # Regularization parameter.
    epsilon: float = 0.1  # Epsilon in the epsilon-SVR model.


_SVROPT = SVROpt()


def fit_SVR(data, opts=_SVROPT):
    """Fit a support vector regression model."""
    model = SVR(kernel=opts.kernel, C=opts.C, epsilon=opts.epsilon)
    return fit_model(model, "Support Vector Regression", data)


@dataclass
class GBROpt:
    """Options for the Gradient Boosting regression model."""
    learning_rate: float = 0.1
    n_estimators: int = 1000  # The number of boosting stages to perform.
    subsample: float = 1.0  # The fraction of samples to be used for fitting the individual base learners.


_GBROPT = GBROpt()


def fit_GBR(data, opts=_GBROPT):
    """Fit a Gradient Boosting Regressor."""
    model = GradientBoostingRegressor(learning_rate=opts.learning_rate, n_estimators=opts.n_estimators,
                                      subsample=opts.subsample)
    return fit_model(model, "Gradient Boosting Regressor", data)


@dataclass
class CNNOpt:
    """Options for the Gradient Boosting regression model."""
    learning_rate: float = 1e-3  # Initial learning rate.
    gamma: float = 0.1  # Scale for updating learning rate at each milestone.
    milestones: List = field(default_factory=lambda: [10, 15, 20])  # Epochs to update the learning rate.
    max_epoch: int = 25  # Maximum number of epochs for training.
    batch_size: int = 64  # Batch size for model training.
    channels: List = field(default_factory=lambda: [256, 128, 32])  # Number of channels in each conv layer.
    kernels: List = field(default_factory=lambda: [3, 3])  # Kernel size for each conv layer.
    pools: List = field(default_factory=lambda: [True, True])  # Whether max-pooling each conv layer.
    linear: List = field(default_factory=lambda: [])  # Number of features in each linear after the conv layers.
    test_epoch: int = 5  # Number of epochs for periodic test using the validation set.


_CNNOPT = CNNOpt()


def fit_CNN(data, opts=_CNNOPT, save_opts=_SaveOPT):
    """Fit a Convolutional Neural Network to predict offloading reward."""
    # Import pytorch.
    import torch
    from torch.utils.data import DataLoader
    # Prepare the dataset.
    train_feature, val_feature, train_reward, val_reward = data
    train_data = EdgeDetectionDataset(train_feature, train_reward)
    val_data = EdgeDetectionDataset(val_feature, val_reward)
    train_dataloader = DataLoader(train_data, batch_size=opts.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=opts.batch_size)
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # Build the CNN model.
    model = EdgeDetectionNet(opts.channels, opts.kernels, opts.pools, opts.linear).to(device)
    print(model)
    # Load weights if specified.
    if save_opts.load and save_opts.model_dir != '':
        model.load_state_dict(torch.load(os.path.join(save_opts.model_dir, 'wts.pth')))
    # Declare loss function, optimizer, and scheduler.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=0.1)

    # Define the training and test function.
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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

    # The training loop.
    for t in range(opts.max_epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        if t % opts.test_epoch == 0:
            test(val_dataloader, model, loss_fn)
        scheduler.step()
    # Estimate the offloading reward for both training and validation set.
    with torch.no_grad():
        train_est, val_est = list(), list()
        time1 = time.perf_counter()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            train_est.append(model(X).cpu().numpy())
        train_est = np.concatenate(train_est).flatten()
        time2 = time.perf_counter()
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            val_est.append(model(X).cpu().numpy())
        val_est = np.concatenate(val_est).flatten()
        time3 = time.perf_counter()
    train_time = (time2 - time1) / len(train_dataloader.dataset)
    val_time = (time3 - time2) / len(val_dataloader.dataset)
    # Save model if specified.
    if save_opts.save and save_opts.model_dir != '':
        Path(save_opts.model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_opts.model_dir, 'wts.pth'))
    return {"train_est": train_est, "val_est": val_est, "train_time": train_time, "val_time": val_time}


def main(opts):
    # Load the weak detector feature maps for the training and validation dataset.
    ifpool = opts.pool_size > 0
    train_feature = load_feature(opts.train_dir, opts.stage, pool=ifpool, size=opts.pool_size)
    val_feature = load_feature(opts.val_dir, opts.stage, pool=ifpool, size=opts.pool_size)
    # Load the offloading rewards for the training dataset.
    train_reward = np.load(opts.train_label)['mapi']
    val_reward = np.load(opts.val_label)['mapi']
    assert len(train_feature) == len(
        train_reward), "Inconsistent number of training feature maps and offloading rewards."
    assert len(val_feature) == len(
        val_reward), "Inconsistent number of validation feature maps and offloading rewards."
    # Select and fit the regression model.
    model_names = ['LR', 'EN', 'BR', 'SVR', 'GBR', 'CNN']
    models = [fit_LR, fit_EN, fit_BR, fit_SVR, fit_GBR, fit_CNN]
    try:
        model_idx = model_names.index(opts.model)
        model = models[model_idx]
    except ValueError:
        print("Please select a regression model from 'LR' (Linear Regression), 'EN' (Elastic Net), " +
              "'BR' (Bayesian Ridge), 'SVR' (Support Vector Regression), 'GBR' (Gradient Boosting Regressor), " +
              "and 'CNN' (Convolutional Neural Network).")
    if not ifpool:
        # Check if model selection is consistent with pooling decision.
        assert opts.model == 'CNN', "Only fully convolutional NN can take input with different shapes. " + \
                                    "Please set model to 'CNN' if you choose to skip the RoI pooling step."
        # Force batch size to 1 and set no linear layers when input feature maps have different shapes.
        _CNNOPT.batch_size = 1
        _CNNOPT.linear = []
    _SaveOPT.model_dir = opts.model_dir
    result = model((train_feature, val_feature, train_reward, val_reward))
    # Save the estimated offloading reward.
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(opts.save_dir, f'{opts.model}_estimate.npz'), **result)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('train_dir', help="Directory that saves the weak detector feature maps for the training set.")
    args.add_argument('val_dir', help="Directory that saves the weak detector feature maps for the validation set.")
    args.add_argument('train_label', help="Path to the offloading reward for the training set.")
    args.add_argument('val_label', help="Path to the offloading reward for the validation set.")
    args.add_argument('save_dir', help="Directory to save the estimated offloading reward.")
    args.add_argument('--stage', type=int, default=23,
                      help="Stage number of the selected feature map. For yolov5 detectors, " +
                           "this should be a number between [0, 23].")
    args.add_argument('--pool_size', type=int, default=8,
                      help="Size (H,W) of the feature maps after using RoI pooling. If 0, skip RoI pooling.")
    args.add_argument('--model', type=str, default='LR',
                      help="Type of the regression model. Available choices include 'LR' (Linear Regression), " +
                           "'EN' (Elastic Net), 'BR' (Bayesian Ridge), 'SVR' (Support Vector Regression), " +
                           "'GBR' (Gradient Boosting Regressor), and 'CNN' (Convolutional Neural Network).")
    args.add_argument('--model_dir', type=str, default='', help="Directory to save the model weights.")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
