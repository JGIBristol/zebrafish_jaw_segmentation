"""
Visualisation of things happening in the training process

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_loss_axis(
    axis: plt.Axes, train_losses: list[list[float]], val_losses: list[list[float]]
) -> None:
    """
    Plot on an axis
    """
    assert len(train_losses) == len(val_losses)

    epochs = np.arange(len(train_losses))

    train_loss = np.array([np.mean(epoch_loss) for epoch_loss in train_losses])
    val_loss = np.array([np.mean(epoch_loss) for epoch_loss in val_losses])

    axis.plot(epochs, train_loss, label="Train")

    # Find quartiles - the mean might be outside this, which would be interesting wouldn't it
    train_loss_upper = [np.percentile(epoch_loss, 75) for epoch_loss in train_losses]
    train_loss_lower = [np.percentile(epoch_loss, 25) for epoch_loss in train_losses]
    axis.fill_between(epochs, train_loss_lower, train_loss_upper, alpha=0.5, color="C0")

    axis.plot(epochs, val_loss, label="Validation")

    val_loss_upper = [np.percentile(epoch_loss, 75) for epoch_loss in val_losses]
    val_loss_lower = [np.percentile(epoch_loss, 25) for epoch_loss in val_losses]
    axis.fill_between(epochs, val_loss_lower, val_loss_upper, alpha=0.5, color="C1")

    axis.set_title("Loss")
    axis.set_xlabel("Epoch")
    axis.legend()


def plot_losses(
    train_losses: list[list[float]], val_losses: list[list[float]]
) -> matplotlib.figure.Figure:
    """
    Plot the training and validation losses against epoch

    :param train_losses: list of lists of floats, the training losses for each epoch

    """
    fig, axis = plt.subplots()

    plot_loss_axis(axis, train_losses, val_losses)

    fig.tight_layout()
    return fig
