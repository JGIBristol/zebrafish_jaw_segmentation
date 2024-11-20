"""
Visualisation of things happening in the training process

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_losses(
    train_losses: list[list[float]], val_losses: list[list[float]]
) -> plt.Figure:
    """
    Plot the training and validation losses against epoch

    :param train_losses: list of lists of floats, the training losses for each epoch

    """
    assert len(train_losses) == len(val_losses)

    epochs = np.arange(len(train_losses))

    train_loss = np.array([np.mean(epoch_loss) for epoch_loss in train_losses])
    val_loss = np.array([np.mean(epoch_loss) for epoch_loss in val_losses])

    min_loss = min(np.min(train_loss), np.min(val_loss))
    log_train_loss = np.log(train_loss - min_loss + 1)
    log_val_loss = np.log(val_loss - min_loss + 1)

    fig, axis = plt.subplots()

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

    fig.tight_layout()
    return fig
