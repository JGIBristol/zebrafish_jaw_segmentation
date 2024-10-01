"""
For a toy set of data, find the loss function at
chance and perfect performance

"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from fishjaw.model import model


def _gen_img(rng: np.random.Generator, img_size: tuple[int, int, int]) -> torch.Tensor:
    """
    Generate a binary image, as onehot

    """
    # Generate binary image
    img = (rng.random(img_size) > 0.5).astype(np.int64)

    # Turn it into a tensor + add batch/channel dimensions
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)


def _to_prediction(img: torch.Tensor) -> torch.Tensor:
    """
    Add the right number of channels such that this represents a onehot prediction

    """
    return F.one_hot(img.squeeze(dim=0)).permute(
        0, 4, 1, 2, 3
    )  # pylint: disable=not-callable


def main():
    """
    Generate some toy data, get the loss function, find the loss at perfect
    performance, generate lots more toy data, find the loss at chance performance

    """
    rng = np.random.default_rng()

    loss_config = {
        "loss": "monai.losses.TverskyLoss",
        "loss_options": {
            "include_background": False,
            "to_onehot_y": True,
            "alpha": 0.5,
            "beta": 0.5,
            "sigmoid": True,
        },
    }
    loss = model.lossfn(loss_config)

    img_size = 255, 256, 257
    reference_image = _gen_img(rng, img_size)

    # Find the loss function for the reference image compared to itself
    reference_pred = _to_prediction(reference_image)
    print(f"Loss at perfect performance: {loss(reference_pred, reference_image)}")

    # For lots of random images, find the loss function compared to the reference image
    # this could be accelerated with numpy (or via GPU...?) but it's not necessary
    n_gen = 100
    losses = [None] * n_gen

    for i in trange(n_gen):
        img = _gen_img(rng, img_size)
        pred = _to_prediction(img)
        losses[i] = loss(pred, reference_image)

    print(f"Loss at chance performance: {np.mean(losses)}")

    fig, axis = plt.subplots()

    axis.hist(losses, bins=int(np.sqrt(n_gen)))
    axis.set_title(
        f"Loss at chance performance\n(n=100)\n{np.mean(losses):.5f}+-{np.std(losses):.5f}"
    )

    fig.tight_layout()
    fig.savefig("chance_loss.png")


if __name__ == "__main__":
    main()
