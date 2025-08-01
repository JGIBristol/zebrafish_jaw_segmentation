"""
Model arch and training loop

"""

import torch
from monai.networks.nets import AttentionUnet


def get_model(device) -> AttentionUnet:
    """
    Hard-coded architecture - I don't really care about squeezing performance
    out of this model, we just need it to give us a reasonable cropping window

    """
    return AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        strides=(2, 2, 2),
        channels=(4, 8, 16, 32),
        dropout=0.05,
    ).to(device)


def kl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence loss
    """
    # Apply log-softmax to predictions
    pred = torch.nn.functional.log_softmax(pred.view(pred.size(0), -1), dim=1)

    # Ensure target is normalized (if not already)
    target = target.view(pred.size(0), -1)
    target = target / target.sum(dim=1, keepdim=True)

    # Compute KL divergence
    return torch.nn.functional.kl_div(pred, target, reduction="batchmean")
