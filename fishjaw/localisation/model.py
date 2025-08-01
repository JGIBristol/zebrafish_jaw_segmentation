"""
Model arch and training loop

"""

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
