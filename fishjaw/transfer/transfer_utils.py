"""
Utilities
"""

import torch

from monai.networks.nets.attentionunet import AttentionLayer

from fishjaw.model import data, model


def attn_unet_param_type_regex() -> dict[str, str]:
    """
    Regex matching the various weight types for the attention u net
    """
    return {
        "down_conv_0_weight": r".*conv.0.conv.weight",
        "down_conv_0_bias": r".*conv.0.conv.bias",
        "down_conv_1_weight": r".*conv.1.conv.weight",
        "down_conv_1_bias": r".*conv.1.conv.bias",
        "down_adn_0_weight": r".*conv.0.adn.N.weight",
        "down_adn_0_bias": r".*conv.0.adn.N.bias",
        "down_adn_1_weight": r".*conv.1.adn.N.weight",
        "down_adn_1_bias": r".*conv.1.adn.N.bias",
        "attention_wg_0_weight": r".*attention.W_g.0.conv.weight",
        "attention_wg_0_bias": r".*attention.W_g.0.conv.bias",
        "attention_wg_1_weight": r".*attention.W_g.1.weight",
        "attention_wg_1_bias": r".*attention.W_g.1.bias",
        "attention_wx_0_weight": r".*attention.W_x.0.conv.weight",
        "attention_wx_0_bias": r".*attention.W_x.0.conv.bias",
        "attention_wx_1_weight": r".*attention.W_x.1.weight",
        "attention_wx_1_bias": r".*attention.W_x.1.bias",
        "attention_psi_0_weight": r".*attention.psi.0.conv.weight",
        "attention_psi_0_bias": r".*attention.psi.0.conv.bias",
        "attention_psi_1_weight": r".*attention.psi.1.weight",
        "attention_psi_1_bias": r".*attention.psi.1.bias",
        "upconv_weight": r".*upconv.up.conv.weight",
        "upconv_bias": r".*upconv.up.conv.bias",
        "upconv_adn_weight": r".*upconv.up.adn.N.weight",
        "upconv_adn_bias": r".*upconv.up.adn.N.bias",
        "merge_weight": r".*merge.conv.weight",
        "merge_bias": r".*merge.conv.bias",
        "merge_adn": r".*merge.adn.A.weight",
    }


def fine_tune_model(
    config: dict,
    model_name: str,
    data_config: data.DataConfig,
    train_layers: str = list[int],
    lr_multiplier: float = 0.1,
    epochs_frozen: int = 150,
    epochs_unfrozen: int = 50,
) -> tuple[torch.nn.Module, list[list[float]], list[list[float]]]:
    """
    Fine-tune a model on the provided data

    :param config: the configuration dictionary from userconf.yml
    :param model_name: name of the model to fine tune (will be read from disk)
    :param data_config: the data to use for training
    :param train_layers: a list of integers: the layers to train
    :param lr_multiplier: multiplier for the learning rate, compared to the value in config
    :param epochs_frozen: number of epochs to train with frozen layers
    :param epochs_unfrozen: number of epochs to train with all layers unfrozen
    """
    for l in train_layers:
        assert isinstance(l, int), f"Expected int, got {type(l)}"
        assert 0 <= l <= 5

    # Load the model from disk fresh so that we don't overwrite anything in memory
    new_model = model.load_model(model_name)
    net = new_model.load_model(set_eval=False)
    net.to(config["device"])

    # Freeze all the parameters
    for param in net.parameters():
        param.requires_grad = False

    # Get the bits of the model
    # Head, encoder-decoder, final convolution
    _, encdec, _ = net.model

    def unfreeze_attention_layers(
        module: torch.nn.Module, current_depth: int, target_depths: list[int]
    ):
        """
        Recursively unfreeze AttentionLayer parameters at specified depths
        """
        if isinstance(module, AttentionLayer):
            if current_depth in target_depths:
                for name, param in module.named_parameters():
                    if "submodule" not in name:
                        param.requires_grad = True

            # Recurse into submodule
            if hasattr(module, "submodule"):
                unfreeze_attention_layers(
                    module.submodule, current_depth + 1, target_depths
                )

        # Handle Sequential containers (like in submodule)
        elif isinstance(module, torch.nn.Sequential):
            for child in module.children():
                unfreeze_attention_layers(child, current_depth, target_depths)

    unfreeze_attention_layers(encdec, current_depth=0, target_depths=train_layers)

    # Create a new optimiser that only updates the unfrozen layers
    # Get the right optimiser from the config
    # and set the learning rate to a lower value
    optimiser = getattr(torch.optim, config["optimiser"])(
        (p for p in net.parameters() if p.requires_grad),
        lr=config["learning_rate"] * lr_multiplier,
    )

    # Create a loss function
    loss = model.lossfn(config)

    # Train the model with the frozen layers
    train_config = model.TrainingConfig(
        config["device"],
        epochs_frozen,
        torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=config["lr_lambda"]),
    )

    net, train_losses, val_losses = model.train(
        net, optimiser, loss, data_config, train_config
    )

    # Train the whole model

    return
