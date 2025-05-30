"""
Utilities
"""


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
