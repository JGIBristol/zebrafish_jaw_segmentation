"""
Investigate the effect of removing layers and/or skip connections on the segmentation

"""

import argparse


def main(*, subject: int, model_name: str, threshold: float):
    """
    Load in a model and the data, then evaluate with and without each layer/skip connection

    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference with and without attention"
    )

    parser.add_argument(
        "subject",
        nargs="?",
        help="The subject to perform inference on. Defaults to using the test data.",
        choices={273, 274, 218, 219, 120, 37},
        type=int,
        default=None,
    )
    parser.add_argument(
        "model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarising the output (needed for meshing)",
    )
    main(**vars(parser.parse_args()))
