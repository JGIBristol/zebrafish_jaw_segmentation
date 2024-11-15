"""
Perform an ablation study on the requested data.

This will run the model on some data with and without the attention mechanism enabled-
we replace the attention block with the identity function, effectively disabling it.

Then makes some plots showing the results with and without the attention mechanism.

"""

import argparse

from fishjaw.model import model


def main(args: argparse.Namespace):
    """
    Get the model and data and run it with and without the attention mechanism.

    """
    # Load the model and training-time config
    model_state = model.load_model(args.model_name)
    config = model_state.config

    net = model_state.load_model(set_eval=True)
    net.to("cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
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

    main(parser.parse_args())
