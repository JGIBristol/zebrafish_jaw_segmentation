"""
Compare a selected slice through the segmentation between humans and the model

"""

from fishjaw.util import files


def main():
    """
    Read images from the RDSF and the model from disk, perform inference
    then plot slices
    """
    out_dir = files.script_out_dir() / "compare_slices"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    
    # load model
    # load input image
    # perform inference
    # load human segmentations
    # plot slices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a selected slice through the segmentation between humans and the model"
    )
    parser.add_argument(
        "model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
        default="paper_model.pkl",
    )

    main(**vars(parser.parse_args()))
