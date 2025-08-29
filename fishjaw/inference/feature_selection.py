"""
Tools for performing feature selection once we've segmented the jaws
"""
import pandas as pd

from . import read

def add_metadata_cols(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns describing fish metadata to a feature dataframe.

    Adds metadata including fish age, length and mutation status
    (wt/het/hom/mosaic). Assumes the dataframe is indexed by fish number.

    :param feature_df: features, indexed by fish number
    :returns: the same dataframe, with metadata columns added to the right
              of existing columns

    """
