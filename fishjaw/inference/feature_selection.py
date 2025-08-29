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
    fish_ns = feature_df.index

    metadata_df = pd.DataFrame(columns=["age", "length", "genotype"], index=fish_ns)

    for n in fish_ns:
        meta = read.metadata(n)
        metadata_df.loc[n, "age"] = meta.age
        metadata_df.loc[n, "length"] = meta.length
        metadata_df.loc[n, "genotype"] = meta.genotype

    return pd.concat([feature_df, metadata_df], axis=1)
