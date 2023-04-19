import pandas as pd
import numpy as np


def write_parquet(iframe,
                  output,
                  **kwargs):
    """
    Write IntervalFrame to parquet

    Parameters
    ----------
        iframe : IntervalFrame
            IntervalFrame to write
        output : str
            Path to output file
        **kwargs : dict
            Keyword arguments to pass to to_parquet

    Returns
    -------
        None
    """

    # Write to parquet
    df = iframe.df
    df.loc[:,"_labels"] = iframe.index.labels
    df.loc[:,"_starts"] = iframe.index.starts
    df.loc[:,"_ends"] = iframe.index.ends
    df.to_parquet(output, **kwargs)

    # Drop columns
    df.drop(columns=["_labels", "_starts", "_ends"], inplace=True)