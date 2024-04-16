import pandas as pd
import numpy as np


def write_text(iframe,
                  output,
                  **kwargs):
    """
    Write IntervalFrame to text file

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
    df.loc[:,"labels"] = iframe.index.labels
    df.loc[:,"starts"] = iframe.index.starts
    df.loc[:,"ends"] = iframe.index.ends
    df.to_csv(output, header=True, index=True, sep="\t", **kwargs)

    # Drop columns
    df.drop(columns=["labels", "starts", "ends"], inplace=True)