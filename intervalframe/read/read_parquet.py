import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray


def read_parquet(filename: str):
    """
    Read parquet file
    
    Parameters
    ----------
        filename : str
            Path to parquet file
    
    Returns
    -------
        IntervalFrame
    """
        
    # Read parquet
    df = pd.read_parquet(filename)
    
    # Get index
    index = LabeledIntervalArray()
    index.add(df.loc[:,"_starts"].values,
              df.loc[:,"_ends"].values,
              df.loc[:,"_labels"].values)
    
    # Drop columns
    df.drop(columns=["_labels", "_starts", "_ends"], inplace=True)
    
    # Create IntervalFrame
    iframe = IntervalFrame(df=df, intervals=index)
    
    # Return
    return iframe