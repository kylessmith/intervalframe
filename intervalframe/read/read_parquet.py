import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray
from typing import List


def read_parquet(filename: str,
                 index_col: List[str] = ["_starts","_ends","_labels"]) -> IntervalFrame:
    """
    Read parquet file
    
    Parameters
    ----------
        filename : str
            Path to parquet file
        index_col : List[str]
            Columns that define the index
    
    Returns
    -------
        IntervalFrame
    """
        
    # Read parquet
    df = pd.read_parquet(filename)
    
    # Get index
    index = LabeledIntervalArray()
    index.add(df.loc[:,index_col[0]].values,
            df.loc[:,index_col[1]].values,
            df.loc[:,index_col[2]].values)
    
    # Drop columns
    df.drop(columns=index_col, inplace=True)
    
    # Create IntervalFrame
    iframe = IntervalFrame(df=df, intervals=index)
    
    # Return
    return iframe