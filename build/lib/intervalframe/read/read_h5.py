import os
import glob
import numpy as np
import pandas as pd
import h5py
from ailist import IntervalArray, LabeledIntervalArray
from intervalframe import IntervalFrame


def read_h5_DataFrame(h5_group):
    """
    Read pandas.DataFrame from h5py group

    Parameters
    ----------
        h5_group : h5py.Group
            h5py.Group

    Returns
    -------
        df : pandas.DataFrame
            pandas.DataFrame
    """

    # Define dtype categories
    numeric_dtypes = set(['i','u','b','c'])
    string_dtypes = set(['O','U','S'])
    
    # Record axis
    axis = int(np.array(h5_group["axis"]))
    shape = np.array(h5_group["shape"])
    
    # Read index
    if axis == 0:
        df = pd.DataFrame.from_records(np.array(h5_group["values"]), index="index")
        if df.index.dtype.kind in string_dtypes:
            df.index = df.index.values.astype(str)
        
    # Read columns
    elif axis == 1:
        df = pd.DataFrame.from_records(np.array(h5_group["values"]), index="index").T
        if df.columns.dtype.kind in string_dtypes:
            df.columns = df.columns.values.astype(str)

    # Convert numpy object to str
    for i, dtype in enumerate(df.dtypes):
        if dtype.kind == "O":
            df.iloc[:,i] = df.iloc[:,i].values.astype(str)
        
    return df


def read_h5_intervalframe(h5_group):
    """
    Read from IntervalFrame to h5py group

    Parameters
    ----------
        h5_group : h5py.Group
            h5py.group

    Returns
    -------
        None
    """
    
    # Determine type
    if np.array(h5_group["intervals"]["iframe_type"]) == b"labeled":
        # Read LabeledIntervalArray
        n_intervals = h5_group["intervals"]["starts"].shape[0]
        intervals = LabeledIntervalArray()
        intervals.from_array(np.array(h5_group["intervals"]["starts"]),
                             np.array(h5_group["intervals"]["ends"]),
                             np.array(h5_group["intervals"]["labels"]).astype(str))
                                             
        # Read DataFrame
        df = read_h5_DataFrame(h5_group["data_frame"])
        
        # Create IntervalFrame
        iframe = IntervalFrame(intervals=intervals, df=df)
                                             
    else:
        # Read IntervalArray
        n_intervals = h5_group["intervals"]["starts"].shape[0]
        intervals = IntervalArray()
        intervals.from_array(np.array(h5_group["intervals"]["starts"]),
                             np.array(h5_group["intervals"]["ends"]))
                                         
        # Read DataFrame
        df = read_h5_DataFrame(h5_group["data_frame"])
        
        # Create IntervalFrame
        iframe = IntervalFrame(intervals=intervals, df=df)
        
    return iframe
    
    
def read_bed(filename):
    """
    Read from bed formatted file
    
    Parameters
    ----------
        filename : str
            Name of the file
    
    Returns
    -------
        iframe : IntervalFrame
            Intervals from bed file
    """
    
    # Initialize IntervalFrame
    iframe = IntervalFrame()
    
    # Iterate over bed file
    for line in open(filename, "r"):
        fields = line.strip().split("\t")
        
        iframe.add(int(fields[1]), int(fields[2]), label=fields[0])
        
    return iframe
    
    
    
    
    