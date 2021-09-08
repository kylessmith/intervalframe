import os
import glob
import numpy as np
import pandas as pd
import h5py
from ailist import IntervalArray, LabeledIntervalArray


def convert_recarray(rec_array):
    """
    Convert the dtypes of np.recarray to h5py compatible dtypes

    Parameters
    ----------
        rec_array
            numpy.recarray (Array to convert dtypes for)

    Returns
    -------
        rec_array
            numpy.recarray (Array with h5py compatible dtypes)
    """

    dtypes = {o:rec_array.dtype[o].str for o in rec_array.dtype.fields}
    for key_dtype in dtypes:
        if dtypes[key_dtype] == "|O":
            lengths = []
            for i in range(len(rec_array[key_dtype])):
                if isinstance(rec_array[key_dtype][i], str):
                    lengths.append(len(rec_array[key_dtype][i]))
                elif pd.isnull(rec_array[key_dtype][i]):
                    continue
                elif isinstance(rec_array[key_dtype][i], int):
                    continue
                elif isinstance(rec_array[key_dtype][i], float):
                    continue
                else:
                    continue
            max_length = max(lengths) if len(lengths) > 0 else 0
            if max_length > 0:
                dtypes[key_dtype] = "<S{}".format(max_length)
            else:
                dtypes[key_dtype] = "<f8"

    rec_array = rec_array.astype(list(dtypes.items()))

    return rec_array


def write_h5_DataFrame(h5_group, df, axis=0):
    """
    Write pandas.DataFrame to h5py group

    Parameters
    ----------
        h5_group
            h5py.group
        df
            pandas.DataFrame
        axis
            int

    Returns
    -------
        None
    """
    
    # Record axis
    h5_group["axis"] = axis
    h5_group["shape"] = np.array(df.shape)
    
    # Iterate over columns
    if axis == 1:
        if df.index.dtype.kind == "i":
            index_dtypes = "i"
        else:
            index_dtypes = "<S{}".format(df.index.str.len().max())
        if df.columns.dtype.kind == "i":
            columns_dtypes = "<S{}".format(df.columns.str.len().max())
        else:
            columns_dtypes = "<S{}".format(df.columns.str.len().max())

        rec_array = df.T.to_records(index_dtypes=index_dtypes)

        rec_array = convert_recarray(rec_array)

        h5_group["values"] = rec_array
    
    # Iterate over index
    if axis == 0:
        if df.index.dtype.kind == "i":
            index_dtypes = "i"
        else:
            index_dtypes = "<S{}".format(df.index.str.len().max())
        if df.columns.dtype.kind == "i":
            columns_dtypes = "i"
        else:
            columns_dtypes = "<S{}".format(df.columns.str.len().max())

        rec_array = df.to_records(index_dtypes=index_dtypes)

        rec_array = convert_recarray(rec_array)

        h5_group["values"] = rec_array
        
        
def write_h5_intervalframe(iframe, h5_group):
    """
    Write IntervalFrame to h5py group

    Parameters
    ----------
        iframe : IntervalFrame
            Annotated intervals
        h5_group : h5py.Group
            h5py.group

    Returns
    -------
        None
    """
    
    # Write intervals
    h5_intervals = h5_group.create_group("intervals")
    h5_intervals["starts"] = iframe.starts()
    h5_intervals["ends"] = iframe.ends()
    
    # Is multi
    if isinstance(iframe.index, LabeledIntervalArray):
        h5_intervals["iframe_type"] = "labeled"
        h5_intervals["labels"] = iframe.index.extract_labels()

    # Is single
    else:
        h5_intervals["iframe_type"] = "unlabeled"
        
    # Write DataFrame
    h5_df = h5_group.create_group("data_frame")
    write_h5_DataFrame(h5_df, iframe.df)
    