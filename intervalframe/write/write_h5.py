import os
import glob
import numpy as np
import pandas as pd
import h5py
from ailist import IntervalArray, LabeledIntervalArray
from pandas.api.types import is_float_dtype, is_integer_dtype, is_sparse


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


def write_h5_DataFrame(h5_group: h5py.Group,
                       df: pd.DataFrame,
                       axis: int = 0,
                       compression_opts: int = 4) -> None:
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

    # Determine is DataFrame in all float or integer
    if is_float_dtype(df.values) or is_integer_dtype(df.values):
        rec_array = df.values
        h5_group["typing"] = "numeric"
        # Write index
        index = df.index.values
        if index.dtype.str.startswith("|S")  or index.dtype.str.startswith("|O"):
            index = index.astype(bytes)
        h5_group["index"] = index
        # Write columns
        columns = df.columns.values
        if columns.dtype.str.startswith("|S") or columns.dtype.str.startswith("|O"):
            columns = columns.astype(bytes)
        h5_group["columns"] = columns
    
    # Iterate over columns
    elif axis == 1:
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
        h5_group["typing"] = "recarray"

    
    # Iterate over index
    elif axis == 0:
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
        h5_group["typing"] = "recarray"

    h5_group_values = h5_group.create_dataset("values", data=rec_array, compression="gzip",
                                              compression_opts=compression_opts, shape=rec_array.shape)

    return None
        
        
def write_h5_intervalframe(iframe, h5_group, compression_opts=4):
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
    #h5_intervals["starts"] = iframe.starts()
    #h5_intervals["ends"] = iframe.ends()
    if iframe.shape[0] > 0:
        h5_intervals_starts = h5_intervals.create_dataset("starts", data=iframe.starts(), compression="gzip",
                                                        compression_opts=compression_opts)
        h5_intervals_end = h5_intervals.create_dataset("ends", data=iframe.ends(), compression="gzip",
                                                    compression_opts=compression_opts)

        # Is multi
        if isinstance(iframe.index, LabeledIntervalArray):
            h5_intervals["iframe_type"] = "labeled"
            h5_intervals["labels"] = iframe.index.labels

        # Is single
        else:
            h5_intervals["iframe_type"] = "unlabeled"
            
        # Write DataFrame
        has_sparse = np.array([is_sparse(iframe.df[col]) for col in iframe.df.columns.values])
        if has_sparse.any():
            h5_df = h5_group.create_group("sparse_frame")
            write_h5_sparse_frame(h5_df, iframe.df.loc[:,has_sparse])

            h5_df = h5_group.create_group("data_frame")
            write_h5_DataFrame(h5_df, iframe.df.loc[:,~has_sparse], compression_opts=compression_opts)

            h5_group["total_columns"] = iframe.df.columns.values.astype(bytes)
        else:
            h5_df = h5_group.create_group("data_frame")
            write_h5_DataFrame(h5_df, iframe.df, compression_opts=compression_opts)
    

def write_h5_sparse_frame(h5_group, sframe, compression_opts=4):
    """
    Write SparseFrame to h5py group

    Parameters
    ----------
        h5_group : h5py.Group
            h5py.group
        sframe : SparseFrame
            Annotated intervals

    Returns
    -------
        None
    """

    from pandas.core.dtypes.cast import find_common_type

    dtype = find_common_type(sframe.sparse._parent.dtypes.to_list())
    if isinstance(dtype, pd.SparseDtype):
        dtype = dtype.subtype

    cols, rows, data = [], [], []
    for col, (_, ser) in enumerate(sframe.sparse._parent.items()):
        sp_arr = ser.array
        #if sp_arr.fill_value != 0:
        #    raise ValueError("fill value must be 0 when converting to COO matrix")

        row = sp_arr.sp_index.indices
        cols.append(np.repeat(col, len(row)))
        rows.append(row)
        data.append(sp_arr.sp_values.astype(dtype, copy=False))

    cols = np.concatenate(cols)
    rows = np.concatenate(rows)
    data = np.concatenate(data)
    
    # Write intervals
    #h5_intervals = h5_group.create_group("intervals")
    #h5_intervals["starts"] = sframe.starts
    #h5_intervals["ends"] = sframe.ends
    #h5_intervals["labels"] = sframe.index.labels

    # Write DataFrame
    h5_group["cols"] = cols
    h5_group["rows"] = rows
    h5_group["data"] = data
    h5_group["columns"] = sframe.columns.values.astype(bytes)