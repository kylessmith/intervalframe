import os
import glob
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix
from ailist import IntervalArray, LabeledIntervalArray
from intervalframe import IntervalFrame


def read_h5_DataFrame(h5_group: h5py.Group) -> pd.DataFrame:
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
    df_typing = str(np.array(h5_group["typing"]).astype(str))

    # If numeric
    if df_typing == "numeric":
        index = np.array(h5_group["index"])
        if index.dtype.str.startswith("|S") or index.dtype.str.startswith("|O"):
            index = index.astype(str)
        columns = np.array(h5_group["columns"])
        if columns.dtype.str.startswith("|S") or columns.dtype.str.startswith("|O"):
            columns = columns.astype(str)
        df = pd.DataFrame(np.array(h5_group["values"]),
                          index = np.array(h5_group["index"]).astype(str),
                          columns = np.array(h5_group["columns"]).astype(str))
    
    # Read index
    elif axis == 0:
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

    # Determine if empty
    if len(list(h5_group["intervals"].keys())) > 0:
    
        # Determine type
        if np.array(h5_group["intervals"]["iframe_type"]) == b"labeled":
            # Read LabeledIntervalArray
            #n_intervals = h5_group["intervals"]["starts"].shape[0]
            intervals = LabeledIntervalArray()
            intervals.add(np.array(h5_group["intervals"]["starts"]),
                          np.array(h5_group["intervals"]["ends"]),
                          np.array(h5_group["intervals"]["labels"]).astype(str))
                                                
        else:
            # Read IntervalArray
            #n_intervals = h5_group["intervals"]["starts"].shape[0]
            intervals = IntervalArray()
            intervals.add(np.array(h5_group["intervals"]["starts"]),
                          np.array(h5_group["intervals"]["ends"]))
                                            
        # Read DataFrame
        if "sparse_frame" in list(h5_group.keys()):
            sdf = read_h5_sparse(h5_group["sparse_frame"])
            ddf = read_h5_DataFrame(h5_group["data_frame"])
            columns = np.array(h5_group["total_columns"]).astype(str)
            sdf.index = range(sdf.shape[0])
            ddf.index = range(ddf.shape[0])
            df = pd.concat([ddf, sdf], axis=1)
            df = df.loc[:,columns]
        else:
            df = read_h5_DataFrame(h5_group["data_frame"])
        
        # Create IntervalFrame
        iframe = IntervalFrame(intervals=intervals, df=df)

    else:
        iframe = IntervalFrame()
        
    return iframe


def from_spmatrix(data, index=None, columns=None) -> pd.DataFrame:
        """
        Create a new DataFrame from a scipy sparse matrix.
        .. versionadded:: 0.25.0
        Parameters
        ----------
        data : scipy.sparse.spmatrix
            Must be convertible to csc format.
        index, columns : Index, optional
            Row and column labels to use for the resulting DataFrame.
            Defaults to a RangeIndex.
        Returns
        -------
        DataFrame
            Each column of the DataFrame is stored as a
            :class:`arrays.SparseArray`.
        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.eye(3)
        >>> pd.DataFrame.sparse.from_spmatrix(mat)
             0    1    2
        0  1.0  0.0  0.0
        1  0.0  1.0  0.0
        2  0.0  0.0  1.0
        """
        from pandas._libs.sparse import IntIndex

        from pandas import DataFrame

        data = data.tocsc()
        index, columns = pd.DataFrame.sparse._prep_index(data, index, columns)
        n_rows, n_columns = data.shape
        # We need to make sure indices are sorted, as we create
        # IntIndex with no input validation (i.e. check_integrity=False ).
        # Indices may already be sorted in scipy in which case this adds
        # a small overhead.
        data.sort_indices()
        indices = data.indices
        indptr = data.indptr
        array_data = data.data
        dtype = pd.SparseDtype(array_data.dtype, np.nan)
        arrays = []
        for i in range(n_columns):
            sl = slice(indptr[i], indptr[i + 1])
            idx = IntIndex(n_rows, indices[sl], check_integrity=False)
            arr = pd.arrays.SparseArray._simple_new(array_data[sl], idx, dtype)
            arrays.append(arr)
        return DataFrame._from_arrays(
            arrays, columns=columns, index=index, verify_integrity=False
        )


def read_h5_sparse(h5_group):
    """
    Read from Sparse to h5py group

    Parameters
    ----------
        h5_group : h5py.Group
            h5py.group

    Returns
    -------
        None
    """

    data = np.array(h5_group["data"])
    cols = np.array(h5_group["cols"])
    rows = np.array(h5_group["rows"])
    columns = np.array(h5_group["columns"]).astype(str)

    sp_matrix = csr_matrix((data, (rows, cols)))
    sp_df = from_spmatrix(sp_matrix, columns=columns)

    return sp_df
    
    
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
    
    
    
    
    