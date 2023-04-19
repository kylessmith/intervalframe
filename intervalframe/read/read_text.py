import os
import glob
import gzip
import numpy as np
import pandas as pd
from ailist import IntervalArray, LabeledIntervalArray
from intervalframe import IntervalFrame
import h5py
import requests
import io


def read_bed(bed_file,
             header=None,
             skipfirst=False,
             is_url=False):
    """
    Read BED formated file

    Parameters
    ----------
        bed_file : str
            Path to BED file
        header : str, optional
            Path to header file
        skipfirst : bool, optional
            Skip first line
        is_url : bool, optional
            If True, read from URL

    Returns
    -------
        intervals : IntervalFrame
            IntervalFrame frm BED file

    """

    # Initialize intervals
    intervals = LabeledIntervalArray()
    
    # Determine if gzipped
    gzipped = False
    if bed_file.endswith(".gz"):
        gzipped = True
    
    # Determine if URL
    if is_url:
        web_response = requests.get(bed_file, stream=True)
        csv_gz_file = web_response.content
        filename = io.BytesIO(csv_gz_file)
    else:
        filename = bed_file
    
    # Determine number of fields
    if gzipped:
        o = gzip.open(filename, 'r')
        if skipfirst:
            skipline = o.readline()
        n_fields = len(o.readline().strip().split(b"\t"))
    else:
        o = open(filename, "r")
        if skipfirst:
            skipline = o.readline()
        n_fields = len(o.readline().strip().split("\t"))
    o.close()

    # Determine if df present
    read_df = False
    if n_fields > 3:
        read_df = True
        cols = np.arange(n_fields - 3) + 3
    
    # Open file
    if is_url:
        filename = io.BytesIO(csv_gz_file)
    if gzipped:
        o = gzip.open(filename, 'r')
    else:
        o = open(filename, "r")

    # If skip first row
    if skipfirst:
        skipline = o.readline()

    # Read intervals
    if header is not None:
        header_line = o.readline()
    for line in o:
        if gzipped:
            line = line.decode()
        fields = line.strip().split("\t")
        intervals.add(int(fields[1]), int(fields[2]), fields[0])
    o.close()
    
    # Read df
    df = None
    if read_df:
        df = pd.read_csv(bed_file, header=header, index_col=None, sep="\t",
                         usecols=cols, skiprows=int(skipfirst))

    # Create IntervalFrame
    iframe = IntervalFrame(intervals, df)

    return iframe