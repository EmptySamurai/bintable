import bintable.native
import pandas as pd
import os
import sys
import gc
import numpy as np

_ENDIANESS = "<"

def _preprocess_column(arr):
    return np.require(arr, dtype=arr.dtype.newbyteorder(_ENDIANESS), requirements=['A'])

def _preprocess_dict(d):
    for key, array in d.items():
        d[key] = _preprocess_column(array)

    return d


def _df_to_dict(df):
    return {col : df[col].values for col in df.columns}


def write_table(df, filename, append=False):
    if not append and (os.path.exists(filename)):
        os.remove(filename)

    df_dict = _preprocess_dict(_df_to_dict(df))
    bintable.native.write_table(df_dict, filename, append)


def write_dict(columns_dict, filename, append=False):
    if not append and (os.path.exists(filename)):
        os.remove(filename)

    columns_dict = columns_dict.copy() 
    bintable.native.write_table(columns_dict, filename, append)


def read_dict(filename):
    return bintable.native.read_table(filename)


def read_table(filename):
    return pd.DataFrame(read_dict(filename))
