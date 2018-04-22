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
    n_rows = None
    for col_name, array in d.items():
        if not isinstance(col_name, str):
            raise ValueError("Column {0} is not a string".format(col_name))

        if isinstance(array, (np.ndarray, np.generic)):
            if array.ndim != 1:
                raise ValueError("Array {0} is not one dimensional".format(col_name))
        else:
            raise ValueError("Column {0} is not a numpy array".format(col_name))
        
        if n_rows is None:
            n_rows = len(array)
        elif len(array) != n_rows:
            raise ValueError("Not all columns have the same size")

        d[col_name] = _preprocess_column(array)

    return d


def _df_to_dict(df):
    return {col : df[col].values for col in df.columns}

def _check_file_exists(filename):
    if not os.path.exists(filename) and os.path.isfile(filename):
        raise IOError("File {0} doesn't exist".format(filename))

def _prepare_file_for_write(filename, append):
    if os.path.exists(filename):
        if not os.path.isfile(filename):
            raise IOError("{0} exists and is not file".format(filename))
        
        if not append:
            os.remove(filename)



def write_table(df, filename, append=False):
    _prepare_file_for_write(filename, append)

    df_dict = _preprocess_dict(_df_to_dict(df))
    bintable.native.write_table(df_dict, filename, append)


def write_dict(columns_dict, filename, append=False):
    _prepare_file_for_write(filename, append)

    columns_dict = dict(columns_dict)
    bintable.native.write_table(columns_dict, filename, append)


def read_dict(filename):
    _check_file_exists(filename)
    return bintable.native.read_table(filename)


def read_table(filename):
    _check_file_exists(filename)
    return pd.DataFrame(read_dict(filename))
