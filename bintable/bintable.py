import os
import pandas as pd
import numpy as np
import bintable.native


_ENDIANESS = "<"


def _preprocess_column(arr):
    new_dtype = None
    if arr.dtype.kind != "O":
        new_dtype = arr.dtype.newbyteorder(_ENDIANESS)
    return np.require(arr, dtype=new_dtype, requirements=['A'])


def _preprocess_dict(d):
    n_rows = None
    for col_name, array in d.items():
        if not isinstance(col_name, str):
            raise ValueError("Column {0} is not a string".format(col_name))

        if isinstance(array, (np.ndarray, np.generic)):
            if array.ndim != 1:
                raise ValueError(
                    "Array {0} is not one dimensional".format(col_name))
        else:
            raise ValueError(
                "Column {0} is not a numpy array".format(col_name))

        if n_rows is None:
            n_rows = len(array)
        elif len(array) != n_rows:
            raise ValueError("Not all columns have the same size")

        d[col_name] = _preprocess_column(array)

    return d


def _df_to_dict(df):
    return {col: df[col].values for col in df.columns}


def _check_file_exists(filename):
    if not os.path.exists(filename) and os.path.isfile(filename):
        raise IOError("File {0} doesn't exist".format(filename))


def _prepare_file_for_write(filename, append):
    if os.path.exists(filename):
        if not os.path.isfile(filename):
            raise IOError("{0} exists and is not file".format(filename))

    elif append:
        raise IOError(
            "{0} doesn't exist. Nothing to append to".format(filename))


def _write_table_wrapper(columns_dict, filename, append):
    columns_dict = _preprocess_dict(columns_dict)
    _prepare_file_for_write(filename, append)

    bintable.native.write_table(columns_dict, filename, append)


def _read_table_wrapper(filename):
    _check_file_exists(filename)
    return bintable.native.read_table(filename)


def write_table(df, filename, append=False):
    columns_dict = _df_to_dict(df)
    _write_table_wrapper(columns_dict, filename, append)


def write_dict(columns_dict, filename, append=False):
    columns_dict = dict(columns_dict)
    _write_table_wrapper(columns_dict, filename, append)


def read_dict(filename):
    return _read_table_wrapper(filename)


def read_table(filename):
    return pd.DataFrame(read_dict(filename))
