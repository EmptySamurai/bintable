import bintable.native
import pandas as pd
import os
import sys
import gc

def _preprocess_column(arr):
    return arr

def _df_to_dict(df):
    result = {}
    for col in df.columns:
        result[col] = _preprocess_column(df[col].values)
    return result



def write_table(df, filename, append=False):
    if not append and (os.path.exists(filename)):
        os.remove(filename)
    bintable.native.write_table(_df_to_dict(df), filename, append)

def write_dict(columns_dict, filename, append=False):
    if not append and (os.path.exists(filename)):
        os.remove(filename)
    new_dict = {col:_preprocess_column(val) for col, val in columns_dict.items()}
    bintable.native.write_table(new_dict, filename, append)

def read_dict(filename):
    return bintable.native.read_table(filename)

def read_table(filename):
    return pd.DataFrame(read_dict(filename))
    
