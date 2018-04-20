from bintable.native import write_table as write_table_native

def _preprocess_column(arr):
    return arr

def _df_to_dict(df):
    result = {}
    for col in df.columns:
        result[col] = _preprocess_column(df[col].values)
    return result



def write_table(df, filename, append=False):
    write_table_native(_df_to_dict(df), filename, append)
