import pandas as pd
import bintable
import timeit
import random
import numpy as np
import string
import unittest
import gc
import sys
import os

def random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def random_numeric_df(N,M):
    d = {random_string(50) : np.random.random(N) for _ in range(M)}
    return pd.DataFrame(d)

def is_nan(arr):
    return arr!=arr

def arrays_are_equal(arr1, arr2):
    return all((arr1 == arr2) | (is_nan(arr1)==is_nan(arr2)))
    
def write_load_table(d, append = False):
    write_func = bintable.write_dict if isinstance(d, dict) else bintable.write_table
    read_func = bintable.read_dict if isinstance(d, dict) else bintable.read_table

    filename = "temp.bt"
    write_func(d, filename, append=append)
    result = read_func(filename)
    os.remove(filename)
    return result

class TestBinaryTable(unittest.TestCase):

    def _compare_dfs(self, old_df, new_df):
        for col in old_df.columns:
            self.assertTrue(arrays_are_equal(old_df[col].values, new_df[col].values), col +" is not correct")

    def test_float_numeric(self):
        df = random_numeric_df(10000, 10)
        df_new = write_load_table(df)
        self._compare_dfs(df, df_new)

    def test_unicode(self):
        cols = {"COL": np.array(["х", "у", "й!"])}
        cols_new = write_load_table(cols)
        self.assertTrue(arrays_are_equal(cols["COL"], cols_new["COL"]), "Something wrong with unicode")

    def test_ascii(self):
        cols = {"COL": np.array(["a", "bc", "de", "124"], dtype='S')}
        cols_new =  write_load_table(cols)
        self.assertTrue(arrays_are_equal(cols["COL"], cols_new["COL"]), "Something wrong with unicode")

    def test_df_with_text(self):
        df = pd.DataFrame({"STRING_COL": ["fsdf", "bc", "de", np.nan], "NUM_COL": [1,2,3,4]})
        df_new =  write_load_table(df)
        
        self._compare_dfs(df, df_new)


    def test_df_with_random_objects(self):
        df = pd.DataFrame({"COL": [object(), object(), "aa", "bb"], "NUM_COL": [1,2,3,4]})
        
        with self.assertRaises(Exception):
            write_load_table(df)
        
    def test_refs_count_in_dict(self):
        cols = {"COL": np.array(["a", "bc", "de", "124"], dtype='S')}
        cols_new = write_load_table(cols)

        first_col = next(iter(cols_new.keys()))
        first_arr = next(iter(cols_new.values()))
        del cols_new
        gc.collect()
        self.assertEqual(sys.getrefcount(first_col), 2, "Column name has too many references")
        self.assertEqual(sys.getrefcount(first_arr), 2, "Array has too many references")
        
        

if __name__ == '__main__':
    unittest.main()