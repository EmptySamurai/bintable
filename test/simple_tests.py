import pandas as pd
import bintable
import timeit
import random
import numpy as np
import string
import unittest

def random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def random_numeric_df(N,M):
    d = {random_string(50) : np.random.random(N) for _ in range(M)}
    return pd.DataFrame(d)



class TestBinaryTable(unittest.TestCase):

    def test_numeric(self):
        df = random_numeric_df(10000, 10)
        bintable.write_table(df, "simple_write.bt")
        df_new = bintable.read_table("simple_write.bt")

        for col in df.columns:
            self.assertTrue(all(df[col].values == df_new[col].values), col +" is not correct")

    def test_unicode(self):
        cols = {"COL": np.array(["х", "у", "й!"])}
        bintable.write_dict(cols, "unicode_test.bt")
        cols_new = bintable.read_table("unicode_test.bt")
        self.assertTrue(all(cols["COL"] == cols_new["COL"]), "Something wrong with unicode")

    def test_ascii(self):
        cols = {"COL": np.array(["a", "bc", "de", "124"], dtype='S')}
        bintable.write_dict(cols, "ascii_test.bt")
        cols_new = bintable.read_table("ascii_test.bt")
        self.assertTrue(all(cols["COL"] == cols_new["COL"]), "Something wrong with unicode")

if __name__ == '__main__':
    unittest.main()