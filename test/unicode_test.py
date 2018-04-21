import pandas as pd
import bintable
import numpy as np


cols = {"COL": np.array(["х", "у", "й!"])}
bintable.write_dict(cols, "unicode_test.bt")
cols_new = bintable.read_table("unicode_test.bt")
print(all(cols["COL"] == cols_new["COL"]))

