import pandas as pd
import bintable
import numpy as np

df = pd.DataFrame({"STRING_COL": ["fsdf", "bc", "de", np.nan], "NUM_COL": [1,2,3,4]})
bintable.write_table(df, "test.bt")
df_new = bintable.read_table("test.bt")

