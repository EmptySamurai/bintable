import pandas as pd
import bintable
import timeit
import random
import numpy as np
import string

#df = pd.read_csv("/home/nikita.gryaznov/Documents/Work/Other/processor_test/output/feats_quantiles_75_80_85_90_95__aids_reports_02.csv")
def random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def random_df(N,M):
    d = {random_string(50) : np.random.random(N) for _ in range(M)}
    return pd.DataFrame(d)

df = random_df(10000, 10)
bintable.write_table(df, "simple_write.bt")
df_new = bintable.read_table("simple_write.bt")

for col in df.columns:
    print(col, all(df[col].values == df_new[col].values))