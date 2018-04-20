import pandas as pd
import bintable
import timeit

df = pd.read_csv("/home/nikita.gryaznov/Documents/Work/Other/processor_test/output/feats_quantiles_75_80_85_90_95__aids_reports_02.csv")
#df = pd.DataFrame({'a': [1,2,3]})
bintable.write_table(df, "simple_write.bt")