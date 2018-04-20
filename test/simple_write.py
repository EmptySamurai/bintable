import pandas as pd
import bintable

df = pd.DataFrame({'a':[1,2,3]})
bintable.write_table(df, "simple_write.bt")