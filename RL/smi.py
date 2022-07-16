import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/simoncraf/skarb/main/stocks/smi_new_data.csv'
df = pd.read_csv(url)
df = df.set_index('Date')
#euro = np.ones(df.shape[0])
df = df.sum(axis=1)
print(df)

