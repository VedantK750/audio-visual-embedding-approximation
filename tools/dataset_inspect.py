import pandas as pd

df = pd.read_csv("vggsound.csv")

# Column C = third column (index 2)
freq = df.iloc[:, 2].value_counts()
freq.to_csv("class_frequencies.csv")