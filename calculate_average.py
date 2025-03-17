import pandas as pd

file = "evaluation/result/LibreLog with gpt-4o-2024-08-06.csv"
df = pd.read_csv(file)

average_values = df.iloc[:, 1:].mean()
new_row = ['Average'] + average_values.tolist()
df.loc[len(df)] = new_row
df.to_csv(file, index=False)