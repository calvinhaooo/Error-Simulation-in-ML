import pandas as pd
from pandas import read_csv, read_json, DataFrame
import sweetviz as sv
import json


file_path1 = 'data/df_modcloth.csv'
file_path2 = 'data/df_electronics.csv'
file_path3 = 'data/AMAZON_FASHION.csv'
file_path4 = 'data/modcloth_final_data.json'
file_path5 = 'data/renttherunway_final_data.json'

file_path = file_path5
# df = read_csv(file_path)
# df2 = read_csv(file_path2)
# df3 = read_csv(file_path3, names=['item', 'user', 'rating', 'timestamp'])
df = read_json(file_path, lines=True)

# 可以选择目标特征
my_report = sv.analyze(df)
# my_report = sv.compare(df, df2)
my_report.show_html()
