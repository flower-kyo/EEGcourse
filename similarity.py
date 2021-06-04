import pandas as pd
import numpy as np
import os
data_dir = "/data/ZhangHongjun/codes/sleep/openpai/data/sleepedf/sleep-cassette/eeg_fpz_cz"

files = os.listdir(data_dir)
files.sort()

f1_matrix = np.zeros(shape=(len(files), len(files)))  # macro f1 score

"""
# 每次使用一晚睡眠作为训练数据，然后测试其他各晚睡眠的macro F1值
# 往f1_matrix里填入数据
# 保存数据到csv文件中
"""

df = pd.DataFrame(f1_matrix)
df.columns = files
df.index = files

with open("similarity.csv", 'w') as f:
    df.to_csv(f)


print(df)

