"""
使用butterworth滤波器对EEG数据进行滤波，并可视化滤波前后的波形
要求分解的波形：
https://gitee.com/kris_poul/imagebed/raw/master/imgbed/image-20210330174329109.png
上述链接图片中的的5个频段的波形。
参考链接：https://blog.csdn.net/weixin_45366564/article/details/104116985
参考链接：https://www.cnblogs.com/xiaosongshine/p/10831931.html
"""


import numpy as np
import os
data_dir = "/data/ZhangHongjun/codes/sleep/openpai/data/sleepedf/sleep-cassette/eeg_fpz_cz"

files = os.listdir(data_dir)
files.sort()

file = files[0]

file_path = os.path.join(data_dir, file)
npz_data = np.load(file_path)
signal = npz_data['x']
label = npz_data['y']
"""
# signal是一晚睡眠的数据，里面有841个样本。在里面取5个样本（每个睡眠时期取一个样本）出来做频段分解，并可视化。
"""

# 例：取一个W期的样本
w_signal = signal[0]

# 过滤出Delta波，Theta波的频段为小于4Hz，采用低通滤波滤波器

# 过滤出Theta波，Theta波的频段为4-7Hz，采用带通滤波器

# 以此类推 ...















print()

