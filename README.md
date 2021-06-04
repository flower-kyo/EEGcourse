注意！ 之前的数据集不再使用。需要重新下载以下数据集。

在服务器中执行以下操作

下载数据集:

```
wget http://211.71.76.25:3033/edf39_data.tar.gz
```

解压:

```
tar xzvf edf39_data.tar.gz
```

运行代码:

```
python main_supervised.py --data_dir data/sleepedf/sleep-cassette/eeg_fpz_cz
```

