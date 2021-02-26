# -*- encoding: utf-8 -*-

"""
@Author: Gavin
@File: titanic_prediction.py
@Software: Pycharm
@Time: 2021/2/19
@Desc:
    1.kaggle入门比赛
    2.泰坦尼克号存活乘客预测
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import set_option
from pathlib import Path


project_path = Path(__file__).parent.parent


# 数据集
if not os.path.exists(str(Path(project_path, "datasets/train.csv"))) or \
    not os.path.exists(str(Path(project_path, "datasets/test.csv"))):

    raise FileNotFoundError(
        "数据集不存在，请检查：{}".format(
            str(Path(project_path, "datasets/titanic.zip"))))


# 加载数据集
train_file = pd.read_csv(str(Path(project_path, "datasets/train.csv")))
test_file = pd.read_csv(str(Path(project_path, "datasets/test.csv")))
set_option("display.width", 200)
set_option("display.max_columns", 20)
print("train csv head: \n", train_file.head())
print("test csv head: \n", test_file.head())

# 查看数据结构
print("train csv info: \n", train_file.info())
print("test csv info: \n", test_file.info())
print("train csv describe\n", train_file.describe())
print("test csv  describe\n", test_file.describe())
"""
可看出训练集中age、cabin、embarked有空值，测试集中age、fare、cabin有空值
标签中正负样本比例11:7，可能有不均衡的情况出现
Sex、Ticket、carbin、Embarked需要转化成数值特征
需要对数值特征做标准化或归一化处理
"""

# 创建验证集


# 画图展示
train_file.hist(bins=20, figsize=(20, 15))
plt.show()


