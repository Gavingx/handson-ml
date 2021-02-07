# -*- encoding: utf-8 -*-

import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master"
HOUSING_PATH = "D:/Gavin/Project/handson-ml/datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """获取数据url并解压到本地"""
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()



import pandas as pd


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing.describe())

print(housing["ocean_proximity"].value_counts())


# 图形展示
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show()


## 创建测试集

import numpy as np

np.random.seed(42)

#
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
#
#
# import hashlib
#
# def test_set_check(identifier, test_ratio, hash):
#     return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
#
# def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
#     return data.loc[~in_test_set], data.loc[in_test_set]
#
# housing_with_id = housing.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
#
#
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# # 创建新属性
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_household"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"] / housing["households"]
#
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))


# 分层抽样
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# 删除income_cat属性
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1,  inplace=True)


# 数据可视化
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"),
             colorbar=True)
plt.legend()
plt.show()

# 寻找相关性
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributest = ["median_house_value", "median_income", 'total_rooms',
               "housing_median_age"]
scatter_matrix(housing[attributest], figsize=(12, 8))
plt.show()


# 由图中可看到最有潜力能够预测房价中位数的属性时收入中位数，因此放大看看散点图
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.show()


# 数据准备
housing = strat_train_set.drop("median_house_value", axis=1) # drop()会创建一个数据副本，但不影响strat_train_set
housing_labels = strat_train_set["median_house_value"].copy()

# 处理缺失值
# housing.dropna(subset=["total_bedrooms"]) # total_bedrooms删除属性为缺失值所在的那一行
# housing.drop("total_bedrooms", axis=1) # 删除该属性
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median) # 缺失值填充中位数
# 利用inputer来处理缺失值
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
print(imputer.statistics_)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# 处理文本和分类属性
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder() # 能将文本label转化成数字
# housing_cat = housing["ocean_proximity"]
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# print(encoder.classes_)
#
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder() # 转化成one-hot向量
# housing_cat_1hot = encoder.fit_transform((housing_cat_encoded.reshape(-1, 1)))
# print(housing_cat_1hot)
#
from sklearn.preprocessing import LabelBinarizer
# encoder = LabelEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)


# 自定义转化器组合属性特征
from sklearn.base import BaseEstimator, TransformerMixin

rooms_idx, bedrooms_idx, population_idx, household_idx = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_idx]/X[:, household_idx]
        population_per_household = X[:, population_idx]/X[:, household_idx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_idx]/X[:, rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)



# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)
# print(housing_extra_attribs)


# 转化流水线
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler())
])
housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn_features.transformers import DataFrameSelector
from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attribs)),
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attribs)),
    ("label_binarizer", MyLabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
# 运行整条流水线
housing_prepared = full_pipeline.fit_transform(housing)


# 选择和训练模型

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: \t", lin_reg.predict(some_data_prepared))
print("Labels: \t\t", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("lin_rmse: \t", lin_rmse)

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse: \t", tree_rmse)


# 交叉验证
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=10, random_state=42)
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=kfold)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=kfold)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_reg_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=kfold)
forest_rmse = np.sqrt(-forest_reg_scores)
display_scores(forest_rmse)