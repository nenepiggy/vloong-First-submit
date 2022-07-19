# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 21:07:32 2022

@author: 13106
"""

import numpy as np
import scipy.io as scio
import h5py
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression

def mape(y_true,y_pred):
    output_errors = np.average(np.abs(y_pred - y_true)/y_true)
    return output_errors

def build_feature_df(batch_dict):
    """
    建立一个DataFrame，包含加载的批处理字典中所有最初使用的特性
    """

    # print("Start building features ...")

    # 124 cells (3 batches)
    n_cells = len(batch_dict.keys())

    ## Initializing feature vectors:
    # numpy vector with 124 zeros
    cycle_life = np.zeros(n_cells)
    # 1. delta_Q_100_10(V)
    minimum_dQ_100_10 = np.zeros(n_cells)
    variance_dQ_100_10 = np.zeros(n_cells)
    skewness_dQ_100_10 = np.zeros(n_cells)
    kurtosis_dQ_100_10 = np.zeros(n_cells)

    # dQ_100_10_2 = np.zeros(n_cells)
    # 2. Discharge capacity fade curve features
    slope_lin_fit_2_100 = np.zeros(
        n_cells)  # Slope of the linear fit to the capacity fade curve, cycles 2 to 100
    intercept_lin_fit_2_100 = np.zeros(
        n_cells)  # Intercept of the linear fit to capavity face curve, cycles 2 to 100
    discharge_capacity_2 = np.zeros(n_cells)  # Discharge capacity, cycle 2
    diff_discharge_capacity_max_2 = np.zeros(n_cells)  # Difference between max discharge capacity and cycle 2
    discharge_capacity_100 = np.zeros(n_cells)  # for Fig. 1.e
    slope_lin_fit_95_100 = np.zeros(n_cells)  # for Fig. 1.f
    # 3. Other features
    mean_charge_time_2_6 = np.zeros(n_cells)  # Average charge time, cycle 1 to 5
    minimum_IR_2_100 = np.zeros(n_cells)  # Minimum internal resistance

    diff_IR_100_2 = np.zeros(n_cells)  # Internal resistance, difference between cycle 100 and cycle 2

    # Classifier features
    minimum_dQ_5_4 = np.zeros(n_cells)
    variance_dQ_5_4 = np.zeros(n_cells)
    cycle_550_clf = np.zeros(n_cells)

    # iterate/loop over all cells.
    for i, cell in enumerate(batch_dict.values()):
        cycle_life[i] = cell['cycle_life']
        # 1. delta_Q_100_10(V)
        c10 = cell['cycles']['10']
        c100 = cell['cycles']['100']
        dQ_100_10 = c100['Qdlin'] - c10['Qdlin']

        minimum_dQ_100_10[i] = np.log10(np.abs(np.min(dQ_100_10)))
        variance_dQ_100_10[i] = np.log(np.abs(np.var(dQ_100_10)))
        skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10)))
        kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10)))

        # Qdlin_100_10 = cell['cycles']['100']['Qdlin'] - cell['cycles']['10']['Qdlin']
        # dQ_100_10_2[i] = np.var(Qdlin_100_10)

        # 2. Discharge capacity fade curve features
        # Compute linear fit for cycles 2 to 100:
        q = cell['summary']['QD'][1:100].reshape(-1, 1)  # discharge cappacities; q.shape = (99, 1);
        X = cell['summary']['cycle'][1:100].reshape(-1, 1)  # Cylce index from 2 to 100; X.shape = (99, 1)
        linear_regressor_2_100 = LinearRegression()
        linear_regressor_2_100.fit(X, q)

        slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
        intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
        discharge_capacity_2[i] = q[0][0]
        diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]

        discharge_capacity_100[i] = q[-1][0]

        q95_100 = cell['summary']['QD'][94:100].reshape(-1, 1)
        q95_100 = q95_100 * 1000  # discharge cappacities; q.shape = (99, 1);
        X95_100 = cell['summary']['cycle'][94:100].reshape(-1,
                                                           1)  # Cylce index from 2 to 100; X.shape = (99, 1)
        linear_regressor_95_100 = LinearRegression()
        linear_regressor_95_100.fit(X95_100, q95_100)
        slope_lin_fit_95_100[i] = linear_regressor_95_100.coef_[0]

        # 3. Other features
        mean_charge_time_2_6[i] = np.mean(cell['summary']['chargetime'][1:6])
        minimum_IR_2_100[i] = np.min(cell['summary']['IR'][1:100])
        diff_IR_100_2[i] = cell['summary']['IR'][100] - cell['summary']['IR'][1]

        # Classifier features
        c4 = cell['cycles']['4']
        c5 = cell['cycles']['5']
        dQ_5_4 = c5['Qdlin'] - c4['Qdlin']
        minimum_dQ_5_4[i] = np.log10(np.abs(np.min(dQ_5_4)))
        variance_dQ_5_4[i] = np.log10(np.var(dQ_5_4))
        cycle_550_clf[i] = cell['cycle_life'] >= 550

    # combining all featues in one big matrix where rows are the cells and colums are the features
    # note last two variables below are labels/targets for ML i.e cycle life and cycle_550_clf
    features_df = pd.DataFrame({
        "cell_key": np.array(list(batch_dict.keys())),
        "minimum_dQ_100_10": minimum_dQ_100_10,
        "variance_dQ_100_10": variance_dQ_100_10,
        "skewness_dQ_100_10": skewness_dQ_100_10,
        "kurtosis_dQ_100_10": kurtosis_dQ_100_10,
        "slope_lin_fit_2_100": slope_lin_fit_2_100,
        "intercept_lin_fit_2_100": intercept_lin_fit_2_100,
        "discharge_capacity_2": discharge_capacity_2,
        "diff_discharge_capacity_max_2": diff_discharge_capacity_max_2,
        "mean_charge_time_2_6": mean_charge_time_2_6,
        "minimum_IR_2_100": minimum_IR_2_100,
        "diff_IR_100_2": diff_IR_100_2,
        "minimum_dQ_5_4": minimum_dQ_5_4,
        "variance_dQ_5_4": variance_dQ_5_4,
        "cycle_life": cycle_life,
        "cycle_550_clf": cycle_550_clf
    })

    # print("Done building features")
    return features_df

def train_val_split(features_df, regression_type="full", model="regression"):
    """
    划分train&test数据集。注意：数据集要按照指定方式划分
    :param features_df: 包含最初使用的特性dataframe
    :param regression_type: 回归模型的类型
    :param model: 使用模型的flag
    """
    # only three versions are allowed.
    assert regression_type in ["full", "variance", "discharge"]

    # dictionary to hold the features indices for each model version.
    features = {
        "full": [1, 2, 5, 6, 7, 9, 10, 11],
        "variance": [2],
        # "discharge": [1, 2, 3, 4, 7, 8]
        "discharge": [1, 2, 5 , 9, 13]
    }
    
    # features = {
    #     "full": [0, 1, 4, 5, 6, 8, 9, 10],
    #     "variance": [1],
    #     "discharge": [0, 1, 2, 3, 6, 7]
    # }
    # get the features for the model version (full, variance, discharge)
    feature_indices = features[regression_type]
    # get all cells with the specified features
    model_features = features_df.iloc[:, feature_indices]
    # get last two columns (cycle life and classification)
    labels = features_df.iloc[:, -2:]
    # labels are (cycle life ) for regression other wise (0/1) for classsification
    labels = labels.iloc[:, 0] if model == "regression" else labels.iloc[:, 1]

    # split data in to train/primary_test/and secondary test
    train_cells = np.arange(1, 84, 2)
    val_cells = np.arange(0, 84, 2)
    val_cells = np.delete(val_cells,np.where(val_cells==42)[0])
    test_cells = np.arange(84, 124, 1)

    # get cells and their features of each set and convert to numpy for further computations
    x_train = np.array(model_features.iloc[train_cells])
    x_val = np.array(model_features.iloc[val_cells])
    x_test = np.array(model_features.iloc[test_cells])

    # target values or labels for training
    y_train = np.array(labels.iloc[train_cells])
    y_val = np.array(labels.iloc[val_cells])
    y_test = np.array(labels.iloc[test_cells])

    # return 3 sets
    return {"train": (x_train, y_train), "val": (x_val, y_val), "test": (x_test, y_test)}

def my_train(dataset,alpha_train,l1_ratio,log_target,normal_not,normal_inform):
    x_train,y_train = dataset.get("train")
    # 构造一个模型实例
    regr = ElasticNet(random_state=4, alpha=alpha_train, l1_ratio=l1_ratio)
    # 是否要对标签取log
    y_train = np.log(y_train) if log_target else y_train
    # 归一化
    if normal_not:
        y_train = (y_train - normal_inform[1]) / (normal_inform[0] - normal_inform[1])
    # 拟合模型
    regr.fit(x_train, y_train)
    return regr

def my_eval(dataset,model,log_target,normal_not,normal_inform):
    x_train,y_train = dataset.get("train")
    x_val,y_val = dataset.get('val')
    x_test,y_test = dataset.get('test')
    # 用训练集测试模型
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    # 反归一化
    if normal_not:
        pred_train = pred_train*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
        pred_val = pred_val*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
        pred_test = pred_test*(normal_inform[0] - normal_inform[1]) + normal_inform[1]
    # 看看是否要变回来
    if log_target:
        pred_train = np.exp(pred_train)
        pred_val = np.exp(pred_val)
        pred_test = np.exp(pred_test)
    # 求误差
    error_train = mape(y_train, pred_train) * 100
    error_val = mape(y_val, pred_val) * 100
    error_test = mape(y_test, pred_test) * 100
    
    return error_train,error_val,error_test

dataset = np.load('my_data.npy',allow_pickle=True).item()
# function to build features for ML
features_df = build_feature_df(dataset) # 从batch字典数据中提取特征矩阵

normal_or_not = True
log_target = True

if normal_or_not:
    features_df.iloc[:, 1:-2] = features_df.iloc[:, 1:-2].apply(lambda x: (x-x.min())/(x.max()-x.min()) if x.max()!=x.min() else 1, axis=0) # 对输入做归一化

if log_target:
    normal_max_min = (np.log(features_df.iloc[:,-2]).max(),np.log(features_df.iloc[:,-2]).min())
else:
    normal_max_min = (features_df.iloc[:,-2].max(),features_df.iloc[:,-2].min())
    
battery_dataset = train_val_split(features_df, 'discharge') # 将特征矩阵分为训练和测试集

error_best = 50
for alpha in np.logspace(-3,-1,50):
    for lr in np.linspace(0.1,1,10):
        my_model = my_train(battery_dataset,alpha_train=alpha,l1_ratio=lr,log_target=log_target,normal_not=normal_or_not,normal_inform = normal_max_min)
        error_train,error_val,error_test = my_eval(battery_dataset,my_model,log_target,normal_or_not,normal_max_min)
        if error_val < error_best:
            error_best = error_val
            error_train_best = error_train
            error_val_best = error_val
            error_test_best = error_test
            best_model = my_model
            parameter_best = [alpha,lr]
print(f"Regression Error (Train): {error_train_best}%")
print(f"Regression Error (Val): {error_val_best}%")
print(f"Regression Error (Test): {error_test_best}%")
print(best_model.coef_)

# alpha = 0.0001
# lr = 0.2

# for index, row in features_df.iteritems():
#     if (index != 'cell_key') and (index != 'cycle_life') and (index != 'cycle_550_clf'):
#         print(f"{index}: {np.corrcoef(row,np.log(features_df['cycle_life']))[0,1]}")

# my_model = my_train(battery_dataset,alpha_train=alpha,l1_ratio=lr,log_target=log_target,normal_not=normal_or_not,normal_inform = normal_max_min)
# error_train,error_val,error_test = my_eval(battery_dataset,my_model,log_target,normal_or_not,normal_max_min)
# error_train_best = error_train
# error_val_best = error_val
# error_test_best = error_test

# print(f"Regression Error (Train): {error_train_best}%")
# print(f"Regression Error (Val): {error_val_best}%")
# print(f"Regression Error (Test): {error_test_best}%")
# print(my_model.coef_)