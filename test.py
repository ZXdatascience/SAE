from deepautoencoder import StackedAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf
from sklearn import preprocessing
from hpelm import ELM

def add_lag(cons, thres=0.1):
    cons = np.array(cons)
    pa = pacf(cons, nlags=150)
    above_thres_indices = np.argwhere(pa > 0.1)
    above_thres_indices = np.delete(above_thres_indices, 0)
    max_lag = max(above_thres_indices)
    data = []
    labels = []
    for i in range(max_lag, len(cons) - 1):
        new_indices = i - above_thres_indices
        new_series = cons[new_indices]
        data.append(new_series)
        labels.append(cons[i])
    return np.array(data), np.array(labels)


def mean_filter(cons, window_size=2):
    res = []
    for i in range(len(cons) - window_size + 1):
        res.append(np.array(cons[i: i+window_size]).mean())
    return res


def to_30_min(cons):
    i = 0
    res = []
    while i + 1 < len(cons) :
        res.append(cons[i] + cons[i+1])
        i += 2
    return res


# mnist = input_data.read_data_sets(, one_hot=True)
# data, target = mnist.train.images, mnist.train.labels
train_data_name = "data/building1retail.csv"
df = pd.read_csv(train_data_name)
electricity_cons = df['Power (kW)'].values
mean_filtered_cons = mean_filter(electricity_cons)
min30_filtered = to_30_min(mean_filtered_cons)
data, labels = add_lag(min30_filtered)

data_scalar = preprocessing.MinMaxScaler()
labels_scalar = preprocessing.MinMaxScaler()
data = data_scalar.fit_transform(data)
labels = labels_scalar.fit_transform(labels.reshape(-1, 1))
# train / test  split
idx = np.random.rand(data.shape[0]) < 0.8
train_X, train_Y = data[idx], labels[idx]
test_X, test_Y = data[~idx], labels[~idx]
sae_model = StackedAutoEncoder(dims=[50, 50], activations=['sigmoid', 'sigmoid'], epoch=[10000, 10000],
                               noise='mask-0.5', loss='sparse', lr=0.007, batch_size=100, print_step=200)
test_X = test_X[test_Y.reshape(-1) != 0]
test_Y = test_Y[test_Y != 0].reshape(-1, 1)
train_x_ae_output = sae_model.fit_transform(train_X)
test_x_ae_output = sae_model.transform(test_X)
elm = ELM(train_x_ae_output.shape[1], train_Y.shape[1])
elm.add_neurons(50, "sigm")
elm.train(train_x_ae_output, train_Y, "LOO")
test_res = elm.predict(test_x_ae_output)
original_test_Y = labels_scalar.inverse_transform(test_Y).reshape(-1)
original_test_res = labels_scalar.inverse_transform(test_res).reshape(-1)
mape = np.mean(np.absolute((original_test_Y - original_test_res) / original_test_Y))
rmse = np.sqrt(np.mean(np.square(original_test_Y - original_test_res)))
print("test mape loss: {}".format(mape))
print("test rmse loss: {}".format(rmse))
# sae_model.finetunning(train_X, train_Y, 'rmse', dense_activations=['sigmoid', 'sigmoid', 'sigmoid'], dense_layers=[20, 20, 1],
#                       learning_rate=0.002, print_step=200, epoch=30000, batch_size=2000)
# test_res = sae_model.dense_evaluate(test_X, activations=['sigmoid', 'sigmoid', 'sigmoid'])
# # test_Y = test_Y.reshape(-1)
#
# original_test_Y = labels_scalar.inverse_transform(test_Y).reshape(-1)
# original_test_res = labels_scalar.inverse_transform(test_res).reshape(-1)
# mape = np.mean(np.absolute((original_test_Y - original_test_res) / original_test_Y))
# rmse = np.sqrt(np.mean(np.square(original_test_Y - original_test_res)))
# print("test mape loss: {}".format(mape))
# print("test rmse loss: {}".format(rmse))
