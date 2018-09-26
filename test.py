from deepautoencoder import StackedAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf

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

# train / test  split
idx = np.random.rand(data.shape[0]) < 0.8
train_X, train_Y = data[idx], labels[idx]
test_X, test_Y = data[~idx], labels[~idx]

sae_model = StackedAutoEncoder(dims=[50, 50], activations=['linear', 'linear'], epoch=[
                           3000, 3000], loss='rmse', lr=0.007, batch_size=100, print_step=200)
sae_model.fit(train_X)
test_X_ = sae_model.transform(test_X)
sae_model.finetunning(train_X, train_Y, 'rmse', learning_rate=0.005, print_step=200, epoch=2000, batch_size=3000)
