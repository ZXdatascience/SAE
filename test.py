from deepautoencoder import StackedAutoEncoder
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf
from sklearn import preprocessing
from hpelm import ELM
from deepautoencoder.para_optimization import ParaOptimization
from sklearn.model_selection import KFold


def add_lag(cons, thres=0.1):
    cons = np.array(cons)
    pa = pacf(cons, nlags=150)
    above_thres_indices = np.argwhere(pa > thres)
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


if __name__ == '__main__':
    #########################
    # search space for hpyer-parameters
    #########################
    # thres = [0.1]
    # use_mean_filter = [True, False]
    # window_size = [2]
    # ae_learning_rate = [0.001, 0.005, 0.01]
    # noise = ['gaussian', 'mask-0.2', 'mask-0.5']
    # batch_size = [100, 200]
    # epochs = [5000, 10000]
    # ELM_neurons_nums = [20, 50]
    # ELM_neurons_type = ['sigm', 'rbf_l2']

    thres = [0.1]
    use_mean_filter = [False]
    window_size = [2]
    ae_learning_rate = [0.001, 0.005]
    noise = ['gaussian']
    batch_size = [100, 200]
    epochs = [5000]
    ELM_neurons_nums = [20, 50]
    ELM_neurons_type = ['sigm']
    para_dict = {'thres': thres, 'use_mean_filter': use_mean_filter, 'window_size': window_size,
                 'ae_learning_rate': ae_learning_rate, 'noise': noise, 'batch_size': batch_size,
                 'epochs': epochs,  'ELM_neurons_nums': ELM_neurons_nums, 'ELM_neurons_type': ELM_neurons_type}
    search_space = ParaOptimization.grid_search(para_dict)
    CV_result = pd.DataFrame(search_space)
    print(CV_result)
    # Read the data
    train_data_name = "data/building1retail.csv"
    df = pd.read_csv(train_data_name)
    electricity_cons_original = df['Power (kW)'].values
    avg_mapes = []
    avg_rmses = []
    for i, paras in enumerate(search_space):
        print('Running the {} parameter combination'.format(i + 1))
        print(paras)
        #########################
        # Pre-processing Data
        #########################
        if paras['use_mean_filter']:
            electricity_cons = mean_filter(electricity_cons_original, window_size=paras['window_size'])
        else:
            electricity_cons = electricity_cons_original
        min30_filtered = to_30_min(electricity_cons)
        data, labels = add_lag(min30_filtered, thres=paras['thres'])
        data_scalar = preprocessing.MinMaxScaler()
        labels_scalar = preprocessing.MinMaxScaler()
        data = data_scalar.fit_transform(data)
        labels = labels_scalar.fit_transform(labels.reshape(-1, 1))
        #########################
        # 5 Fold Split
        #########################
        kf = KFold(n_splits=5, shuffle=True)
        k_fold_mapes = []
        k_fold_rmses = []
        for train_index, test_index in kf.split(data):
            train_X, test_X = data[train_index], data[test_index]
            train_Y, test_Y = labels[train_index], labels[test_index]

            #########################
            # SAE Model
            #########################
            sae_model = StackedAutoEncoder(dims=[50, 50], activations=['sigmoid', 'sigmoid'],
                                           epoch=[paras['epochs'] for _ in (1, 2)],
                                           noise=paras['noise'], loss='sparse', lr=paras['ae_learning_rate'],
                                           batch_size=paras['batch_size'], print_step=200)
            train_x_ae_output = sae_model.fit_transform(train_X)
            test_x_ae_output = sae_model.transform(test_X)
            elm = ELM(train_x_ae_output.shape[1], train_Y.shape[1])
            elm.add_neurons(paras['ELM_neurons_nums'], paras['ELM_neurons_type'])
            elm.train(train_x_ae_output, train_Y, "CV", "OP", k=5, kmax=300)
            test_res = elm.predict(test_x_ae_output)
            # revert the scale from (0, 1) to the original scale
            original_scale_test_Y = labels_scalar.inverse_transform(test_Y).reshape(-1)
            original_scale_test_res = labels_scalar.inverse_transform(test_res).reshape(-1)
            original_scale_test_X = labels_scalar.inverse_transform(test_X)
            # revert the mean filter
            if paras['use_mean_filter']:
                original_test_Y = (original_scale_test_Y - original_scale_test_X[:, -1]) * paras['window_size'] \
                                  + original_scale_test_X[:, -paras['window_size'] - 1]
                original_test_res = (original_scale_test_res - original_scale_test_X[:, -1]) * paras['window_size'] \
                                  + original_scale_test_X[:, -paras['window_size'] - 1]
            else:
                original_test_Y = original_scale_test_Y
                original_test_res = original_scale_test_res
            err = original_test_Y - original_test_res
            rmse = np.sqrt(np.mean(np.square(original_test_Y - original_test_res)))
            no_zero_test_Y = original_test_Y[original_test_Y != 0]
            no_zero_test_res = original_test_res[original_test_Y != 0]
            mape = np.mean(np.absolute((no_zero_test_Y - no_zero_test_res) / no_zero_test_Y))
            k_fold_mapes.append(mape)
            k_fold_rmses.append(rmse)
        avg_mape = np.mean(k_fold_mapes)
        avg_rmse = np.mean(k_fold_rmses)
        print("test mape loss: {}".format(avg_mape))
        print("test rmse loss: {}".format(avg_rmse))
        avg_mapes.append(avg_mape)
        avg_rmses.append(avg_rmse)
    CV_result['mapes'] = avg_mapes
    CV_result['rmses'] = avg_rmses
    CV_result.to_csv('grid_search_results.csv')

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
