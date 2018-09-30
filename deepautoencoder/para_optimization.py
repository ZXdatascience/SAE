class ParaOptimization:
    @staticmethod
    def grid_search(params):
        param_combinations = []
        params_keys = list(params.keys())
        params_values = list(params.values())
        temp = []
        ParaOptimization.back_track(params_keys, params_values, temp, param_combinations)
        return param_combinations

    @staticmethod
    def back_track(keys, values, temp, result):
        if len(temp) == len(values):
            temp_dict = {}
            for i, key in enumerate(keys):
                temp_dict[key] = temp[i]
            result.append(temp_dict)
            return
        param_range = values[len(temp)]
        for i in range(len(param_range)):
            temp.append(param_range[i])
            ParaOptimization.back_track(keys, values, temp, result)
            temp.pop()

if __name__ == '__main__':
    import pandas as pd
    dic = {1: [1,2,3,4], 'a': [4,2,3,48], 'b': [1,6,7,33], 'c': [1, 2, 4]}
    df = ParaOptimization.grid_search(dic)
    print(df)
    print(pd.DataFrame(df))
    print(len(ParaOptimization.grid_search(dic)))



