import pandas as pd
# from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
# from imblearn.under_sampling import OneSidedSelection, RandomUnderSampler, ClusterCentroids
# from imblearn.combine import SMOTETomek, SMOTEENN


class IMB():
    def __init__(self, name_method, random_state=2022):
        self.seed = random_state
        self.model = self.model_build(name_method)

    def model_build(self, name_method):
        if name_method == 'SMOTE':
            model = SMOTE(random_state=self.seed)
        elif name_method == 'SP':
            model = SP(seed=self.seed)
        else:
            model = RUS(seed=self.seed)
        return model

    def fit_transform(self, data_x, data_y):
        X_new, Y_new = self.model.fit_resample(data_x, data_y)
        return X_new, Y_new


def smote(X, Y, seed=2022):
    sm = SMOTE(random_state=seed)
    X_new, Y_new = sm.fit_resample(X, Y)
    X_ = pd.concat([X, X_new])
    X_smote = X_.drop_duplicates(keep=False)
    Y_smote = Y_new.loc[X_smote.index]
    X_smote['index'] = list(range(X.index.max() + 1, X.index.max() + X_smote.shape[0] + 1))
    Y_smote.index = list(range(X.index.max() + 1, X.index.max() + X_smote.shape[0] + 1))
    X_smote.set_index('index', inplace=True)

    X_res = pd.concat([X, X_smote])
    Y_res = pd.concat([Y, Y_smote])
    X_res_ = X_res.sample(frac=1, random_state=seed)
    Y_res_ = Y_res.loc[X_res_.index]
    print(X_res_.shape, Y_res_.shape)
    return X_res_, Y_res_


class RUS():
    def __init__(self, seed=2022):
        self.seed = seed

    def fit_resample(self, data, label):
        data_1 = data.loc[label['label1'] == 1]
        print(f'class 1 : {data_1.shape[0]}')
        data_0 = data.loc[label['label1'] == 0]
        print(f'class 0 : {data_0.shape[0]}')
        if data_1.shape[0] < data_0.shape[0]:
            data_less = data_1
            data_more = data_0
        else:
            data_less = data_0
            data_more = data_1
        data_more = data_more.sample(n=data_less.shape[0] * 1, random_state=self.seed)
        data_rus = pd.concat([data_more, data_less]).sample(frac=1, random_state=self.seed)
        label_rus = label.loc[data_rus.index]
        return data_rus, label_rus


class SP():
    def __init__(self, seed=2022):
        self.seed = seed

    def fit_resample(self, data, label):
        data_1 = data.loc[label['label1'] == 1]
        print(f'class 1 : {data_1.shape[0]}')
        data_0 = data.loc[label['label1'] == 0]
        print(f'class 0 : {data_0.shape[0]}')
        if data_1.shape[0] < data_0.shape[0]:
            data_less = data_1
            data_more = data_0
        else:
            data_less = data_0
            data_more = data_1
        data_more_sample = data_more.sample(n=data_less.shape[0] * 1, random_state=self.seed)
        label_less = label.loc[data_less.index]
        label_more_sample = label.loc[data_more_sample.index]
        label_more = label.loc[data_more.index]
        alpha = 0.1
        data_sp = alpha * data_more_sample.values + (1 - alpha) * data_less.values
        data_sp = pd.DataFrame(data_sp, columns=data.columns)
        label_sp = alpha * label_more_sample.values + (1 - alpha) * label_less.values
        label_sp = pd.DataFrame(label_sp, columns=label.columns)
        data_sp.index = data_sp.index.values + data.index.max() + 1
        label_sp.index = label_sp.index.values + label.index.max() + 1
        data_rus = pd.concat([data_more, data_less, data_sp]).sample(frac=1, random_state=self.seed)
        label_rus = pd.concat([label_more, label_less, label_sp])
        label_rus = label_rus.loc[data_rus.index]
        return data_rus, label_rus
