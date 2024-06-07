from copy import deepcopy
import numpy as np
from numpy.random.mtrand import random, seed
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from model.evaluate import features_plot
from model.mviilgan import OGAN


class MVI():
    def __init__(self, shape_train_data, co_col, ca_col, task_name, target, seed, method='ours'):
        self.shape_train_data = shape_train_data
        self.co_col = co_col
        self.ca_col = ca_col
        self.task_name = task_name
        self.target = target
        self.seed = seed
        np.random.seed(seed)
        self.method = method
        self.col_label = list(self.target.values())

        if 'ours' in self.method:
            self.model = OGAN(shape_x=shape_train_data, feed_forward_size=shape_train_data, n_heads=2, model_dim=shape_train_data//2, coder_stack=2, noise_dim=128, ablation_sel=method)
        elif 'knn' in self.method:
            self.model = KNNImputer()
        else:
            self.model = Statistics(self.ca_col, self.co_col)

    def manual_nan(self, data, label, dict_ManualRatio, k=0, type_data='Train', flag_saving=False):
        data_na_sum_col = data.isna().sum(axis=0).values
        data_na_sum_col_thre = (data_na_sum_col < data.shape[0] * dict_ManualRatio['a']) # highly missing -> False -> saving col which mr ratio > a -> saving amlost 1-a features
        data_na_sum_row = data.isna().sum(axis=1).values
        data_na_sum_row_thre = (data_na_sum_row < data.shape[1] * dict_ManualRatio['b'])
        data_na_sum_row_thre_ = data_na_sum_row_thre.reshape(-1, 1)
        data_na_sum_col_thre_ = data_na_sum_col_thre.reshape(1, -1)
        data_na_thre_matrix = np.dot(data_na_sum_row_thre_, data_na_sum_col_thre_)
        unif_random_matrix = np.random.uniform(0., 1., size=data.shape)
        index_notan_manul = 1 * (unif_random_matrix < dict_ManualRatio['c'])
        index_notan_manul = index_notan_manul.astype(np.bool)
        data_na_thre_matrix_thre = data_na_thre_matrix * index_notan_manul
        data_na_thre_matrix_thre = ~data_na_thre_matrix_thre
        data_manual_nan = data.where(data_na_thre_matrix_thre)
        index_ManualNan = pd.DataFrame(data_manual_nan.notna(), index=data_manual_nan.index, columns=data_manual_nan.columns)
        if flag_saving:
            index_ManualNan.to_csv(f'mask_{type_data}_ManualNan_KFlod[{k}].csv', index_label=['index'])
        data_manual_nan.dropna(how='all', axis=0, inplace=True)
        label_ = label.loc[data_manual_nan.index]
        return data_manual_nan, label_, index_ManualNan

    @staticmethod
    def drop_nan_sample(data, label, ratio=0.9):
        nan_sample = data.isna().sum(axis=1)
        print(f'sample raw: {data.shape[0]}')
        nan_sample_ratio = nan_sample / data.shape[1]
        drop_nan_sample = nan_sample_ratio[nan_sample_ratio < ratio]
        data_dropnansample = data.loc[drop_nan_sample.index]
        label_dropnansample = label.loc[drop_nan_sample.index]
        print(f'sample curr: {data_dropnansample.shape[0]}')
        print(f'sample drop ratio: {1 - data_dropnansample.shape[0] / data.shape[0]}')
        return data_dropnansample, label_dropnansample


    @staticmethod
    def show_nan_ratio(data):
        sum_nan_all = data.isnull().sum().sum()
        sum_nan_ratio_micro = sum_nan_all / (data.shape[0] * data.shape[1])
        print(f'nan_ratio_micro: {sum_nan_ratio_micro}')
        return sum_nan_ratio_micro

    def fit_transform(self, data, data_raw, label, Fold_k=0, mr=65, test=0):
        if 'ours' in self.method:
            batchsize = 256
            epoch = 50
            # shape_train = data.shape[0] // batchsize * batchsize
            # data = data.iloc[:shape_train]
            # data_raw = data_raw.iloc[:shape_train]
            # label = label.iloc[:shape_train]

        index_notna = data.notna()
        data_randfillna = deepcopy(data)
        list_label_index = []
        list_set_label = np.sort(label[self.target['label1']].unique())
        len_max = 0
        index_max = data.index.max()
        for label_ in list_set_label:
            index_ = label[label[self.target['label1']] == label_].index
            if index_.shape[0] > len_max:
                len_max = index_.shape[0]
            list_label_index.append(index_)
        index_ger_list = []
        index_extend = pd.Index([])
        for label_index in list_label_index:
            len_index_label = label_index.shape[0]
            len_needed_ger = len_max - len_index_label
            ger_index = pd.Index(list(range(index_max + 1, len_needed_ger + index_max + 1)))
            index_ger_list.append(ger_index)  # list append
            index_extend = index_extend.append(ger_index)  # index concat
            index_max = len_needed_ger + index_max

        if 'ours' in self.method:
            index_fake = index_extend.append(data.index)
            for col_ in self.ca_col:
                xx = data[col_].value_counts().values / data[col_].notna().sum()
            df_fake = pd.DataFrame(np.random.normal(0., 1., size=(index_fake.shape[0], data.shape[1])),
                                   index=index_fake, columns=data.columns)
            df_fake_label = pd.DataFrame(np.zeros(shape=(index_fake.shape[0], 1)), index=index_fake,
                                         columns=label.columns)
            for col_, values in df_fake.iteritems():
                unif_random_matrix = np.random.uniform(0., 1., size=(values.shape[0]))
                index_notan_manul = 1 * (unif_random_matrix < 0.95)
                index_notan_manul = index_notan_manul.astype(np.bool)
                index_notan_manul = pd.DataFrame(index_notan_manul, index=df_fake.index, columns=[col_])
                df_fake[col_] = values.where(index_notan_manul[col_].values)
            index_notna_fake = df_fake.isna()  # random normal local nan
            data_fake_randfillna = deepcopy(df_fake)

            for col, values in data_randfillna.iteritems():
                for i, (label_index, ger_index) in enumerate(zip(list_label_index, index_ger_list)):
                    col_label = data_randfillna.loc[label_index, col]
                    mul_ratio = col_label.value_counts().values / col_label.notna().sum()
                    describe_col_label = col_label.describe()
                    index_label_ = label_index.append(ger_index)
                    df_fake_label.loc[index_label_] = i
                    if 'sparse' in col:
                        # df_fake[col] = values.where(index_notan_manul[col].values)
                        # data_randfillna.loc[:, col] = data_randfillna.loc[:, col].apply(
                        #     lambda x: np.random.multinomial(n=1, pvals=mul_ratio_random).argmax() if np.isnan(x) else x)
                        data_randfillna.loc[:, col] = data_randfillna.loc[:, col].apply(
                            lambda x: np.random.normal(loc=0, scale=1) if np.isnan(x) else x)
                        # data_randfillna.loc[:, col] = data_randfillna.loc[:, col].apply(
                        #     lambda x: np.random.uniform(0, 0.01) if np.isnan(x) else x)
                        data_fake_randfillna.loc[index_label_, col] = df_fake.loc[index_label_, col].apply(
                            lambda x: np.random.multinomial(n=1, pvals=mul_ratio).argmax() if np.isnan(x) else x)

                    elif 'dense' in col:
                        data_randfillna.loc[:, col] = data_randfillna.loc[:, col].apply(
                            lambda x: np.random.normal(loc=0, scale=1) if np.isnan(x) else x)
                        # data_randfillna.loc[:, col] = data_randfillna.loc[:, col].apply(
                        #     lambda x: np.random.uniform(0, 0.01) if np.isnan(x) else x)
                        data_fake_randfillna.loc[index_label_, col] = df_fake.loc[index_label_, col].apply(
                            lambda x: np.random.normal(loc=describe_col_label["mean"],
                                                       scale=describe_col_label["std"]) if np.isnan(x) else x)
            data_fake_randfillna.index = data_fake_randfillna.index + index_max
            df_fake_label.index = df_fake_label.index + index_max
            index_notna_fake.index = index_notna_fake.index + index_max
            data_fake_randfillna_sample = data_fake_randfillna.sample(n=data_raw.shape[0], random_state=self.seed)
            label_fake = df_fake_label.loc[data_fake_randfillna_sample.index]
            index_notna_fake_sample = index_notna_fake.loc[data_fake_randfillna_sample.index]

            data_filled, _ = self.model.fit_transform(data_randfillna, data_fake_randfillna_sample,
                                                      label[self.target['label1']],
                                                      data_raw, index_notna.values, index_notna_fake_sample.values,
                                                      epochs=epoch, batch_size=batchsize, seed=self.seed
                                                      )
            data_filled = pd.DataFrame(data_filled,
                                       index=data_randfillna.index,
                                       columns=data_randfillna.columns
                                       )
            data_fake_filled = self.model.transform(data_fake_randfillna_sample, data_fake_randfillna_sample,
                                                    index_notna_fake_sample)
            data_fake_filled = pd.DataFrame(data_fake_filled,
                                            index=data_fake_randfillna_sample.index,
                                            columns=data_fake_randfillna_sample.columns
                                            )
            if 'ger_not' in self.method:
                data_filled = pd.DataFrame(data_filled,
                                           index=data_randfillna.index,
                                           columns=data_randfillna.columns
                                           )
                label_ = label
            else:
                data_filled_wlabel = pd.concat([data_filled, label], axis=1)
                data_fake_filled_wlabel = pd.concat([data_fake_filled, label_fake], axis=1)
                data_fake_filled_wlabel = data_fake_filled_wlabel.sample(frac=0.9)
                data_filled_and_fake = pd.concat([data_filled_wlabel, data_fake_filled_wlabel], axis=0)
                label_ = data_filled_and_fake[[self.target['label1']]]
                label_ = label_.loc[:, ~label_.columns.duplicated()]
                data_filled = data_filled_and_fake.drop(columns=self.target['label1'])
        else:
            data_filled = self.model.fit_transform(data_randfillna)
            label_ = label
            pass
        if type(data_filled) == np.ndarray:
            data_filled_ = pd.DataFrame(data_filled,
                                       index=data_randfillna.index,
                                       columns=data_randfillna.columns
                                       )
        else:
            data_filled_ = data_filled
        return data_filled_, label_

    def transform(self, data, label):
        index_notna = data.notna().values
        data_randfillna = data
        if 'ours' in self.method:
            for col, values in data_randfillna.iteritems():
                if 'sparse' in col:
                    # data_randfillna[col] = data_randfillna[col].apply(
                    #     lambda x: np.random.multinomial(n=1, pvals=mul_ratio).argmax() if np.isnan(x) else x)
                    data_randfillna[col] = data_randfillna[col].apply(
                        lambda x: np.random.normal(loc=0,
                                                   scale=1) if np.isnan(x) else x)

                elif 'dense' in col:
                    # data_randfillna[col] = data_randfillna[col].apply(
                    #     lambda x: np.random.normal(loc=describe_col["mean"],
                    #                                scale=describe_col["std"]) if np.isnan(x) else x)
                    data_randfillna[col] = data_randfillna[col].apply(
                        lambda x: np.random.normal(loc=0,
                                                   scale=1) if np.isnan(x) else x)

            data_filled = self.model.transform(data_randfillna, data, index_notna)
        else:
            data_filled = self.model.transform(data_randfillna)
        data_filled = pd.DataFrame(data_filled, index=data.index, columns=data.columns)
        return data_filled

    def plot_notna(self, data, label):
        data_raw_notan = data.dropna()
        features_plot(data_raw_notan.values, label.loc[data_raw_notan.index].values, seed=seed, flag_tsne=True)
        features_plot(data_raw_notan.values, label.loc[data_raw_notan.index].values, seed=seed, flag_tsne=False)

    def statistics(self, data):
        data_filled = deepcopy(data)
        data_filled[self.ca_col] = data_filled[self.ca_col].fillna(0)
        data_filled[self.co_col] = data[self.co_col].fillna(data[self.co_col].mean(axis=0))
        return data_filled


class Statistics():
    def __init__(self, ca_col, co_col):
        self.ca_col = ca_col
        self.co_col = co_col

        self.train_recode = pd.DataFrame([], columns=ca_col + co_col, index=[0])

    def fit_transform(self, data):
        data_filled = deepcopy(data)
        data_filled = data_filled.fillna(data_filled.median(axis=0))
        data_filled = data_filled.fillna(0)
        return data_filled

    def transform(self, data):
        data_filled = deepcopy(data)
        data_filled = data_filled.fillna(data_filled.median(axis=0))
        data_filled = data_filled.fillna(0)
        return data_filled
