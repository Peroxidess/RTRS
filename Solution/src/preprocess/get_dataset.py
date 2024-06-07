import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer
import torch
import torch.utils.data as data


class DataPreprocessing():
    def __init__(self, train_data, val_data=pd.DataFrame([]), test_data=pd.DataFrame([]),
                 target=None, ca_feat_th=8, task_name='reg', seed=2022,
                 flag_label_onehot=False,
                 flag_ex_null=False, flag_ex_std_flag=False, flag_ex_occ=False,
                 flag_ca_co_sel=True, flag_ca_fac=True, flag_onehot=True, flag_nor=True,
                 flag_feat_emb=False, flag_RUS=False, flag_confusion=False, flaq_save=False):

        self.col_sp = train_data.filter(regex=r'label|次序|编号|action|ID').columns.tolist()
        if not target:
            self.col_label = train_data.filter(regex=r'label').columns.tolist()
            self.target = {}
            for col in self.col_label:
                self.target[col] = col
        else:
            self.col_label = list(target.values())
            self.target = target
        self.col_action = train_data.filter(regex=r'action').columns.tolist()
        self.train_data = self.drop_labelnan(train_data)
        self.val_data = self.drop_labelnan(val_data)
        self.test_data = self.drop_labelnan(test_data)
        self.data_all = None

        self.ca_feat_th = ca_feat_th
        self.task_name = task_name
        self.seed = seed
        self.flag_label_onehot = flag_label_onehot
        self.flag_ex_null = flag_ex_null
        self.flag_ex_std_flag = flag_ex_std_flag
        self.flag_ex_occ = flag_ex_occ
        self.flag_ca_co_sel = flag_ca_co_sel
        self.flag_ca_fac = flag_ca_fac
        self.flag_onehot = flag_onehot
        self.flag_nor = flag_nor
        self.flag_feat_emb = flag_feat_emb
        self.flag_RUS = flag_RUS
        self.flag_confusion = flag_confusion
        self.flaq_save = flaq_save

    def drop_labelnan(self, data):
        data_ = pd.DataFrame([])
        if not data.empty:
            data_ = data.dropna(subset=list(self.target.values()))
        return data_

    def features_ex(self, data, drop_NanRatio=0.95):
        col_drop = set()
        # 排除 空值过多的特征
        col_null = {}
        if self.flag_ex_null:
            data_null_count = \
                data.isnull().sum() / data.shape[0]
            col_null = data_null_count[data_null_count > drop_NanRatio].index

        # 排除 标准差过小的特征
        col_std = {}
        if self.flag_ex_std_flag:
            # std = StandardScaler()
            # data_ = pd.DataFrame(std.fit_transform(data), index=data.index, columns=data.columns)
            data_des = data.describe()
            col_std = data_des.columns[data_des.loc['std'] < 0.1]
            for col_ in self.col_sp:
                try:
                    col_std.remove(col_)
                except:
                    pass

        # 排除 某一值占比过多的特征
        col_occ_r = {}
        data_temp = data.fillna(-1)
        if self.flag_ex_occ:
            dict_occ_r = {}
            for col_ in data_temp.columns:
                val_count = data_temp[col_].value_counts()
                occ_r = val_count.iloc[0] / data_temp.shape[0]
                dict_occ_r[col_] = occ_r
            df_occ_r = pd.DataFrame.from_dict(dict_occ_r, orient='index').sort_values(by=0)
            col_occ_r = df_occ_r[df_occ_r[0] > 0.95].index
        if self.flag_ex_null or self.flag_ex_std_flag or self.flag_ex_occ:
            col_drop = set(col_occ_r) | set(col_null) | set(col_std)
            # col_dropno = set(self.col_sp) & col_drop
        col_drop = col_drop - set(self.col_sp) - set(['tab'])
        return col_drop

    def ca_co_sel(self, data):
        if self.flag_ca_co_sel:
            dict_col_rename = {}
            ca_col = []
            co_col = []
            col_to2excluded = self.col_sp + ['tab']
            data_columns = data.drop(columns=col_to2excluded)
            for col in data_columns.columns:
                data_col = data[col]
                col_feat_num = len(set(data_col))
                if self.ca_feat_th >= col_feat_num > 1 or data[col].dtypes == 'object':
                    col_ = str(col) + '_sparse'
                    ca_col.append(col_)
                    # data.rename(columns={col: col_}, inplace=True)
                # if col_feat_num > self.ca_feat_th and (data[col].dtypes == 'float64' or data[col].dtypes == 'int64'):
                elif col_feat_num > self.ca_feat_th or data[col].dtypes == 'float64':
                    col_ = str(col) + '_dense'
                    co_col.append(col_)
                    # data.rename(columns={col: col_}, inplace=True)
                else:
                    col_ = col
                    print('ca co can`t handle', data[col].dtype, col)
                dict_col_rename[col] = col_
        else:
            ca_col = data.filter(regex=r'sparse').columns.tolist()
            co_col = data.filter(regex=r'dense').columns.tolist()
        return ca_col, co_col, dict_col_rename

    def feat_emb(self, ca_col, co_col):
        emb_col = ca_col
        feat_dict = {}
        tc = 0
        for col in emb_col:
            if col in co_col:
                feat_dict[col] = {0: tc}
                tc += 1
            elif col in ca_col:
                us = self.data_all[col].unique()
            feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
            tc += len(us)
        feat_dim = tc
        return feat_dict, feat_dim

    def process(self):
        if not self.test_data.empty:
            self.test_data['tab'] = 2
        if not self.val_data.empty:
            self.val_data['tab'] = 1
        self.train_data['tab'] = 0
        self.col_sp.append('tab')
        self.data_all = pd.concat([self.train_data, self.val_data, self.test_data], axis=0)

        col_drop = self.features_ex(self.data_all[self.data_all['tab'] == 0].drop(columns=self.col_sp), drop_NanRatio=0.95)
        self.data_all.drop(columns=col_drop, inplace=True)
        # if self.flaq_save:
        #     self.data_all.drop(columns=['tab']).to_csv('./data_preprocessed_row.csv', encoding='gb18030', index_label='subject_id')

        if self.data_all[self.col_label].dtypes[0] == 'object':
            self.data_all[self.col_label] = self.data_all[self.col_label].apply(LabelEncoder().fit_transform)

        ca_col = self.data_all.columns
        co_col = self.data_all.columns
        if self.flag_ca_co_sel:
            ca_col, co_col, dict_col_rename = self.ca_co_sel(self.data_all)
            self.data_all.rename(columns=dict_col_rename, inplace=True)

        if self.flag_ca_fac:
            col_fac = self.data_all.select_dtypes('object').columns
            if not col_fac.empty:
                self.data_all[col_fac] = pd.concat([self.data_all[col_fac].apply(lambda ser: pd.factorize(ser, sort=True)[0])])
                self.data_all[col_fac] = self.data_all[col_fac].replace({-1: np.nan})

        if self.flag_onehot:
            self.data_all = pd.get_dummies(self.data_all, columns=ca_col, dummy_na=False)
            for ca_col_ in ca_col:
                ca_col_dum = self.data_all.filter(regex=f'{ca_col_}').columns
                data_temp = self.data_all[ca_col_dum].sum(axis=1)
                index2nan = self.data_all[data_temp == 0].index
                if not index2nan.empty:
                    self.data_all.loc[index2nan, ca_col_dum] = np.nan
            ca_col = self.data_all.filter(regex=r'sparse').columns.tolist()
        nor = None
        if self.flag_nor:
            # co_col.extend(self.col_action) # 回归目标也需要归一化避免在sup_label分类预测中的模型崩溃
            # co_col.extend(['label1'])
            mms = MinMaxScaler(feature_range=(0, 1))
            std = StandardScaler()
            qt = QuantileTransformer
            nor_sp = mms
            self.data_all[self.col_action] = pd.DataFrame(nor_sp.fit_transform(self.data_all[self.col_action]), columns=self.col_action,
                                                 index=self.data_all.index)
            for nor in [mms]:
                if len(co_col) > 0:
                    self.data_all[co_col] = pd.DataFrame(nor.fit_transform(self.data_all[co_col]), columns=co_col, index=self.data_all.index)
            co_col = [x for x in co_col if x not in self.col_sp]
        self.train_data = self.data_all[self.data_all['tab'] == 0]
        self.val_data = self.data_all[self.data_all['tab'] == 1]
        self.test_data = self.data_all[self.data_all['tab'] == 2]

        if self.flag_RUS:
            set_label = self.train_data[self.target['label1']].unique()
            data_less_num = self.train_data.shape[0]
            list_data_sample = []
            for label_ in set_label:
                data_ = self.train_data.loc[self.train_data[self.target['label1']] == label_]
                print(f'class {label_}: {data_.shape[0]}')
                if data_.shape[0] < data_less_num:
                    data_less_num = data_.shape[0]
            for label_ in set_label:
                data_ = self.train_data.loc[self.train_data[self.target['label1']] == label_]
                data_sample = data_.sample(n=data_less_num, random_state=self.seed)
                list_data_sample.append(data_sample)
            self.train_data = pd.concat(list_data_sample).sample(frac=1, random_state=self.seed)
        if self.flaq_save:
            self.data_all = pd.concat([self.train_data, self.val_data, self.test_data], axis=0)
            self.data_all.to_csv('./data_preprocessed.csv', index=False, encoding='gb18030', index_label='subject_id')

        if self.flag_confusion:
            index_sample = self.train_data.sample(frac=0.2, random_state=self.seed).index
            train_data_ = self.train_data.loc[index_sample]['label1'].replace({0: 1, 1: 0})
            self.train_data.loc[index_sample, 'label1'] = train_data_
        self.train_data.drop(columns=['tab'], inplace=True)
        self.val_data.drop(columns=['tab'], inplace=True)
        self.test_data.drop(columns=['tab'], inplace=True)
        if 'tab' in self.col_sp:
            self.col_sp.remove('tab')
        col_all = co_col + ca_col + self.col_sp
        return self.train_data[col_all], self.val_data[col_all], self.test_data[col_all], ca_col, co_col, nor

    def anomaly_dectection(self, col_):
        col_sp = copy.deepcopy(self.col_sp)
        col_sp.append('tab')
        col_sp.remove('action')
        col_list = self.train_data.filter(regex=r'dense').columns.tolist()
        col_list.append('action')
        # col_list = self.train_data.filter(regex=rf'{col_}').columns
        data_raw = pd.concat([self.train_data, self.val_data, self.test_data], axis=0)
        index_outer = pd.Index([])
        for col in col_list:
            std_ = data_raw[col].std()
            mean_ = data_raw[col].mean()
            index_low = data_raw[data_raw[col] < mean_ - 3 * std_].index
            index_high = data_raw[data_raw[col] > mean_ + 3 * std_].index
            index_outer = index_outer.append([index_low, index_high])
        print(f'index_outer shape: {index_outer.shape}')
        clean_data = data_raw.drop(index=index_outer)
        if not self.test_data.empty:
            self.train_data = clean_data[clean_data['tab'] == 0]
            self.val_data = clean_data[clean_data['tab'] == 1]
            self.test_data = clean_data[clean_data['tab'] == 2]
        else:
            self.train_data = clean_data
    pass


class MyDataset(data.Dataset): # 继承torch.utils.data.Dataset
    def __init__(self,
                 data,
                 label=None,
                 random_seed=0):
        super(MyDataset, self).__init__()
        self.rnd = np.random.RandomState(random_seed)
        data = data.astype('float32')

        list_data = []
        if label is not None:
            for index_, values_ in data.iterrows():
                y = torch.LongTensor([label.loc[index_].astype('int64')]).squeeze()
                x = data.loc[index_].values
                list_data.append((x, y))
        else:
            for index_, values_ in data.iterrows():
                x = data.loc[index_].values
                list_data.append((x))

        self.shape = x.shape
        self.data = list_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
