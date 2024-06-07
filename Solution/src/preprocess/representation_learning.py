import copy
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from model.vae import AutoEncoder, VAutoEncoder, CVAutoEncoder, CAutoEncoder
import torch
import torch.optim as optim
import torch.utils.data as Data


class RepresentationLearning:
    def __init__(self, dim_input, method='KMeans', seed=2022):
        self.method = method
        if method == 'KMeans':
            self.model = Cluster(seed)
        elif method == 'ours':
            self.model = ZSR(dim_input, seed=seed)
        elif method == 'ae':
            self.model = AE(dim_input, seed)

    def fit_transform(self, train_x, train_y=None, val_x=None, val_y=None):
        col_ = train_x.filter(regex=r'dense|sparse').columns
        col_2 = train_x.filter(regex=r'病灶信息_肿瘤|消融范围|功率').columns
        data_x, data_z = self.model.fit_transform(train_x[col_], train_y, val_x[col_], val_y)
        data_x_all = pd.concat([train_x, data_z], axis=1)
        return data_x_all

    def transform(self, x, y=None):
        col_ = x.filter(regex=r'dense|sparse').columns
        col_2 = x.filter(regex=r'病灶信息_肿瘤|消融范围|功率').columns
        data_x, data_z = self.model.transform(x[col_], y)
        data_x_all = pd.concat([x, data_z], axis=1)
        return data_x_all


class Cluster:
    def __init__(self, seed=2022, n_clusters=1000):
        self.model = KMeans(n_clusters=n_clusters)

    def fit_transform(self, train_x, train_y=None, val_x=None, val_y=None):
        self.model.fit_transform(train_x)
        centers_ = self.model.cluster_centers_
        centers_df = pd.DataFrame(centers_, columns=train_x.columns)
        centers_df['cluster_y'] = list(range(centers_.shape[0]))
        data_x = pd.DataFrame([])
        for sample_cluster in self.model.labels_:
            df_new = centers_df[centers_df['cluster_y'] == sample_cluster]
            data_x = pd.concat([data_x, df_new], axis=0)
        data_x.index = train_x.index
        data_ = data_x.drop(columns=['cluster_y'])
        mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output = metric_rec(train_x.values, data_.values)
        print(f'mae_re {mae_re} mape_re {mape_re}')
        return data_

    def transform(self, x, y=None):
        labels_ = self.model.predict(x)
        centers_ = self.model.cluster_centers_
        centers_df = pd.DataFrame(centers_, columns=x.columns)
        centers_df['cluster_y'] = list(range(centers_.shape[0]))
        data_x = pd.DataFrame([])
        for sample_cluster in labels_:
            df_new = centers_df[centers_df['cluster_y'] == sample_cluster]
            data_x = pd.concat([data_x, df_new], axis=0)
        data_x.index = x.index
        data_ = data_x.drop(columns=['cluster_y'])
        mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output = metric_rec(x.values, data_.values)
        print(f'mae_re {mae_re} mape_re {mape_re}')
        return data_

    def box_plot(self, data_x_all):
        df_box = pd.DataFrame([])
        centers_ = self.model.cluster_centers_
        for sample_cluster in range(centers_.shape[0]):
            df_new = data_x_all[data_x_all['cluster_y'] == sample_cluster]
            df_tmp = df_new['action']
            df_tmp.reset_index(drop=True, inplace=True)
            df_box = pd.concat([df_box, df_tmp], axis=1)
            df_box.columns = list(range(df_box.shape[1]))
            lko = 1
        df_sort = df_box.T
        df_sort['mean'] = df_box.T.mean(axis=1)
        df_sort['median'] = df_box.T.median(axis=1)
        df_sort.sort_values(by=['median', 'mean'], inplace=True)
        df_box_sort = df_sort.T
        df_box_sort.boxplot(grid=False)
        pass


class ZSR():
    def __init__(self, dim_input, flag_model_load=False, name_method='zsr', seed=2022):
        self.seed = seed
        self.dim_input = dim_input
        self.flag_model_load = flag_model_load
        self.name_method = name_method
        self.nor = QuantileTransformer()
        if flag_model_load == True:
            self.model = torch.load('./ZSR.pth')
        else:
            if 'cvae' in self.name_method:
                self.model = CVAutoEncoder(dim_input, z_dim=24)
            elif '_ae' in self.name_method:
                self.model = AutoEncoder(dim_input, z_dim=32)
            elif 'vae' in self.name_method:
                self.model = VAutoEncoder(dim_input, z_dim=32)
            else:
                self.model = CAutoEncoder(dim_input, z_dim=12)
            self.ae = AutoEncoder(dim_input, z_dim=12)

    def input_process(self, data, data_rec, state=None):
        data_dataloader = self.MyDataset(data, data_rec)
        return data_dataloader

    def fit_transform(self, x, y=None, val_x=None, val_y=None, epoch=3.0, epochs_ae=300):
        noise_s = np.random.normal(size=x.shape)
        noise_val = np.random.normal(size=val_x.shape)

        if 'noise_none' in self.name_method or 'masked' in self.name_method:
            val_x_noise = val_x
        else:
            x = x + noise_s
            val_x_noise = val_x + noise_val

        data_dataloader = self.input_process(x, x)
        optim_ = optim.Adam(self.ae.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optim_, step_size=50, gamma=0.99)
        data_loader = Data.DataLoader(data_dataloader, batch_size=256, worker_init_fn=np.random.seed(self.seed))
        total_ae_loss_list = []
        loss_ae = torch.nn.MSELoss()
        for epoch_ae in range(epochs_ae):
            total_ae_loss_ = 0
            self.ae.train()
            for data_x, data_rec in data_loader:
                optim_.zero_grad()
                noise = torch.randn(size=data_x.shape)
                x_rec, z, mu, logvar = self.ae(data_x + 1e-2 * noise)
                rec_ae_loss = loss_ae(data_rec, x_rec)
                rec_ae_loss.backward()
                optim_.step()
                total_ae_loss_ += rec_ae_loss.detach().numpy()
            total_ae_loss_list.append(total_ae_loss_)
            _ = scheduler.step()
        x_rec_tra, _, _, _ = self.ae(torch.tensor(x.values, dtype=torch.float32))
        x_rec_tra = pd.DataFrame(x_rec_tra.detach().numpy(), index=x.index, columns=x.columns)
        x_rec_sparse = x_rec_tra.filter(regex=r'sparse')
        x_sparse = x.filter(regex=r'sparse')
        x_noise = copy.deepcopy(x)
        alpha = 0.1
        x_noise[x_noise.filter(regex=r'sparse').columns] = alpha * x_rec_sparse + (1 - alpha) * x_sparse

        if 'noise_none' in self.name_method:
            x_noise = x
            val_x_noise = val_x
        else:
            x_noise = x + noise_s
            val_x_noise = val_x + noise_val
        if not self.flag_model_load:
            x_lr_list = []
            step = 1
            if 'cluster' in self.name_method:
                bin_list = range(100, len(x_noise) - 100, 1200)
            elif 'equal' in self.name_method:
                bin_list = [61, 61, 29, 29]
                step = 2
            else:
                if 'short' in self.name_method:
                    bin_list = [29, 11, 3]
                elif 'wide' in self.name_method:
                    bin_list = [61, 29, 3]
                else:
                    bin_list = [59, 53, 47, 43, 41, 37, 31, 29, 23, 19]
            for bs in bin_list:
                if 'cluster' in self.name_method:
                    x_lr = self.scale_fuzz_cluster(x_noise, y, bs)
                else:
                    x_lr = self.scale_fuzz(x_noise, y, bs)
                x_lr_list.append(x_lr)
            x_lr_list.append(x_noise)
            if 'equal' in self.name_method:
                x_lr_list.append(x_noise)
            len_x_lr = len(x_lr_list)
            if 'h2l' in self.name_method:
                x_lr_list = x_lr_list[::-1]
            count_all = 0
            for index_ in range(0, len_x_lr - 1, step):  # x_lr_list: lr--->hr
                x_lr_re = x_lr_list[index_]
                x_hr_re = x_lr_list[index_ + 1]
                data_dataloader = self.input_process(x_lr_re, x_hr_re)
                optim_ = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
                # scheduler = torch.optim.lr_scheduler.StepLR(optim_, step_size=1, gamma=0.99)
                data_loader = Data.DataLoader(data_dataloader, batch_size=256, worker_init_fn=np.random.seed(self.seed))
                epoch = epoch ** 1.5
                epoch_ = min(math.ceil(epoch), 125)
                if epoch > 1000:
                    epoch = 1000
                for iter_count in range(epoch_):
                    count_all += 1
                    total_vae_loss_ = 0
                    self.model.train()
                    for data_lr, data_hr in data_loader:
                        optim_.zero_grad()
                        noise = torch.randn(size=data_lr.shape)
                        x_rec, z, mu, logvar = self.model(data_lr + 1e-2 * noise, torch.tensor(1 / (index_ + 1)))
                        rec_loss = self.model.loss(data_hr, x_rec, mu, logvar)
                        rec_loss.backward()
                        optim_.step()
                    total_vae_loss_ += rec_loss.detach().numpy()
                    if iter_count % 5 == 0:
                        self.model.eval()
                        x_hr_raw = val_x_noise.values
                        x_tensor = torch.Tensor(val_x_noise.values)
                        x_hr_rec, z_hr, _, _ = self.model(x_tensor, torch.tensor(0))
                        x_np = x_hr_rec.detach().numpy()
                        self.model.train()
                        mse_re_eval, mae_re_eval, mape_re_eval, r2_re_eval, near_mae_re_eval, x_new_output_eval = metric_rec(
                            x_hr_raw, x_np)
                        print(
                            f'index_{index_} count{iter_count} val total_vae_loss_ {total_vae_loss_} mse_re {mse_re_eval.round(4)} mae_re {mae_re_eval.round(4)} mape_re {mape_re_eval}')
                # _ = scheduler.step()
            # torch.save(self.model, 'ZSR.pth')

        self.model.eval()
        x_ = x_noise
        x_tensor = torch.Tensor(x_.values)
        x_hr, z_hr, _, _ = self.model(x_tensor, torch.tensor(0))
        x_hr_2, z_hr_2, _, _ = self.model(x_hr, torch.tensor(0))
        self.model.train()
        x_np = x_hr.detach().numpy()
        z_np = z_hr.detach().numpy()

        z_nor = self.nor.fit_transform(z_np)
        x_df = pd.DataFrame(x_np, columns=x.columns, index=x.index)
        z_df = pd.DataFrame(z_nor, index=x.index)
        z_df.columns = [str(col) + '_dense' for col in range(z_df.shape[1])]
        mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output = metric_rec(x.values, x_np)
        mse_input, mae_input, mape_input, r2_input, near_mae_input, near_mae_input = metric_rec(x.values, x_.values)
        mse_noise, mae_noise, mape_noise, r2_noise, near_mae_noise, near_mae_noise = metric_rec(x.values,
                                                                                                x_noise.values)

        print(
            f'train mae_re {mae_re} mape_re {mape_re} mae_input {mae_input} mape_input {mape_input} mae_noise{mae_noise} mape_noise {mape_noise}')
        return x_df, z_df

    def transform(self, x, y=None, flag_model_load=False):
        x_noise = x
        x_tensor = torch.Tensor(x_noise.values)
        x_hr, z_hr, _, _ = self.model(x_tensor, torch.tensor(0))
        x_np = x_hr.detach().numpy()
        z_np = z_hr.detach().numpy()
        mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output = metric_rec(x.values, x_np)
        print(f'test mae_re {mae_re} mape_re {mape_re}')

        z_nor = self.nor.transform(z_np)

        x_df = pd.DataFrame(x_np, columns=x.columns, index=x.index)
        z_df = pd.DataFrame(z_nor, index=x.index)
        z_df.columns = [str(col) + '_dense' for col in range(z_df.shape[1])]
        return x_df, z_df

    def scale_fuzz_cluster(self, data, label, bins_stride=7):
        model = Cluster(n_clusters=bins_stride)
        fuzzy_data = model.fit_transform(data)
        alpha = 0.1
        fuzzy_data_ = fuzzy_data * (1 - alpha) + data * alpha
        return pd.DataFrame(fuzzy_data_, index=data.index, columns=data.columns)

    def scale_fuzz(self, data, label, bins_stride=7):
        data_fuzz = None
        for col_index, values_ in data.iteritems():
            data_col = values_.values
            set_data_col = set(data_col)
            bins = max(int(np.floor(len(set_data_col) / bins_stride)), 2)
            histogram_data_col_weight = np.histogram(data_col, bins=bins, weights=data_col)[0]
            histogram_data_col = np.histogram(data_col, bins=bins)

            histogram_per = pd.Series(histogram_data_col[1])
            histogram_post = pd.Series(histogram_data_col[1]).shift()
            histogram_ave_ = (histogram_per + histogram_post) / 2
            histogram_ave_ = histogram_ave_.dropna(axis=0, how='any')

            data_col_bins = pd.cut(data_col, bins=bins, labels=histogram_ave_, ordered=False)
            data_col_bins_df = pd.Series(data_col_bins)
            if data_fuzz is None:
                data_fuzz = data_col_bins_df
            else:
                data_fuzz = np.vstack((data_fuzz, data_col_bins_df))
        data_fuzz = np.transpose(data_fuzz)
        alpha = 0.05
        data_fuzz_ = data_fuzz * (1 - alpha) + data.values * alpha
        data_fuzz_df = pd.DataFrame(data_fuzz_, index=data.index, columns=data.columns)
        return data_fuzz_df

    class MyDataset(Data.Dataset):
        def __init__(self,
                     data,
                     data_rec,
                     random_seed=0):
            self.rnd = np.random.RandomState(random_seed)
            data = data.astype('float32')
            data_rec = data_rec.astype('float32')

            list_data = []
            for index_, values_ in data.iterrows():
                x = data.loc[index_].values
                x_rec = data_rec.loc[index_].values
                list_data.append((x, x_rec))

            self.shape = data.shape
            self.data = list_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            data = self.data[idx]
            return data


class AE():
    def __init__(self, dim_input, seed):
        self.seed = seed
        self.dim_input = dim_input
        self.model = AutoEncoder(dim_input, z_dim=14, seed=seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = VAutoEncoder(dim_input, z_dim=32, seed=seed)

    def input_process(self, data, y):
        data_dataloader = self.MyDataset(data, y)
        return data_dataloader

    def transform(self, x, y=None):
        x_rec, z_, _, _ = self.model(torch.Tensor(x.values))
        x_np = x_rec.detach().cpu().numpy()
        z_np = z_.detach().cpu().numpy()
        mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output = metric_rec(x.values, x_np)
        print(f'mae_re {mae_re} mape_re {mape_re}')
        x_df = pd.DataFrame(x_np, columns=x.columns, index=x.index)
        z_df = pd.DataFrame(z_np, index=x.index)
        z_df.columns = [str(col) + '_dense' for col in range(z_df.shape[1])]
        return x_df, z_df

    def fit_transform(self, x, y=None, val_x=None, val_y=None, epoch=60):
        data_dataloader = self.input_process(x, y)
        optim_ = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-5)
        data_loader = Data.DataLoader(data_dataloader, batch_size=256, worker_init_fn=np.random.seed(self.seed))
        for iter_count in range(epoch):
            total_vae_loss_ = 0
            self.model.train()
            for data, y in data_loader:
                data = data.to(self.device)
                noise = torch.randn(size=data.shape).to(self.device)
                optim_.zero_grad()
                x_rec, z, mu, logvar = self.model(data + 1e-2 * noise)
                rec_loss = self.model.loss(data, x_rec, mu, logvar)
                rec_loss.backward()
                optim_.step()
            total_vae_loss_ += rec_loss.detach().cpu().numpy()
            if iter_count % 5 == 0:
                # print(f'total_vae_loss_ {total_vae_loss_}')
                self.model.eval()
                x_tensor = torch.Tensor(val_x.values)
                x_hr, z_hr, _, _ = self.model(x_tensor)
                self.model.train()
                x_np = x_hr.detach().cpu().numpy()
                mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output = metric_rec(val_x.values, x_np)
                print(f'mae_re {mae_re} mape_re {mape_re}')
        self.model.eval()
        x_tensor = torch.Tensor(x.values)
        x_hr, z_hr, _, _ = self.model(x_tensor)
        x_np = x_hr.detach().cpu().numpy()
        z_np = z_hr.detach().cpu().numpy()
        x_df = pd.DataFrame(x_np, columns=x.columns, index=x.index)
        z_df = pd.DataFrame(z_np, index=x.index)
        z_df.columns = [str(col) + '_dense' for col in range(z_df.shape[1])]
        return x_df, z_df

    class MyDataset(Data.Dataset):
        def __init__(self,
                     data,
                     label=None,
                     random_seed=0):
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


def metric_rec(data, data_rec):
    mms = MinMaxScaler(feature_range=(0.1, 10))
    data_nor = mms.fit_transform(data)
    data_rec_nor = mms.transform(data_rec)
    mse_re = mean_squared_error(data, data_rec)
    mae_re = mean_absolute_error(data, data_rec)
    mape_re = mean_absolute_percentage_error(data_nor, data_rec_nor)
    r2_re = r2_score(data, data_rec)
    near_mae_re, x_new_output = near_mae(data, data_rec)
    return mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output


def near_mae(x_raw, x):
    x_new = []
    x_diff_min_list = []
    for index_ in range(x_raw.shape[0]):
        x_tmp = x - x_raw[index_]
        index_min = np.argmin(abs(x_tmp).mean(axis=1))
        x_new.append(x[index_min])
        x_diff_min = abs(x_tmp.mean(axis=1)).min()
        x_diff_min_list.append(x_diff_min)
    x_diff_min_mean = np.mean(x_diff_min_list)
    return x_diff_min_mean, np.array(x_new)

