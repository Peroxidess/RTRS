import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as Data
from preprocess.get_dataset import MyDataset
from sklearn.cluster import KMeans


class ActiveLearning():
    def __init__(self, args, shape_inp, name_method_AL='ours'):
        self.seed = args.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        args.ae_shape_inp = shape_inp
        args.ae_latent_dim = max(shape_inp // 2, 6)  # The dimension higher than the original data makes the hidden space separable
        args.ae_train_iterations = 60
        args.ae_beta = 0.01
        self.name_method = name_method_AL
        self.model_AL = BinsFuzz(args, name_method_AL)

    def preprocessing(self, train_x, val_x, train_label, val_label):
        self.model_AL.fit_transform_ae(train_x, val_x, train_label, val_label)

    def data_pool_init(self, train_x, label, method_init):
        label_data_index, unlabel_data_index = self.model_AL.data_pool_init(train_x, label, method_init, num_init=85)
        label_data = train_x.loc[label_data_index]
        unlabel_data = train_x.loc[unlabel_data_index]
        label_data_label = label.loc[label_data_index]
        unlabel_data_label = label.loc[unlabel_data_index]
        return label_data, label_data_label, unlabel_data, unlabel_data_label

    def data_choose(self, model, label_data, train_label, unlabel_data, unlabel_data_label, num_choose_AL, method_name_AL='ours', target={'label1': 'label'}, epoch_AL=0):
        pred_diff_index = self.model_AL.sample_choose(model, label_data, train_label, unlabel_data, unlabel_data_label, method_AL=method_name_AL,
                                                    num_choose=num_choose_AL, epoch_AL=epoch_AL)
        samples = unlabel_data.loc[pred_diff_index]
        samples_label = unlabel_data_label.loc[pred_diff_index][[target['label1']]]
        unlabel_data.drop(index=pred_diff_index, inplace=True)
        unlabel_data_label = unlabel_data_label.loc[unlabel_data.index]
        label_data = pd.concat([label_data, samples])
        label_data_label = pd.concat([train_label, samples_label])
        return label_data, label_data_label, unlabel_data, unlabel_data_label


class BinsFuzz():
    def __init__(self, args, name_method):
        self.args = args
        self.seed = self.args.seed + 1
        self.args.cuda = args.cuda and torch.cuda.is_available()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.name_method = name_method

    def data_pool_init(self, data, label, method_init, num_init=4):
        data = pd.DataFrame(self.hidden_pred_ae(data), index=data.index)
        if 'ours' in method_init:
            if 'singlesmall' in method_init:
                bins_stride_list = [2]
            elif 'singlelarge' in method_init:
                bins_stride_list = [31]
            else:
                bins_stride_list = [2, 3, 5, 7]
            data_bins_sum_df = pd.DataFrame([])
            for bins_stride in bins_stride_list:
                data_bins_sum = pd.DataFrame(self.scale_margin(data, label, bins_stride), index=data.index)
                data_bins_sum_df = pd.concat([data_bins_sum_df, data_bins_sum], axis=1)
            data_bins_sum_sort = data_bins_sum_df.max(axis=1).sort_values(ascending=False)
            data_index_BinsNum_max = data_bins_sum_sort.iloc[:num_init//2].index
            data_index_BinsNum_min = data_bins_sum_sort.iloc[-num_init//2:].index
            data_index_BinsNum = np.hstack((data_index_BinsNum_min, data_index_BinsNum_max))
            data_label = data.loc[data_index_BinsNum]
        elif 'clusters' in method_init:
            n_clusters = 2
            clf_0 = KMeans(n_clusters=n_clusters)
            clf_1 = KMeans(n_clusters=n_clusters)
            index_0 = label[label.values == 0].index
            index_1 = label[label.values == 1].index
            xx = set(index_0).intersection(set(index_1))
            data_c0 = clf_0.fit_predict(data.loc[index_0])
            data_labeled_index = []
            for n in range(n_clusters):
                data_labeled_index.append(list(np.argsort(clf_0.transform(data)[:, n])[::][:1]))
                data_labeled_index.append(list(np.argsort(clf_0.transform(data)[:, n])[::][-1:]))
            data_c1 = clf_1.fit_predict(data.loc[index_1])
            for n in range(n_clusters):
                data_labeled_index.append(list(np.argsort(clf_1.transform(data)[:, n])[::][:1]))
                data_labeled_index.append(list(np.argsort(clf_1.transform(data)[:, n])[::][-1:]))
            data_labeled_index = set(np.squeeze(data_labeled_index))
            data_label = data.iloc[list(data_labeled_index)]
        else:
            data_label = data.sample(n=num_init, random_state=self.seed)
        data_unlabel = data.drop(index=data_label.index)
        return data_label.index, data_unlabel.index

    def fit_transform_ae(self, dataset, dataset_val, train_label, val_label):
        data_labeled = dataset
        data_unlabeled = dataset
        data_labeled = MyDataset(data_labeled, None)
        data_unlabeled = MyDataset(data_unlabeled, None)
        optim_vae = optim.Adam(self.vae.parameters(), lr=1e-3, weight_decay=1e-5)
        if self.args.cuda:
            self.vae = self.vae.cuda()
        data_labeled_loader = Data.DataLoader(data_labeled, batch_size=256, worker_init_fn=np.random.seed(self.seed))
        data_unlabeled_loader = Data.DataLoader(data_unlabeled, batch_size=256, worker_init_fn=np.random.seed(self.seed))
        for iter_count in range(self.args.ae_train_iterations):
            total_vae_loss_ = 0
            for data_labeled_batch, data_unlabeled_batch in zip(data_labeled_loader, data_unlabeled_loader): #labeled data loader is the same as unlabeled data loader here
                if self.args.cuda:
                    data_labeled_batch = data_labeled_batch.cuda()
                    data_unlabeled_batch = data_unlabeled_batch.cuda()

                self.vae.train()
                optim_vae.zero_grad()
                recon_labeled, z, mu, logvar = self.vae(data_labeled_batch)
                rec_loss = self.vae.loss(data_labeled_batch, recon_labeled, mu, logvar, self.args.ae_beta)
                recon_unlab, unlab_z, unlab_mu, unlab_logvar = self.vae(data_unlabeled_batch)
                transductive_loss = self.vae.loss(data_unlabeled_batch,
                                                  recon_unlab, unlab_mu, unlab_logvar, self.args.ae_beta)

                total_vae_loss = rec_loss
                total_vae_loss.backward()
                optim_vae.step()
                total_vae_loss_ += total_vae_loss.detach().numpy()
            # print(f'train loss {total_vae_loss_}')
            self.vae.eval()
            recon_labeled, z, mu, logvar = self.vae(torch.Tensor(dataset_val.values))
            rec_loss = self.vae.loss(torch.Tensor(dataset_val.values), recon_labeled, mu, logvar, self.args.ae_beta)
            total_vae_loss = rec_loss
            if iter_count % 20 == 0:
                print(f'val loss {total_vae_loss}')
        pass

    def transform_ae(self, dataset):
        data_labeled = dataset
        data_labeled = MyDataset(data_labeled, None)
        self.vae.eval()
        if self.args.cuda:
            self.vae = self.vae.cuda()
        data_labeled_loader = Data.DataLoader(data_labeled, batch_size=64)
        for iter_count in range(self.args.train_iterations):
            for data_labeled_batch in data_labeled_loader:
                if self.args.cuda:
                    data_labeled_batch = data_labeled_batch.cuda()

                recon_labeled, z, mu, logvar = self.vae(data_labeled_batch[0])
                rec_loss = self.vae.loss(data_labeled_batch[0], recon_labeled, mu, logvar, self.args.beta)
                total_vae_loss = rec_loss
                print(f'loss {total_vae_loss}')

    def scale_abnormal(self, data, label, bins_stride=3):
        data_col_bins_abnormal_sum = pd.Series(np.zeros(shape=(data.shape[0],)), data.index)
        for col_index, values_ in data.iteritems():
            data_col = values_.values
            set_data_col = set(data_col)
            bins = max(int(np.floor(len(set_data_col) / bins_stride)), 2)
            histogram_data_col = np.histogram(data_col, bins=bins)
            index_hist_num_threshold = copy.deepcopy(histogram_data_col[0])
            index_hist_num_threshold[index_hist_num_threshold < index_hist_num_threshold.max() * 0.1] = 0
            index_hist_num = copy.deepcopy(index_hist_num_threshold)
            histogram_data_col_IndexNonzero = pd.Series(np.where(index_hist_num_threshold != 0)[0])
            histogram_data_col_IndexNonzero_pre = histogram_data_col_IndexNonzero.shift(1)
            histogram_data_col_IndexNonzero_post = histogram_data_col_IndexNonzero.shift(-1)
            histogram_data_col_IndexNonzero_pre.fillna(-1, inplace=True)
            histogram_data_col_IndexNonzero_post.fillna(index_hist_num_threshold.shape[0], inplace=True)
            indexzero_pre = histogram_data_col_IndexNonzero - histogram_data_col_IndexNonzero_pre - 1
            indexzero_post = histogram_data_col_IndexNonzero_post - histogram_data_col_IndexNonzero - 1
            indexzero_sum = indexzero_pre + indexzero_post
            index_hist_num[np.where(index_hist_num_threshold != 0)[0]] = indexzero_sum
            indexzero_max = index_hist_num.argsort()[::-1]
            abnormal_ratio = index_hist_num / bins
            indexzero_label = abnormal_ratio

            data_col_bins_abnormal = pd.cut(data_col, bins=bins, labels=indexzero_label, ordered=False)
            data_col_bins_abnormal = pd.Series(data_col_bins_abnormal, index=data.index).astype('float32')
            data_col_bins_abnormal_sum += data_col_bins_abnormal / data.shape[1]
        data_col_bins_abnormal_sum = pd.Series(data_col_bins_abnormal_sum, index=data.index)

        return data_col_bins_abnormal_sum

    def scale_margin(self, data, label, bins_stride=3):
        data_bins_sum = None
        for col_index, values_ in data.iteritems():
            data_col = values_.values
            set_data_col = set(data_col)
            bins = max(int(np.floor(len(set_data_col) / bins_stride)), 2)
            histogram_data_col = np.histogram(data_col, bins=bins)
            data_col_bins_num = pd.cut(data_col, bins=bins, labels=histogram_data_col[0], ordered=False).astype('int32')
            if data_bins_sum is not None:
                data_bins_sum += data_col_bins_num
            else:
                data_bins_sum = copy.deepcopy(data_col_bins_num)
        data_bins_sum_nor = data_bins_sum / bins_stride
        return data_bins_sum_nor

    def scale_fuzz(self, data, label, bins_stride=3):
        data_fuzz = None
        for col_index, values_ in data.iteritems():
            data_col = values_.values
            set_data_col = set(data_col)
            bins = max(int(np.floor(len(set_data_col) / bins_stride)), 2)

            histogram_data_col = np.histogram(data_col, bins=bins)

            histogram_per = pd.Series(histogram_data_col[1])
            histogram_post = pd.Series(histogram_data_col[1]).shift()
            histogram_ave_ = (histogram_per + histogram_post)/2
            histogram_ave_ = histogram_ave_.dropna(axis=0, how='any')

            data_col_bins = pd.cut(data_col, bins=bins, labels=histogram_ave_, ordered=False)
            data_col_bins = pd.Series(data_col_bins)
            if data_fuzz is None:
                data_fuzz = data_col_bins
            else:
                data_fuzz = np.vstack((data_fuzz, data_col_bins))
        data_fuzz = np.transpose(data_fuzz)
        data_fuzz_df = pd.DataFrame(data_fuzz, index=data.index, columns=data.columns)
        return data_fuzz_df

    def sample_choose(self, model, label_data, train_label, unlabel_data, unlabel_data_label, method_AL='ours', num_choose=80, epoch_AL=0):
        hidden_z_tra_unlabel = self.hidden_pred_ae(unlabel_data)
        if 'ours' in method_AL:
            try:
                pred_train = model.predict_proba(hidden_z_tra_unlabel)
            except:
                pred_train = model.predict(hidden_z_tra_unlabel)
            if 'singlesmall' in method_AL:
                bins_stride_list = [2]
            elif 'singlelarge' in method_AL:
                bins_stride_list = [31]
            else:
                bins_stride_list = [7, 11, 13]
            pred_diff_ratio_df_all = pd.DataFrame([])
            for bins_stride in bins_stride_list:
                hidden_z_tra_unlabel_fuzz = self.scale_fuzz(hidden_z_tra_unlabel, unlabel_data_label, bins_stride)
                hidden_z_tra_unlabel_abnormal = self.scale_abnormal(hidden_z_tra_unlabel, unlabel_data_label, bins_stride)
                try:
                    pred_train_fuzz = model.predict_proba(hidden_z_tra_unlabel_fuzz)
                except:
                    pred_train_fuzz = model.predict(hidden_z_tra_unlabel_fuzz)

                pred_train_ = 1 - pred_train
                pred_rejection = pred_train_
                pred_train_fuzz_ = 1 - pred_train_fuzz
                pred_rejection_fuzz = pred_train_fuzz_
                if 'nec' in method_AL:
                    pred_diff2 = pred_rejection - pred_rejection_fuzz
                else:
                    pred_diff = pred_train - pred_train_fuzz

                pred_diff_ratio = pred_diff / pred_train
                pred_diff_df = pd.Series(abs(pred_diff), index=unlabel_data.index)
                pred_diff_ratio_df = pd.Series(abs(pred_diff_ratio), index=unlabel_data.index)

                data_bins_sum = pd.Series(self.scale_margin(unlabel_data, unlabel_data_label, bins_stride), index=unlabel_data.index)
                if 'nondiversity' in method_AL:
                    initial_criteria = pred_diff_df
                    Intermediate_criteria = pred_diff_df
                    final_criteria = pred_diff_ratio_df
                elif 'nonuncertainty' in method_AL:
                    initial_criteria =hidden_z_tra_unlabel_abnormal + 1e-2 * (data_bins_sum / data_bins_sum.max())
                    Intermediate_criteria = hidden_z_tra_unlabel_abnormal + 1e-2 * (data_bins_sum / data_bins_sum.max())
                    final_criteria = hidden_z_tra_unlabel_abnormal + 1e-2 * (data_bins_sum / data_bins_sum.max())
                else:
                    initial_criteria = hidden_z_tra_unlabel_abnormal
                    Intermediate_criteria = pred_diff_df
                    final_criteria = pred_diff_ratio_df
                if epoch_AL <= 5:
                    pred_diff = initial_criteria
                elif 5<epoch_AL<=120:
                    pred_diff = Intermediate_criteria
                else:
                    pred_diff = final_criteria
                pred_diff_ratio_df_all = pd.concat([pred_diff_ratio_df_all, pred_diff], axis=1)
            pred_diff_ratio_sort = pred_diff_ratio_df_all.max(axis=1).sort_values(ascending=False)
            index_ = pred_diff_ratio_sort.index[:num_choose]
            pred_diff_index = index_
            unlabel_data_label_plot = copy.deepcopy(unlabel_data_label)
            unlabel_data_label_plot.loc[pred_diff_index] = 2
        elif 'entropy' in method_AL:
            try:
                pred_train = model.predict_proba(hidden_z_tra_unlabel)
            except:
                pred_train, _ = model.predict(hidden_z_tra_unlabel.values)
            pred_train[np.where(pred_train == 0)] = 1e-8
            entropy_list = -1 * pred_train[:, 0] * np.log(pred_train[:, 0]) - pred_train[:, 1] * np.log(pred_train[:, 1])
            entropy_df = pd.Series(entropy_list, index=unlabel_data.index)
            pred_diff_index = np.argsort(entropy_list)[::-1][:num_choose]
            pred_diff_index = entropy_df.iloc[pred_diff_index].index
        elif 'losspred' in method_AL:
            pred_loss, features = model.predict(hidden_z_tra_unlabel.values)
            loss_tensor = model.loss_pred(features)
            loss_pred = loss_tensor.detach().numpy()
            loss_df = pd.Series(abs(loss_pred.reshape(-1, )), index=hidden_z_tra_unlabel.index)
            loss_sort = loss_df.sort_values(ascending=False)
            pred_diff_index = loss_sort.index[:num_choose]
        elif 'confidence' in method_AL: # same as entropy
            try:
                pred_train = model.predict_proba(hidden_z_tra_unlabel)
            except:
                pred_train, _ = model.predict(hidden_z_tra_unlabel.values)

            pred_confusion = (1 + pred_train - pred_train.max(axis=1).reshape(-1, 1)).sum(axis=1).reshape(-1, )
            pred_train_ = pred_confusion
            pred_train_df = pd.Series(pred_train_, index=hidden_z_tra_unlabel.index)
            pred_train_sort = pred_train_df.sort_values()
            pred_diff_index = pred_train_sort.index[:num_choose]
        else:
            pred_diff_index = unlabel_data.sample(n=num_choose, random_state=self.seed).index
        return pred_diff_index
