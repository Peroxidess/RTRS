import copy
import time
import re
import pandas as pd
import arguments
import os
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from preprocess import load_data
from preprocess.get_dataset import DataPreprocessing
from preprocess.missing_values_imputation import MVI
from preprocess.imbalance_process import IMB
from model.evaluate import q_plot, EHRDM_metric_show
from model.ActiveLearning import ActiveLearning
from preprocess.representation_learning import RepresentationLearning
from model.ReinforcementLearning import RL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def model_tra_eval_RL(train_x_filled_ger, train_label_ger, val_set, val_label, test_z, test_label, epoch_RL, epoch_AL, k, seed=0):
    state_dim = train_x_filled_ger.shape[1] - 3 - train_x_filled_ger.filter(regex=r'action').shape[1]
    model = RL(state_dim=state_dim,
               action_dim=train_x_filled_ger.filter(regex=r'action').shape[1], max_episodes=epoch_RL, seed=args.seed,
               ckpt_dir='', method_RL=method_name_RL)
    model.store_data(train_x_filled_ger, train_label_ger)
    model.agent.learn_class(train_x_filled_ger.drop(columns=train_x_filled_ger.filter(regex=r'时间|label').columns),
                            train_label_ger[['label1']],
                            val_set.drop(columns=val_set.filter(regex=r'时间|label').columns), val_label[['label1']])

    q_test_df = model.eval(copy.deepcopy(test_z), test_label, type_task='RL')
    q_table = q_plot(copy.deepcopy(q_test_df), ['q_record'], name_method=method_name_RL, fold=k)
    mortality_ratio_q = q_table['mortality_ratio_q'].mean()

    mortality_ratio_q_record0 = q_table[q_table['label1'] == 0]['mortality_ratio_q_record'].mean()
    mortality_ratio_q_record1 = q_table[q_table['label1'] == 1]['mortality_ratio_q_record'].mean()
    print(
        f'epoch_AL {epoch_AL} mortality_ratio_q_record_diff ratio {mortality_ratio_q / (mortality_ratio_q_record1 - mortality_ratio_q_record0)}')

    q_train_df_js = q_table.to_json(indent=4)
    q_train_df_js_records = q_table.to_json(orient='records', indent=4)
    q_train_df_js_dict = json.loads(q_train_df_js)
    q_train_df_js_records_dict = json.loads(q_train_df_js_records)
    return model, q_train_df_js_dict, q_train_df_js_records_dict


def run(train_data, test_data, target, args, trial) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    target: dict
    if args.test_ratio == 0 or not test_data.empty:
        train_set = train_data
        test_set = test_data
    else:
        train_set, test_set = train_test_split(train_data, test_size=args.test_ratio, random_state=args.seed, shuffle=False)

    metric_df_all = pd.DataFrame([])
    pred_train_df_all = pd.DataFrame([])
    pred_test_df_all = pd.DataFrame([])
    metric_AL_AllFlod = pd.DataFrame([])
    history_df = pd.DataFrame([])
    kf = KFold(n_splits=args.n_splits)
    for k, (train_index, val_index) in enumerate(kf.split(train_set)):
        train_set_cv = train_set.iloc[train_index]
        val_set_cv = train_set.iloc[val_index]
        test_set_cv = copy.deepcopy(test_set)

        dp = DataPreprocessing(train_set_cv, val_set_cv, test_set_cv, None, seed=args.seed,
                               flag_label_onehot=False,
                               flag_ex_null=True, flag_ex_std_flag=False, flag_ex_occ=False,
                               flag_ca_co_sel=True, flag_ca_fac=True, flag_onehot=True, flag_nor=True,
                               flag_feat_emb=False, flag_RUS=False, flag_confusion=False, flaq_save=False)
        if args.Flag_DataPreprocessing:
            train_set_cv, val_set_cv, test_set_cv, ca_col, co_col, nor = dp.process()
            # train_set_cv.to_csv(f'./Train_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
            # val_set_cv.to_csv(f'./Val_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
            # test_set_cv.to_csv(f'./Test_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
        else:
            train_set_cv = pd.read_csv(f'../DataSet/{args.task_name}/processed_flod/data_kflod/Train_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            val_set_cv = pd.read_csv(f'../DataSet/{args.task_name}/processed_flod/data_kflod/Val_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            test_set_cv = pd.read_csv(f'../DataSet/{args.task_name}/processed_flod/data_kflod/Test_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            nor = None

        col_ = train_set_cv.columns
        val_set_cv = val_set_cv[col_]
        test_set_cv = test_set_cv[col_]
        ca_col = train_set_cv.filter(regex=r'sparse').columns.tolist()
        co_col = train_set_cv.filter(regex=r'dense').columns.tolist()

        train_label = train_set_cv[[x for x in target.values()]]
        val_label = val_set_cv[[x for x in target.values()]]
        test_label = test_set_cv[[x for x in target.values()]]
        train_x = train_set_cv.drop(columns=target.values())
        val_x = val_set_cv.drop(columns=target.values())
        test_x = test_set_cv.drop(columns=target.values())

        print(f'train_x shape {train_x.shape} | val_x shape {val_x.shape} | test_x shape {test_x.shape}')

        # missing values imputation start
        mr = args.missing_ratio
        if args.Flag_MVI:
            col_ = train_x.filter(regex=r'dense|sparse').columns
            col_sp = train_x.columns.drop(col_)

            mvi = MVI(col_.shape[0], co_col, ca_col, args.task_name, target, args.seed, method=args.method_MVI)
            nan_ratio = mvi.show_nan_ratio(train_x)
            if mr != 1.:
                train_x_mn, train_label, index_ManualNan_train = mvi.manual_nan(train_x[col_], train_label, args.dict_ManualRatio, k, type_data='Train', flag_saving=args.Flag_Mask_Saving)
                train_x = train_x.loc[train_x_mn.index]
                val_x_mn, val_label, index_ManualNan_val = mvi.manual_nan(val_x[col_], val_label, args.dict_ManualRatio, k, type_data='Val', flag_saving=args.Flag_Mask_Saving)
                test_x_mn, test_label, index_ManualNan_test = mvi.manual_nan(test_x[col_], test_label, args.dict_ManualRatio, k, type_data='Test', flag_saving=args.Flag_Mask_Saving)
            else:
                train_x_mn = train_x
                val_x_mn = val_x
                test_x_mn = test_x

            train_x_filled, train_label = mvi.fit_transform(train_x_mn[col_], train_x[col_], train_label, k, mr, args.test)
            val_x_filled = mvi.transform(val_x_mn[col_], val_label)
            test_x_filled = mvi.transform(test_x_mn[col_], test_label)
            train_x_filled = pd.concat([train_x_filled, train_x[col_sp]], axis=1)
            val_x_filled = pd.concat([val_x_filled, val_x[col_sp]], axis=1)
            test_x_filled = pd.concat([test_x_filled, test_x[col_sp]], axis=1)
        else: # loding imputed data
            path_root = f'../DataSet/mimic/filled/mr{mr}/'
            path_list_label = os.listdir(path_root)
            path_train = None
            for path in path_list_label:
                path_re = re.search(r'TrainFilled_mvi\[{}\]_KFlod\[{}\]'.format(args.method_mvi, k), path)
                if path_re:
                    path_train = path_root + path_re.string
                path_re = re.search(r'ValFilled_mvi\[{}\]_KFlod\[{}\]'.format(args.method_mvi, k), path)
                if path_re:
                    path_val = path_root + path_re.string
                path_re = re.search(r'TestFilled_mvi\[{}\]_KFlod\[{}\]'.format(args.method_mvi, k), path)
                if path_re:
                    path_test = path_root + path_re.string
            if path_train is None:
                continue
            train_x_filled = pd.read_csv(path_train, index_col=['index'])
            train_label = train_x_filled[['label_dead']]
            train_label.rename(columns={'label_dead': 'label1'}, inplace=True)
            train_x_filled.drop(columns=['label_dead'], inplace=True)
            val_x_filled = pd.read_csv(path_val, index_col=['index'])
            val_label = val_x_filled[['label_dead']]
            val_label.rename(columns={'label_dead': 'label1'}, inplace=True)
            val_x_filled.drop(columns=['label_dead'], inplace=True)
            test_x_filled = pd.read_csv(path_test, index_col=['index'])
            test_label = test_x_filled[['label_dead']]
            test_label.rename(columns={'label_dead': 'label1'}, inplace=True)
            test_x_filled.drop(columns=['label_dead'], inplace=True)
        # missing values imputation end

        # Dimension reduction Start
        if args.method_ReprL_ is not None:
            represent = RepresentationLearning(train_x_filled.shape[1]-5, args.method_ReprL_)
            train_z = represent.fit_transform(train_x_filled, train_label[target['label1']], val_x_filled, val_label[target['label1']])
            val_z = represent.transform(val_x_filled, val_label[target['label1']])
            test_z = represent.transform(test_x_filled, test_label[target['label1']])
        else:
            train_z, val_z, test_z = train_x_filled, val_x_filled, test_x_filled
        # Dimension reduction End

        # imbalanced learning start
        if args.Deal_Imbalance_Method == 'False' or args.Deal_Imbalance_Method == 'cgain' or args.Deal_Imbalance_Method == 'ours':
            train_x_filled_ger, train_label_ger = train_z, train_label
        else:
            imb = IMB(args.Deal_Imbalance_Method)
            train_x_filled_ger, train_label_ger = imb.fit_transform(train_z, train_label)
        # imbalanced learning end

        # RL
        method_name_RL = args.method_name_RL
        epoch_RL = [300, 300, 300, 300, 300]
        # if method_name_RL is not None:
        if args.method_AL is None:
            state_dim = train_x_filled_ger.shape[1] - 3 - train_x_filled_ger.filter(regex=r'action').shape[1]
            model = RL(state_dim=state_dim,
                       action_dim=train_x_filled_ger.filter(regex=r'action').shape[1], max_episodes=epoch_RL[k], seed=args.seed, ckpt_dir='', method_RL=method_name_RL)

            model.store_data(train_x_filled_ger, train_label_ger)
            model.agent.learn_class(train_x_filled_ger.drop(columns=train_x_filled_ger.filter(regex=r'时间|label').columns), train_label_ger[['label1']],
                                    val_z.drop(columns=val_z.filter(regex=r'时间|label').columns), val_label[['label1']])
            critic_loss_all, actor_loss_all, q_test_list, q_test_df_mean_epoch = model.learn(val_z, val_label)
            history_df_tmp = pd.DataFrame(critic_loss_all, columns=[f'critic_{k}'])
            history_df_tmp[f'action_{k}'] = actor_loss_all
            history_df = pd.concat([history_df, history_df_tmp], axis=1)

            q_test_df = model.eval(copy.deepcopy(test_z), test_label, type_task='RL')
            q_table = q_plot(copy.deepcopy(q_test_df), ['q_record'], name_method=method_name_RL, fold=k)
            mortality_ratio_q = q_table['mortality_ratio_q'].mean()
            print(f'mortality_ratio_q {mortality_ratio_q}')

            q_train_df_js = q_table.to_json(indent=4)
            q_train_df_js_records = q_table.to_json(orient='records', indent=4)
            if k == 0:
                try:
                    with open(f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_.json') as f:
                        q_train_df_js_all_ = json.load(f)
                        # q_train_df_js_all_ = json.loads(q_train_df_js_all_)
                        q_train_df_js_all_['metric'][f'fold {k}'] = {'all': json.loads(q_train_df_js)}
                        # q_train_df_js_all = json.dumps(q_train_df_js_all_, indent=4)

                    with open(
                            f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_record_.json') as f:
                        q_train_df_js_all_records_ = json.load(f)
                        # q_train_df_js_all_records_ = json.loads(q_train_df_js_all_records_)
                        q_train_df_js_all_records_['metric'][f'fold {k}'] = {'all': json.loads(q_train_df_js_records)}
                        # q_train_df_js_all_records = json.dumps(q_train_df_js_all_records_, indent=4)
                except:
                    q_train_df_js_all_ = {'args': vars(args), 'metric': {f'fold {k}': {'all': json.loads(q_train_df_js)}}}
                    q_train_df_js_all_records_ = {'args': vars(args), 'metric': {f'fold {k}':  {'all': json.loads(q_train_df_js_records)}}}
            else:
                with open(f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_.json') as f:
                    q_train_df_js_all_ = json.load(f)
                    # q_train_df_js_all_ = json.loads(q_train_df_js_all_)
                    try:
                        q_train_df_js_all_['metric'][f'fold {k}']['all'] = json.loads(q_train_df_js)
                    except:
                        q_train_df_js_all_['metric'][f'fold {k}'] = {'all': json.loads(q_train_df_js)}
                    # q_train_df_js_all = json.dumps(q_train_df_js_all_, indent=4)

                with open(f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_record_.json') as f:
                    q_train_df_js_all_records_ = json.load(f)
                    # q_train_df_js_all_records_ = json.loads(q_train_df_js_all_records_)
                    try:
                        q_train_df_js_all_records_['metric'][f'fold {k}']['all'] = json.loads(q_train_df_js_records)
                    except:
                        q_train_df_js_all_records_['metric'][f'fold {k}'] = {'all': json.loads(q_train_df_js_records)}
                    # q_train_df_js_all_records = json.dumps(q_train_df_js_all_records_, indent=4)

            with open(f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_record_.json', 'w') as f:
                json.dump(q_train_df_js_all_records_, f)
            with open(f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_.json', 'w') as f:
                json.dump(q_train_df_js_all_, f)
            if k != 0:
                EHRDM_metric_show(q_train_df_js_all_)
            # RL end

        # active learning start
        if args.method_AL is not None:

            al = ActiveLearning(args, train_x_filled_ger.shape[1], args.method_AL)
            label_data, label_data_label, unlabel_data, unlabel_data_label = al.data_pool_init(train_x_filled_ger, train_label_ger, args.method_AL)

            metric_AL_iter = pd.DataFrame([])
            metric = pd.DataFrame([])

            args.num_choose_AL = 35

            AL_dict = {}
            AL_records_dict = {}
            for epoch_AL in range(50):
                hidden_z_tra = label_data
                hidden_z_val = val_z
                hidden_z_test = test_z

                model, q_train_df_js_dict, q_train_df_js_records_dict = model_tra_eval_RL(hidden_z_tra, label_data_label,
                                                                                          hidden_z_val, val_label,
                                                                                          hidden_z_test, test_label,
                                                                                          epoch_RL[k],
                                                                                          epoch_AL, k=k)

                AL_dict[f'epoch {epoch_AL}'] = q_train_df_js_dict
                AL_records_dict[f'epoch {epoch_AL}'] = q_train_df_js_records_dict

                metric_AL_iter = pd.concat([metric_AL_iter, metric])

                label_data, label_data_label, unlabel_data, unlabel_data_label = al.data_choose(model, label_data, label_data_label,
                                                                                                unlabel_data, unlabel_data_label,
                                                                                                num_choose_AL=args.num_choose_AL,
                                                                                                method_name_AL=args.method_AL,
                                                                                                target=target,
                                                                                                epoch_AL=epoch_AL)

            if k == 0:
                try:
                    with open(
                            f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_{args.method_AL}.json') as f:
                        q_train_df_js_all_ = json.load(f)
                        try:
                            q_train_df_js_all_['metric'][f'fold {k}']['AL'] = AL_dict
                        except:
                            pass
                    with open(
                            f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_record_{args.method_AL}.json') as f:
                        q_train_df_js_all_records_ = json.load(f)
                        try:
                            q_train_df_js_all_records_['metric'][f'fold {k}']['AL'] = AL_records_dict
                        except:
                            pass
                except:
                    q_train_df_js_all_ = {'args': vars(args),
                                          'metric': {f'fold {k}': {
                                              'AL': {f'epoch {epoch_AL}': json.loads(q_train_df_js)}}}}
                    q_train_df_js_all_records_ = {'args': vars(args), 'metric': {
                        f'fold {k}': {'AL': {f'epoch {epoch_AL}': json.loads(q_train_df_js_records)}}}}
            else:
                try:
                    with open(
                            f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_{args.method_AL}.json') as f:
                        q_train_df_js_all_ = json.load(f)
                        try:
                            q_train_df_js_all_['metric'][f'fold {k}']['AL'] = AL_dict
                        except:
                            pass
                    with open(
                            f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_record_{args.method_AL}.json') as f:
                        q_train_df_js_all_records_ = json.load(f)
                        try:
                            q_train_df_js_all_records_['metric'][f'fold {k}']['AL'] = AL_records_dict
                        except:
                            pass
                except:
                    print(f'wrong')

            with open(
                    f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_record_{args.method_AL}.json',
                    'w') as f:
                json.dump(q_train_df_js_all_records_, f)
            with open(
                    f'metric_{args.method_MVI}_{args.method_ReprL_}_{args.method_name_RL}_js_{args.method_AL}.json',
                    'w') as f:
                json.dump(q_train_df_js_all_, f)
            if k != 0:
                pass

            metric_AL_iter.columns = pd.MultiIndex.from_product([[f'flod {10 * trial + k}'], metric_AL_iter.columns])
            metric_AL_AllFlod = pd.concat([metric_AL_AllFlod, metric_AL_iter], axis=1)
    return metric_df_all, pred_train_df_all, pred_test_df_all, metric_AL_AllFlod


if __name__ == "__main__":
    args = arguments.get_args()

    test_prediction_all = pd.DataFrame([])
    train_prediction_all = pd.DataFrame([])
    history_df_all = pd.DataFrame([])
    metric_AllFold = pd.DataFrame([])
    metric_AllRun = pd.DataFrame([])

    for trial in range(args.nrun):
        print('rnum : {}'.format(trial))
        args.seed = (trial * 55) % 2022  # a different random seed for each run

        # data fetch
        # input: file path
        # output: data with DataFrame
        train_data, test_data, target = load_data.data_load(args.task_name, args.seed)
        for method_MVI in ['ours_ger_not']:
            args.method_MVI = method_MVI
            for method_ReprL_ in ['ours']:
                args.method_ReprL_ = method_ReprL_
                for method_name_RL in ['ADRL']:
                    args.method_name_RL = method_name_RL
                    for method_name_RL_post in ['LC']:
                        args.method_name_RL = method_name_RL + '_' + method_name_RL_post
                        # run model
                        # input: train_data
                        # output: metric, train_prediction, test_prediction
                        metric_df, train_prediction, test_prediction, metric_AL_AllFlod = run(train_data, test_data, target, args,
                                                                                          trial)

        metric_AllFold = pd.concat([metric_AllFold, metric_df], axis=0)
        test_prediction_all = pd.concat([test_prediction_all, test_prediction], axis=1)
        train_prediction_all = pd.concat([train_prediction_all, train_prediction], axis=1)
        metric_AllRun = pd.concat([metric_AllRun, metric_AL_AllFlod], axis=1)

        local_time = time.strftime("%m_%d_%H_%M", time.localtime())

    pass
pass
