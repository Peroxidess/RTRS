import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from model.evaluate import EHRDM_metric_show

# color_list = sns.color_palette('hls', 16)
nor = MinMaxScaler()


def initPlot():
    # rc = {'font.sans-serif': 'SimSun',
    #       'axes.unicode_minus': False}
    # myfont = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=10.5)
    sns.set(font_scale=1.0, style="white", palette="muted",
            color_codes=True, font='Times New Roman')  # set( )设置主题，调色板更常用
    config = {
    #     "font.family": 'serif',
    #     "font.size": 10.5,
        "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    #     "font.serif": ['STZhongsong'],  # 华文中宋
    #     'axes.unicode_minus': False  # 处理负号，即-号
    }
    plt.rcParams.update(config)
    # plt.rcParams['font.sans-serif'] = 'SimSun'
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
    # sns.set(font=myfont.get_name())


def foo(metric_df_single):
    columns_q = metric_df_single.filter(regex=r'^q').columns
    for columns_ in columns_q:
        metric_df_single[columns_] = nor.fit_transform(metric_df_single[columns_].values.reshape(-1, 1))
    q_test_df_class0 = metric_df_single[metric_df_single['label1'] == 0]
    q_test_df_class1 = metric_df_single[metric_df_single['label1'] == 1]
    q_test_df_class0_sort = q_test_df_class0.sort_values(by='q_record', ascending=False)
    action_mae_list = []
    for n in range(7, 700, 7):
        q_test_df_mortality_sort_ = q_test_df_class0_sort.iloc[:n]
        action_mae = abs(
            q_test_df_mortality_sort_[q_test_df_mortality_sort_['label1'] == 0]['action_diff']).mean()
        action_mae_list.append(action_mae)
    metric_groupby_single.loc[fold, 'MAE 1%'] = action_mae_list[0]
    metric_groupby_single.loc[fold, 'MAE 5%'] = action_mae_list[4]
    metric_groupby_single.loc[fold, 'MAE 10%'] = action_mae_list[9]
    metric_groupby_single.loc[fold, 'MAE 20%'] = action_mae_list[19]
    metric_groupby_single.loc[fold, 'MAE all'] = action_mae_list[-1]
    metric_groupby_single.loc[fold, 'fold'] = fold
    metric_groupby_single.loc[fold, 'method'] = 'ours_ger_not' + '_' + 'ours' + '_' + 'ADRL_RRC'
    q_test_df_class0_q_record = q_test_df_class0['q_record'].mean()
    q_test_df_class1_q_record = q_test_df_class1['q_record'].mean()
    mortality_ratio_q_record0 = q_test_df_class0['mortality_ratio_q_record'].mean()
    mortality_ratio_q_record1 = q_test_df_class1['mortality_ratio_q_record'].mean()
    q_test_df_class01_diff = q_test_df_class0_q_record - q_test_df_class1_q_record

    metric_groupby_single.loc[fold, 'q record 0'] = q_test_df_class0_q_record
    metric_groupby_single.loc[fold, 'q record 1'] = q_test_df_class1_q_record
    metric_groupby_single.loc[fold, 'relapse_ratio_q_record 0'] = mortality_ratio_q_record0
    metric_groupby_single.loc[fold, 'relapse_ratio_q_record 1'] = mortality_ratio_q_record1
    metric_groupby_single.loc[fold, 'q record diff'] = q_test_df_class0_q_record - q_test_df_class1_q_record
    metric_groupby_single.loc[fold, 'q r'] = metric_df_single['q'].mean() * q_test_df_class01_diff
    metric_groupby_single.loc[fold, 'relapse ratio diff ratio'] = metric_df_single['mortality_ratio_q'].mean() / \
                                                                    (
                                                                                mortality_ratio_q_record1 - mortality_ratio_q_record0)
    return metric_groupby_single


initPlot()

# Active Learning plot
# color_list = sns.color_palette('hls', 2)
# fig, axes = plt.subplots(4, 1, figsize=(8, 12), squeeze=False)  # grid subplots
# axes = axes.flatten()
# path_ = 'metric_ours_ger_not_ours_ADRL_LC'
# path_AL = 'ours'
# path_AL_dict = {'ours': 'MSB', 'random': 'Baseline'}
# metric_df_all_method = pd.DataFrame([])
# for count_method, path_AL in enumerate(['ours', 'random']):
#     metric_df_all_fold = pd.DataFrame([])
#     with open(f'{path_}_js_{path_AL}.json') as f:
#         q_train_df_js_all_ = json.load(f)
#     with open(f'{path_}_js_record_{path_AL}.json') as f:
#         q_train_df_js_all_records_ = json.load(f)
#
#     for fold in range(5):
#         metric_groupby_single = pd.DataFrame([])
#         # if fold == 4:
#         #     continue
#         dict_ = q_train_df_js_all_records_['metric'][f'fold {fold}']['AL']
#         metric_last = pd.json_normalize(q_train_df_js_all_records_['metric'][f'fold {fold}']['all'])
#         for count, (k_, v_) in enumerate(dict_.items()):
#             metric_df_single = pd.json_normalize(v_)
#             metric_groupby_single_ = foo(metric_df_single)
#             metric_groupby_single_['epoch'] = count
#             metric_df_all_fold = pd.concat([metric_df_all_fold, metric_groupby_single_], axis=0)
#         metric_groupby_single_pro_last = foo(metric_last)
#         metric_groupby_single_pro_last['epoch'] = count + 1
#         metric_df_all_fold = pd.concat([metric_df_all_fold, metric_groupby_single_], axis=0)
#     metric_df_all_fold['method'] = path_AL
#     metric_df_all_method = pd.concat([metric_df_all_method, metric_df_all_fold])
#     metric_df_all_method.reset_index(drop=True, inplace=True)
#
#     xlim_list = [[0, 49], [0, 20], [15, 35], [30, 49]]
#     ylim_list = [[0, 13], [1.1, 13], [0.3, 2], [0, 1.5]]
#     ci_list = [95, 90, 85, 80]
#     labelpad_list = [14, 6, 10, 10]
#     for i in range(4):
#         plt.sca(axes[i])
#         ax = sns.regplot(data=metric_df_all_method, x='epoch', y='relapse ratio diff ratio', order=14, ci=ci_list[i],
#                      truncate=True, n_boot=500, x_estimator=np.mean, scatter=False, label=path_AL_dict[path_AL],
#                      line_kws={"linewidth": 1.5, "color": color_list[count_method]})
#         # ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
#         plt.ylim(ylim_list[i][0], ylim_list[i][1])
#         plt.xlim(xlim_list[i][0], xlim_list[i][1])
#         if i == 0:
#             interval_x = 5
#             interval_y = 3
#             ax.set_yticks([int(i) for i in range(ylim_list[i][0], ylim_list[i][1] + interval_y, interval_y)])
#         else:
#             interval_x = 2
#             interval_y = 2
#         if i != 3:
#             plt.xlabel('')
#         else:
#             plt.xlabel('Epoch', labelpad=5)
#
#         plt.ylabel('NERR', labelpad=labelpad_list[i])
#         ax.set_xticks([i for i in range(xlim_list[i][0], xlim_list[i][1] + interval_x - 1, interval_x)])
#         plt.legend()
# # plt.show()
# # xx = 1
# plt.savefig(f'LearningCurve_NERR.png', dpi=660, bbox_inches='tight', transparent=True)
# # Active Learning plot

metric_df_all_methods = pd.DataFrame([])
metric_groupby_single = pd.DataFrame([])
metric_groupby_all = pd.DataFrame([])
metric_save = pd.DataFrame([], index=['MVIIL-GAN, MSLR, ADRL, RRC',
                                      'MVIIL-GAN, MSLR, ADRL, None',
                                      'MVIIL-GAN, MSLR, TD3-BC, RRC',
                                      'MVIIL-GAN, MSLR, TD3-BC, None',
                                      'MVIIL-GAN, None, ADRL, RRC',
                                      'MVIIL-GAN, None, ADRL, None',
                                      'MVIIL-GAN, None, TD3-BC, RRC',
                                      'MVIIL-GAN, None, TD3-BC, None',
                                      'None, MSLR, ADRL, RRC',
                                      'None, MSLR, ADRL, None',
                                      'None, MSLR, TD3-BC, RRC',
                                      'None, MSLR, TD3-BC, None',
                                      'None, None, ADRL, RRC',
                                      'None, None, ADRL, None',
                                      'None, None, TD3-BC, RRC',
                                      'None, None, TD3-BC, None'], columns=['relapse_ratio_nor_q', 'MAE 1%', 'MAE 5%', 'MAE 10%', 'MAE 20%'])
for method_MVI in ['ours_ger_not', 's']:
    for method_ReprL_ in ['ours', 'None']:
        for method_name_RL in ['ADRL', 'TD3-BC']:
            for method_name_RL_ in ['LC', 'None']:
                path_ = f'metric_{method_MVI}_{method_ReprL_}_{method_name_RL}_{method_name_RL_}'
                # if path_ != 'metric_ours_ger_not_ours_ADRL_RRC':
                #     xx = 1
                #         or path_ == 'metric_s_None_ADRL_LC' \
                #         or path_ == 'metric_s_None_ADRL_None' \
                #         or path_ == 'metric_s_None_TD3-BC_None':
                        # or path_ == 'metric_ours_ger_not_None_ADRL_None'\
                    # continue
                metric_df_all_fold = pd.DataFrame([])
                with open(f'{path_}_js_.json') as f:
                    q_train_df_js_all_ = json.load(f)
                    # metric_df = pd.read_json(q_train_df_js_all_)
                    # q_train_df_js_all_ = json.loads(q_train_df_js_all_)
                with open(f'{path_}_js_record_.json') as f:
                    q_train_df_js_all_records_ = json.load(f)
                    # q_train_df_js_all_records_ = json.loads(q_train_df_js_all_records_)

                # for fold in range(5):
                #     tmp = q_train_df_js_all_['metric'][f'fold {fold}']
                #     q_train_df_js_all_['metric'][f'fold {fold}'] = {'all': tmp}
                #     tmp_ = q_train_df_js_all_records_['metric'][f'fold {fold}']
                #     q_train_df_js_all_records_['metric'][f'fold {fold}'] = {'all': tmp_}
                # with open(f'metric_{method_MVI}_{method_ReprL_}_{method_name_RL}_{method_name_RL_}_js_.json', 'w') as f:
                #     json.dump(q_train_df_js_all_, f)
                # with open(f'metric_{method_MVI}_{method_ReprL_}_{method_name_RL}_{method_name_RL_}_js_record_.json', 'w') as f:
                #     json.dump(q_train_df_js_all_records_, f)

                for fold in range(5):
                    # if fold != 3:
                    #     continue
                    metric_df_single = pd.json_normalize(q_train_df_js_all_records_['metric'][f'fold {fold}']['all'])
                    metric_df_single['fold'] = fold
                    r2
                    columns_q = metric_df_single.filter(regex=r'^q').columns
                    for columns_ in columns_q:
                        metric_df_single[columns_] = nor.fit_transform(metric_df_single[columns_].values.reshape(-1, 1))
                    q_test_df_class0 = metric_df_single[metric_df_single['label1'] == 0]
                    q_test_df_class1 = metric_df_single[metric_df_single['label1'] == 1]
                    q_test_df_class0_sort = q_test_df_class0.sort_values(by='q_record', ascending=False)
                    action_mae_list = []
                    for n in range(7, 700, 7):
                        q_test_df_mortality_sort_ = q_test_df_class0_sort.iloc[:n]
                        action_mae = abs(
                            q_test_df_mortality_sort_[q_test_df_mortality_sort_['label1'] == 0]['action_diff']).mean()
                        action_mae_list.append(action_mae)
                    metric_groupby_single.loc[fold, 'MAE 1%'] = action_mae_list[0]
                    metric_groupby_single.loc[fold, 'MAE 5%'] = action_mae_list[4]
                    metric_groupby_single.loc[fold, 'MAE 10%'] = action_mae_list[9]
                    metric_groupby_single.loc[fold, 'MAE 20%'] = action_mae_list[19]
                    metric_groupby_single.loc[fold, 'MAE all'] = action_mae_list[-1]
                    metric_groupby_single.loc[fold, 'fold'] = fold
                    metric_groupby_single.loc[fold, 'method'] = method_MVI + '_' + method_ReprL_ + '_' + method_name_RL
                    q_test_df_class0_q_record = q_test_df_class0['q_record'].mean()
                    q_test_df_class1_q_record = q_test_df_class1['q_record'].mean()
                    mortality_ratio_q_record0 = q_test_df_class0['mortality_ratio_q_record'].mean()
                    mortality_ratio_q_record1 = q_test_df_class1['mortality_ratio_q_record'].mean()
                    q_test_df_class01_diff = q_test_df_class0_q_record - q_test_df_class1_q_record

                    metric_groupby_single.loc[fold, 'q record 0'] = q_test_df_class0_q_record
                    metric_groupby_single.loc[fold, 'q record 1'] = q_test_df_class1_q_record
                    metric_groupby_single.loc[fold, 'relapse_ratio_q_record 0'] = mortality_ratio_q_record0
                    metric_groupby_single.loc[fold, 'relapse_ratio_q_record 1'] = mortality_ratio_q_record1
                    metric_groupby_single.loc[fold, 'q record diff'] = q_test_df_class0_q_record - q_test_df_class1_q_record
                    metric_groupby_single.loc[fold, 'q r'] = metric_df_single['q'].mean() * q_test_df_class01_diff
                    metric_df_all_fold = pd.concat([metric_df_all_fold, pd.DataFrame(metric_df_single.mean(axis=0).values.reshape(1, -1),
                                                                                     columns=metric_df_single.columns)], axis=0)
                metric_groupby_all = pd.concat([metric_groupby_all, metric_groupby_single], axis=0)
                metric_df_all_fold['method'] = method_MVI + '_' + method_ReprL_ + '_' + method_name_RL + '_' + method_name_RL_
                metric_df_all_methods = pd.concat([metric_df_all_methods, metric_df_all_fold], axis=0)
metric_groupby_all.reset_index(drop=True, inplace=True)
metric_df_all_methods.reset_index(drop=True, inplace=True)
metric_df_all_methods = pd.concat([metric_df_all_methods, metric_groupby_all.drop(columns=['fold', 'method'])], axis=1)
metric_df_all_methods['relapse_ratio_nor_q'] = metric_df_all_methods['mortality_ratio_q'] / (metric_df_all_methods['relapse_ratio_q_record 1'] - metric_df_all_methods['relapse_ratio_q_record 0'])
metric_df_all_methods['method'].replace({'ours_ger_not_ours_ADRL_LC': 'MVIIL-GAN, MSLR, ADRL, RRC',
                                              'ours_ger_not_ours_ADRL_None': 'MVIIL-GAN, MSLR, ADRL, None',
                                              'ours_ger_not_ours_TD3-BC_LC': 'MVIIL-GAN, MSLR, TD3-BC, RRC',
                                              'ours_ger_not_ours_TD3-BC_None': 'MVIIL-GAN, MSLR, TD3-BC, None',
                                              'ours_ger_not_None_ADRL_LC': 'MVIIL-GAN, None, ADRL, RRC',
                                              'ours_ger_not_None_ADRL_None': 'MVIIL-GAN, None, ADRL, None',
                                              'ours_ger_not_None_TD3-BC_LC': 'MVIIL-GAN, None, TD3-BC, RRC',
                                              'ours_ger_not_None_TD3-BC_None': 'MVIIL-GAN, None, TD3-BC, None',
                                              's_ours_ADRL_LC': 'None, MSLR, ADRL, RRC',
                                              's_ours_ADRL_None': 'None, MSLR, ADRL, None',
                                              's_ours_TD3-BC_LC': 'None, MSLR, TD3-BC, RRC',
                                              's_ours_TD3-BC_None': 'None, MSLR, TD3-BC, None',
                                              's_None_ADRL_LC': 'None, None, ADRL, RRC',
                                              's_None_ADRL_None': 'None, None, ADRL, None',
                                              's_None_TD3-BC_LC': 'None, None, TD3-BC, RRC',
                                              's_None_TD3-BC_None': 'None, None, TD3-BC, None',
                                              }, inplace=True)
df_tmp = pd.DataFrame([], index=metric_df_all_methods.index)
df_tmp[['method 1', 'method 2', 'method 3', 'method 4']] = metric_df_all_methods['method'].str.split(', ', expand=True)
metric_df_all_methods = pd.concat([metric_df_all_methods, df_tmp], axis=1)
metric_df_all_methods['num_mod'] = df_tmp[['method 1', 'method 2', 'method 3', 'method 4']].replace({'None': 0, 'TD3-BC': 0, 'MVIIL-GAN': 1, 'MSLR': 1, 'ADRL': 1, 'RRC': 1}).sum(1)
name_method_list = ['MVIIL-GAN', 'MSLR', 'ADRL', 'RRC']
for count, index_, in enumerate(name_method_list):
    if count == 2:
        index_None = 'TD3-BC'
    else:
        index_None = 'None'

    df_1 = metric_df_all_methods[metric_df_all_methods[f'method {count + 1}'] == index_]
    df_2 = metric_df_all_methods[metric_df_all_methods[f'method {count + 1}'] == index_None]

    df_1['method_diff'] = ''
    df_2['method_diff'] = ''
    for i in range(len(name_method_list)):
        if i == 0:
            sep = ''
        else:
            sep = ', '
        if i == count:
            df_1['method_diff'] += sep + '-'
            df_2['method_diff'] += sep + '-'
        else:
            df_1['method_diff'] += sep + df_1[f'method {i + 1}']
            df_2['method_diff'] += sep + df_2[f'method {i + 1}']

    columns_num = metric_df_all_methods.filter(regex=r'^q|MAE|mortality|relapse').columns
    df_diff = df_1.reset_index(drop=True)[columns_num] - df_2.reset_index(drop=True)[columns_num]
    df_diff['method_diff'] = df_1.reset_index(drop=True)['method_diff']
    metric_save_diff = df_diff.groupby(by=['method_diff']).mean().round(3).applymap(
        lambda x: '{:.3f}'.format(x)).astype('str') \
                  + ' (' + df_diff.groupby(by=['method_diff']).std().round(3).applymap(
        lambda x: '{:.3f}'.format(x)).astype('str') + ')'
    # metric_save_diff.to_csv(f'metric_save_diff_{index_}_mean.csv', index_label='method')

    plt.figure(figsize=(8, 3))
    color_list = sns.color_palette('hls', 8)
    box_single = sns.boxplot(x='method_diff', y='relapse_ratio_nor_q', data=df_diff, palette=color_list)
    plt.ylabel(f'NERR', fontsize=10.5)
    plt.xlabel(f'', fontsize=10.5)
    plt.xticks(rotation=-25)
    # plt.show()
    # plt.savefig(f'boxplot_relapse_ratio_nor_q_diff_{index_}_0118.png', dpi=660, bbox_inches='tight', transparent=True)
    plt.close()


# metric_df_all_methods_group = metric_df_all_methods.groupby(by=['method']).mean()
metric_save = metric_df_all_methods.groupby(by=['method']).mean().round(3).applymap(lambda x: '{:.3f}'.format(x)).astype('str') \
              + ' (' + metric_df_all_methods.groupby(by=['method']).std().round(3).applymap(lambda x: '{:.3f}'.format(x)).astype('str') + ')'
# metric_save.to_csv('metric_save_mean.csv', index_label='method')

metric_df_all_methods_groupNumMod_mean = metric_df_all_methods.groupby(by=['num_mod']).mean()
metric_df_all_methods_groupNumMod_std = metric_df_all_methods.groupby(by=['num_mod']).std()
metric_byNumMod_save = metric_df_all_methods_groupNumMod_mean.round(3).applymap(lambda x: '{:.3f}'.format(x)).astype('str') \
              + ' (' + metric_df_all_methods_groupNumMod_std.round(3).applymap(lambda x: '{:.3f}'.format(x)).astype('str') + ')'
# metric_byNumMod_save.to_csv('metric_byNumMod.csv', index_label='method')

plt.figure(figsize=(8, 3))

# color_list = sns.color_palette('hls', 16)
# box_single = sns.boxplot(x='method', y='relapse_ratio_nor_q', data=metric_df_all_methods,
#                          palette=color_list)

color_list = sns.color_palette('hls', 5)
box_single = sns.boxplot(x='num_mod', y='relapse_ratio_nor_q', data=metric_df_all_methods, palette=color_list)

plt.ylabel(f'NERR', fontsize=10.5)

plt.xlabel(f'Number of modules', fontsize=10.5)

# plt.xlabel(f'', fontsize=10.5)
# plt.xticks(rotation=-25)

# sns.move_legend(box_single, "upper left", bbox_to_anchor=(0.5, 0.5))
# plt.savefig(f'boxplot_mortality_ratio_q.png', dpi=660, bbox_inches='tight', transparent=True)
# plt.savefig(f'boxplot_relapse_ratio_nor_q_NumMod_0118.png', dpi=660, bbox_inches='tight', transparent=True)
# plt.show()
plt.close()

color_list = sns.color_palette('hls', 16)
plt.figure(figsize=(16, 6))
sns.boxplot(x='method', y='relapse_ratio_nor_q', data=metric_df_all_methods, palette=color_list)
plt.ylabel(f'NEER', fontsize=10.5)
plt.xlabel(f'', fontsize=10.5)
plt.xticks(rotation=-25)
# plt.show()
# plt.savefig(f'boxplot_relapse_ratio_nor_q_all_0118.png', dpi=660, bbox_inches='tight', transparent=True)
plt.close()
pass
# EHRDM_metric_show(q_train_df_js_all_)
