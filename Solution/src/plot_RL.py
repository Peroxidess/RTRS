import os
import numpy as np
import pandas as pd
from model.evaluate import features_plot, plot_box, history_plot, q_plot, q_trend_plot, plot_pair, RL_history_plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def initPlot():
    sns.set(font_scale=1.3, style="white", palette="muted",
            color_codes=True, font='Times New Roman')  # set( )设置主题，调色板更常用
    # mpl.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题


# initPlot()

root = './plot RL'
# root = ''
# method_name_RL_list = ['KNN', 'Lasso', 'GBR', 'IQL', 'CQL', 'BCQ', 'BEAR', 'REM', 'TD3-BC', 'ADRL', 'ADRL_wo_AE', 'ADRL_wo_RP', 'ADRL_wo_AD']
# method_name_RL_list = ['ADRL', 'ADRL_wo_AE', 'ADRL_wo_RP', 'ADRL_wo_AD']
# method_name_RL_list = ['IQL', 'CQL', 'BCQ', 'BEAR', 'REM', 'TD3-BC', 'ADRL']
method_name_RL_list = ['ADRL']

metric_all = pd.DataFrame([])
for method_name_RL in method_name_RL_list:
    q_test_df_mortality = pd.read_csv(f'./q_table_test_mortality_ratio_{method_name_RL}_fold0.csv')
    q_mean = q_test_df_mortality['mortality_ratio_q'].mean()
    print(f'{method_name_RL} q_mean {q_mean}')
    q_zero_mean = q_test_df_mortality['mortality_ratio_q_zero'].mean()
    q_mean_mean = q_test_df_mortality['mortality_ratio_q_mean'].mean()
    q_random_mean = q_test_df_mortality['mortality_ratio_q_random'].mean()
    q_median = q_test_df_mortality['mortality_ratio_q'].median()
    q_zero_median = q_test_df_mortality['mortality_ratio_q_zero'].median()
    q_mean_median = q_test_df_mortality['mortality_ratio_q_mean'].median()
    q_random_median = q_test_df_mortality['mortality_ratio_q_random'].median()
    q_std = q_test_df_mortality[q_test_df_mortality['label1'] == 0]['mortality_ratio_q_record'].std()

    q_test_df_class0 = q_test_df_mortality[q_test_df_mortality['label1'] == 0]
    # q_test_df_mortality_sort = q_test_df_class0.sort_values(by='q_record', ascending=False)
    # q_test_df_class0_sort = q_test_df_class0.sort_values(by='mortality_ratio_q_record')
    q_test_df_class0_sort = q_test_df_class0.sort_values(by='q_record', ascending=False)
    action_mae_list = []
    for n in range(5, 600, 5):
        q_test_df_mortality_sort_ = q_test_df_class0_sort.iloc[:n]
        action_mae = abs(q_test_df_mortality_sort_[q_test_df_mortality_sort_['label1'] == 0]['action_diff']).mean()
        action_mae_list.append(action_mae)
    print(f'{method_name_RL} MAE {action_mae_list[0]}')
    metric = pd.DataFrame([], index=[method_name_RL])
    metric['MEAN'] = q_mean.round(3)
    metric['MEDIAN'] = q_median.round(3)
    metric['MAE'] = action_mae_list[0].round(3)
    metric_all = pd.concat([metric_all, metric], axis=0)
pass
# metric_all.to_csv('./performance_all.csv')


# fig_list = []
# for method_name_RL in method_name_RL_list:
#     q_test_df_box = pd.read_csv(root + f'./q_test_df_box_all_flod_{method_name_RL}.csv')
#     plot_box(q_test_df_box, method_name_RL) #single

fig, axes = plt.subplots(4, 2, figsize=(6, 12), squeeze=False) # grid subplots
axes = axes.flatten()
fig.delaxes(axes[-1])
legend_elements = []
color_list = sns.color_palette('hls', 6)
sns.set(font_scale=1)
for n, method_name_RL in enumerate(method_name_RL_list):
    q_test_df_box = pd.read_csv(root + f'./q_test_box_all_flod_{method_name_RL}.csv')
    q_test_df_box['index'].replace(regex=True, inplace=True, to_replace=r'_', value=r' ')
    plt.sca(axes[n])
    nor = MinMaxScaler()
    q_test_df_box['q values nor'] = nor.fit_transform(q_test_df_box['q values'].values.reshape(-1, 1))
    if n == len(method_name_RL_list) - 1:
        hue = 'Index'
    else:
        hue = None
    q_test_df_box['index'].replace({'q record 0': 'q Non-replase', 'q record 1': 'q Replase', 'q': 'q RL'}, inplace=True)
    q_test_df_box['index'] = q_test_df_box['index'].str.split(expand=True)[1]
    q_test_df_box['index'].replace({'mean': 'Mean', 'random': 'Random', 'record': 'Record'},
                                   inplace=True)
    q_test_df_box.rename(columns={'index': 'Index'}, inplace=True)
    box_single = sns.boxplot(x='Index', y='q values nor', data=q_test_df_box, hue=hue, width=0.8, dodge=False, palette=color_list)
    if n == len(method_name_RL_list) - 1:
        sns.move_legend(box_single, "upper left", bbox_to_anchor=(1.30, 1.0))
    title = f'{" ".join(method_name_RL.split("_"))}'
    plt.title(title, fontsize=9, y=0.98)
    if n % 2 == 0:
        plt.ylabel(f'q values', fontsize=9)
    else:
        plt.ylabel(f'', fontsize=9)
    # plt.xlabel(' ')
    plt.setp(axes, xticks=[], xlabel='')
    # ax.yaxis.set_label_coords(-0.12, 0.5)
    plt.xticks(rotation=-25)
# plt.tight_layout()
# plt.show()
# plt.close()
# plt.savefig(f'boxplot_comp_all_new.png', dpi=450, bbox_inches='tight', transparent=True)

# for method_name_RL in method_name_RL_list:
#     history_df = pd.read_csv(root + f'./history_df_{method_name_RL}.csv')
#     # history_plot([history_df.filter(regex=r'critic')], [method_name_RL], f'{method_name_RL}_critic')
#     RL_history_plot([history_df.filter(regex=r'critic')], [method_name_RL], f'{method_name_RL}_critic')
#     RL_history_plot([history_df.filter(regex=r'action')], [method_name_RL], f'{method_name_RL}_action')

# for method_name_RL in method_name_RL_list:
#     q_df = pd.read_csv(root + f'./q_test_all_flod_{method_name_RL}.csv')
#     q_plot(q_df, ['q_record'], name_method=method_name_RL, flag_save=False)

# metric_list = ['q']
# for method_name_RL in method_name_RL_list:
#     path_ = root + '/history_q' + f'/{method_name_RL}'
#     file_dir_list = os.listdir(path_)
#     history_df_metric = pd.DataFrame([])
#     for file_dir in file_dir_list:
#         history_df = pd.read_csv(path_ + '/' + file_dir)
#         for metric in metric_list:
#             history_df_metric = pd.concat([history_df_metric, history_df[metric]], axis=1)
#     history_plot([history_df_metric.filter(regex=rf'{metric}')], [metric], f'{method_name_RL}')
