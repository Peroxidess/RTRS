import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='liver_processed_class', # Classification task needs to add "_class"
                        help='{PPH_vag/ces_reg}, {wine}, {steam} {mimic_preprocessed_class} {mimic_ppc_class} {simulate} {thyroid} {examination} {liver_processed}')
    parser.add_argument('--nrun', type=int, default=1,
                        help='total number of runs[default: 1]')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='cross-validation fold, 1 refer not CV [default: 1]')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='proportion of test sets divided from training set, '
                             '0 refer dataset has its own test set [default: 0.2]')
    parser.add_argument('--val_ratio', type=float, default=0.,
                        help='proportion of test sets divided from training set [default: 0.2]')
    parser.add_argument('--method_MVI', type=str, default='ours_ger_not',
                        help='missing values imputation method [default: "mice"]')
    parser.add_argument('--dict_ManualRatio', type=dict, default={'drop_NanRatio': 1, 'a': 0.5, 'b': 0.5, 'c': 0.5},
                        help='ratio of manual missing values [default: ""]')
    parser.add_argument('--missing_ratio', type=float, default=0.,
                        help='ratio of manual missing values [default: ""]')
    parser.add_argument('--method_ReprL_', type=str, default='ours_noise_none',
                        help='Representation Learning Method[default: "ours"]')
    parser.add_argument('--method_name_RL', type=str, default='ADRL_LC',
                        help='Reinforcement Learning Method[default: "ADRL"]')
    parser.add_argument('--Flag_LoadMetric', type=bool, default=False, metavar='N',
                        help='overload metric training before[default: False]')
    parser.add_argument('--Flag_DataPreprocessing', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_MVI', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Deal_Imbalance_Method', type=str, default='False', metavar='N',
                        help='[default: False SMOTE BorderlineSMOTE ADASYN KMeansSMOTE RandomOverSampler OneSidedSelection RandomUnderSampler ClusterCentroids SMOTETomek SMOTEENN]')
    parser.add_argument('--method_AL', type=str, default='random', metavar='N',
                        help='[default: None]')
    parser.add_argument('--Flag_downstream', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_Mask_Saving', type=bool, default=False, metavar='N',
                        help='[default: True]')
    parser.add_argument('--test', type=int, default=0, metavar='N',
                        help='[default: 0]')
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    # parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    # parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    args = parser.parse_args()

    # if not os.path.exists(args.out_path):
    #     os.mkdir(args.out_path)

    return args
