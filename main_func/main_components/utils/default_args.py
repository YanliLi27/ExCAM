import argparse


def default_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument('--batch_size', type=int, default=16, help='Dataloader batch size')
    parser.add_argument('--groups', type=int, default=1, help='If use group convolution')

    # for model
    parser.add_argument('--randomization', type=bool, default=False, help='If start randomization')
    parser.add_argument('--random_severity', type=int, default=0, choices=[0, 1, 2, 3, 4], help='n/4 randomization')

    # for file management and loop
    parser.add_argument('--fold_order', type=int, default=0, help='For cross validation')

    # cam required args
    parser.add_argument('--target_category_flag', type=int, default=None, help='Output category')
    parser.add_argument('--confidence_weight_flag', type=bool, default=False, help='Output category')

    # optional function -- core improvements
    parser.add_argument('--use_stat', type=bool, default=False, help='For using analyzer and im')
    # for predictor
    parser.add_argument('--stat_maxmin_flag', type=bool, default=False, help='For maxmin normalization')
    parser.add_argument('--remove_minus_flag', type=bool, default=True, help='If remove the part of heatmaps less than 0')
    parser.add_argument('--im_selection_mode', type=str, default='diff_top', choices=['max', 'top', 'diff_top', 'freq', 'index', 'all'],
                        help='If use statistic')
    parser.add_argument('--im_selection_extra', type=float, default=0.05, help='attributes of selection mode')
    parser.add_argument('--max_iter', default=None, help='max iteration to save time')

    args = parser.parse_args()
    return args