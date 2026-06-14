import os
import sys

datasets = [
'agnewsdataraw-8000_trans_subst_10',
'searchsnippets_trans_subst_10',
'stackoverflow_trans_subst_10',  
'biomedical_trans_subst_10',
'TS_trans_subst_10',
'T_trans_subst_10',
'S_trans_subst_10',
'tweet-original-order_trans_subst_10'
]

seed = 0
for data in datasets:
    if data == 'agnewsdataraw-8000_trans_subst_10':
        num_classes = 4
        classes = 4
        pre_step = 600
        second_stage = 1000
        lambda2 = 5
    elif data == 'searchsnippets_trans_subst_10':
        num_classes = 8
        classes = 8
        pre_step = 600
        second_stage = 1000
        lambda2 = 1.25
    elif data == 'stackoverflow_trans_subst_10':
        num_classes = 20
        classes = 20
        pre_step = 600
        second_stage = 1000
        lambda2 = 5
    elif data == 'biomedical_trans_subst_10':
        num_classes = 20
        classes = 20
        pre_step = 600
        second_stage = 1000
        lambda2 = 5
    elif data == 'TS_trans_subst_10':
        num_classes = 152
        classes = 152
        pre_step = 600
        second_stage = 1000
        lambda2 = 1.25
    elif data == 'T_trans_subst_10':
        num_classes = 152
        classes = 152
        pre_step = 600
        second_stage = 1000
        lambda2 = 1.25
    elif data == 'S_trans_subst_10':
        num_classes = 152
        classes = 152
        pre_step = 600
        second_stage = 1000
        lambda2 = 1.25
    elif data == 'tweet-original-order_trans_subst_10':
        num_classes = 89
        classes = 89
        pre_step = 600
        second_stage = 1000
        lambda2 = 1.25
    else:
        sys.stderr.write("wrong dataset name")
        sys.exit(1)

    sys.stderr.write("\n**********************************************\n")
    sys.stderr.write("dataset: {}".format(data))
    sys.stderr.write("\n**********************************************\n")
    sys.stderr.flush()

    os.system(
        'CUDA_VISIBLE_DEVICES=0  python main.py --dataname {} --num_classes {} --classes {} --pre_step {} --lambda2 {} --second_stage {} --seed {}'.format(
            data, num_classes, classes, pre_step, lambda2, second_stage, seed))


