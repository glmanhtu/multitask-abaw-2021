import argparse
import pickle
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='read two annotations files')
parser.add_argument('--aff_wild2_pkl', type=str, default='/home/mvu/Documents/datasets/mixed/affwild-2/annotations.pkl')
parser.add_argument('--ExpW_pkl', type=str, default='/home/mvu/Documents/datasets/mixed/expw/annotations.pkl')
parser.add_argument('--save_path', type=str, default='/home/mvu/Documents/datasets/mixed/mixed_EXPR_annotations.pkl')
parser.add_argument('--disable_test_set', action='store_true', help='Disable using test set')
args = parser.parse_args()
Expr_list = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
share_vids = {'134', '13-30-1920x1080', '128', '127', '137', '157', '45-24-1280x720', '136', '158', '163',
              '125-25-1280x720', '24-30-1920x1080-2', '162', '76-30-640x280', '144', '129', '27-60-1280x720',
              'video63', 'video67', '131', '122-60-1920x1080-2', '161', '135-24-1920x1080_left', '122', '53-30-360x480',
              '146', '135', '153', '8-30-1280x720', '150', '155', '133', '125', '121', '140', '48-30-720x1280',
              '140-30-632x360', '114-30-1280x720', '122-60-1920x1080-3', '151', '119-30-848x480', '120-30-1280x720',
              'video66', 'video94', '154', '77-30-1280x720', '72-30-1280x720', '82-25-854x480', '12-24-1920x1080',
              '121-24-1920x1080', '129-24-1280x720', '1-30-1280x720', '123', '28-30-1280x720-2', '282', '117', '143',
              '21-24-1920x1080', '118', '156', '132-30-426x240', '132', '84-30-1920x1080', 'video58',
              '123-25-1920x1080', '25-25-600x480', '160', '107-30-640x480', '26-60-1280x720', '124-30-720x1280',
              '113-60-1280x720', 'video61', '131-30-1920x1080', '122-60-1920x1080-4', '138', '112-30-640x360', '148',
              '149', '141', 'video79', '126', '81-30-576x360', 'video93', '120', 'video1', '115-30-1280x720'}

AU_list = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
# test_set = ['video47', '137-30-1920x1080', '56-30-1080x1920', 'video73', '128-24-1920x1080', '165', '139', '159']
# if args.disable_test_set:
#     test_set = []


def filtering(data):
    # From https://arxiv.org/pdf/2002.03399.pdf

    # to_del = []
    # for index, row in data.iterrows():
    # 	# The expression is labeled as happy, but the valence is labeled as negative
    # 	if row['label'] == 4 and row['valence'] != -2 and row['valence'] < 0:
    # 		to_del.append(index)
    #
    # 	# The expression is labeled as sad, but the valence is labeled as positive
    # 	elif row['label'] == 5 and row['valence'] > 0:
    # 		to_del.append(index)
    #
    # 	# The expression is labeled as neutral, but sqrt(valence^2 + arousal^2) > 0.5
    # 	elif row['label'] == 0:
    # 		if math.sqrt(row['valence'] ** 2 + row['arousal'] ** 2) > 0.5:
    # 			to_del.append(index)
    # filtered_data = data.drop(data.index[to_del])
    # print(f'Filtered {len(data) - len(filtered_data)} records')
    # return filtered_data
    return data


def read_aff_wild2():
    total_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
    # training set
    train_data = total_data['EXPR_Set']['Train_Set']
    va_data = total_data['VA_Set']['Train_Set']
    au_data = total_data['AU_Set']['Train_Set']

    paths = []
    labels = []
    va_labels = []
    au_labels = []
    for video in train_data.keys():
        df = train_data[video]
        # if video in test_set:
        #     continue
        if video in va_data.keys():
            df = pd.merge(df, va_data[video], how='left', on=['path', 'frames_ids']).fillna(value=-2)
            df = filtering(df)
        else:
            df['valence'] = -2
            df['arousal'] = -2
        if video in au_data.keys():
            df = pd.merge(df, au_data[video], how='left', on=['path', 'frames_ids']).fillna(value=-2)
            df = filtering(df)
        else:
            for key in AU_list:
                df[key] = -2
        labels.append(df['label'].values.astype(np.float32))
        va_labels.append(np.stack([df['valence'], df['arousal']], axis=1))
        paths.append(df['path'].values)
        au_labels.append(df[AU_list].values.astype(np.float32))
        assert len(paths) == len(labels) == len(va_labels)

    # undersample the neutral samples by 10
    # undersample the happy and sad samples by 20
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    va_labels = np.concatenate(va_labels, axis=0)
    au_labels = np.concatenate(au_labels, axis=0)
    # neutral
    keep_10 = np.array([True if i % 10 == 0 else False for i in range(len(labels))])
    to_drop = labels == 0
    to_drop = to_drop * (~keep_10)
    labels = labels[~to_drop]
    paths = paths[~to_drop]
    va_labels = va_labels[~to_drop]
    au_labels = au_labels[~to_drop]
    # happy
    keep_2 = np.array([True if i % 2 == 0 else False for i in range(len(labels))])
    to_drop = labels == 4
    to_drop = to_drop * (~keep_2)
    labels = labels[~to_drop]
    paths = paths[~to_drop]
    va_labels = va_labels[~to_drop]
    au_labels = au_labels[~to_drop]
    # sadness
    keep_2 = np.array([True if i % 2 == 0 else False for i in range(len(labels))])
    to_drop = labels == 5
    to_drop = to_drop * (~keep_2)
    labels = labels[~to_drop]
    paths = paths[~to_drop]
    va_labels = va_labels[~to_drop]
    au_labels = au_labels[~to_drop]
    origin = np.array(['affwild2' for _ in range(len(paths))])

    data = {'label': labels, 'path': paths, 'va': va_labels, 'origin': origin, 'au': au_labels}
    # validation set
    val_data = total_data['EXPR_Set']['Validation_Set']
    paths = []
    labels = []
    for video in val_data.keys():
        # if video in share_vids:
        # 	continue
        # if video in test_set:
        #     continue
        df = val_data[video]
        labels.append(df['label'].values.astype(np.float32))
        paths.append(df['path'].values)
    # undersample the neutral samples by 10
    # undersample the happy and sad samples by 20
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)

    origin = np.array(['affwild2' for _ in range(len(paths))])
    val_out_data = {'label': labels, 'path': paths, 'origin': origin}

    # paths = []
    # labels = []
    # for video in test_set:
    #     if video in train_data.keys():
    #         df = train_data[video]
    #     else:
    #         df = val_data[video]
    #     labels.append(df['label'].values.astype(np.float32))
    #     paths.append(df['path'].values)
    #     # undersample the neutral samples by 10
    #     # undersample the happy and sad samples by 20
    # if len(paths) > 0:
    #     paths = np.concatenate(paths, axis=0)
    #     labels = np.concatenate(labels, axis=0)
    #
    # origin = np.array(['affwild2' for _ in range(len(paths))])
    # test_data = {'label': labels, 'path': paths, 'origin': origin}
    return data, val_out_data


def merge_two_datasets():
    data_aff_wild2, data_aff_wild2_val = read_aff_wild2()
    data_ExpW = pickle.load(open(args.ExpW_pkl, 'rb'))
    # change the label integer, because of the different labelling in two datasets
    ExpW_to_aff_wild2 = [1, 2, 3, 4, 5, 6, 0]
    data_ExpW['label'] = np.array([ExpW_to_aff_wild2[x] for x in data_ExpW['label']])
    data_ExpW['va'] = np.array([[-2., -2.] for _ in range(len(data_ExpW['path']))])
    data_ExpW['origin'] = np.array(['expw' for _ in range(len(data_ExpW['path']))])
    data_ExpW['au'] = np.full((len(data_ExpW['path']), len(AU_list)), -2)
    data_merged = {'label': np.concatenate((data_aff_wild2['label'], data_ExpW['label']), axis=0),
                   'va': np.concatenate((data_aff_wild2['va'], data_ExpW['va']), axis=0),
                   'au': np.concatenate((data_aff_wild2['au'], data_ExpW['au']), axis=0),
                   'path': list(data_aff_wild2['path']) + data_ExpW['path'],
                   'origin': list(data_aff_wild2['origin']) + list(data_ExpW['origin'])}
    print("Dataset\t" + "\t".join(Expr_list))
    print("Aff_wild2 dataset:\t" + "\t".join([str(sum(data_aff_wild2['label'] == i)) for i in range(len(Expr_list))]))
    print("ExpW dataset:\t" + "\t".join([str(sum(data_ExpW['label'] == i)) for i in range(len(Expr_list))]))
    return {'Train_Set': data_merged, 'Validation_Set': data_aff_wild2_val}


def plot_distribution(data):
    all_samples = data['label']
    histogram = np.zeros(len(Expr_list))
    for i in range(len(Expr_list)):
        find_true = sum(all_samples == i)
        histogram[i] = find_true
    plt.bar(np.arange(len(Expr_list)), histogram)
    plt.xticks(np.arange(len(Expr_list)), Expr_list)
    plt.show()


if __name__ == '__main__':
    # data_file = read_all_image()
    data_file = merge_two_datasets()
    pickle.dump(data_file, open(args.save_path, 'wb'))
    plot_distribution(data_file['Train_Set'])
