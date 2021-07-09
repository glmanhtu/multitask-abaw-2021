import argparse
import pickle
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='read two annotations files')
parser.add_argument('--aff_wild2_pkl', type=str, default='/home/mvu/Documents/datasets/mixed/affwild-2/annotations.pkl')
parser.add_argument('--VA_pkl', type=str, default='/home/mvu/Documents/datasets/mixed/afew-va/annotations.pkl')
parser.add_argument('--save_path', type=str, default='/home/mvu/Documents/datasets/mixed/mixed_VA_annotations.pkl')
# parser.add_argument('--disable_test_set', action='store_true', help='Disable using test set')
args = parser.parse_args()
VA_list = ['valence', 'arousal']

# Videos on both training and validating set
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

# test_set = ['video70', '16-30-1920x1080', 'video96', '234', '200', '52-30-1280x720_right', '291', '126-30-1080x1920',
#             '238', 'video56', 'video74_left', '320', 'video88', 'video91', '317']
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
    train_data = total_data['VA_Set']['Train_Set']
    expr_data = total_data['EXPR_Set']['Train_Set']
    au_data = total_data['AU_Set']['Train_Set']
    paths = []
    va_labels = []
    expr_labels = []
    au_list = []
    for video in train_data.keys():
        # if video in test_set:
        #     continue
        data = train_data[video]
        if video in expr_data.keys():
            data = pd.merge(data, expr_data[video], how='left', on=['path', 'frames_ids']).fillna(value=-2)
            data = filtering(data)
        else:
            data['label'] = -2

        if video in au_data.keys():
            data = pd.merge(data, au_data[video], how='left', on=['path', 'frames_ids']).fillna(value=-2)
            data = filtering(data)
        else:
            for key in AU_list:
                data[key] = -2

        va_labels.append(np.stack([data['valence'], data['arousal']], axis=1))
        expr_labels.append(data['label'].values.astype(np.float32))
        au_list.append(data[AU_list].values.astype(np.float32))
        paths.append(data['path'].values)
    paths = np.concatenate(paths, axis=0)
    va_labels = np.concatenate(va_labels, axis=0)
    expr_labels = np.concatenate(expr_labels, axis=0)
    au_list = np.concatenate(au_list, axis=0)

    origin = np.array(['affwild2' for _ in range(len(paths))])
    train_out_data = {'label': va_labels, 'path': paths, 'expr': expr_labels, 'origin': origin, 'au': au_list}
    # validation set
    val_data = total_data['VA_Set']['Validation_Set']
    paths = []
    va_labels = []
    for video in val_data.keys():
        # if video in share_vids:
        #     continue
        # if video in test_set:
        #     continue
        data = val_data[video]
        va_labels.append(np.stack([data['valence'], data['arousal']], axis=1))
        paths.append(data['path'].values)
    paths = np.concatenate(paths, axis=0)
    va_labels = np.concatenate(va_labels, axis=0)

    origin = np.array(['affwild2' for _ in range(len(paths))])
    val_out_data = {'label': va_labels, 'path': paths, 'origin': origin}

    # paths = []
    # va_labels = []
    # for video in test_set:
    #     if video in val_data:
    #         data = val_data[video]
    #     else:
    #         data = train_data[video]
    #
    #     va_labels.append(np.stack([data['valence'], data['arousal']], axis=1))
    #     paths.append(data['path'].values)
    #
    # if len(paths) > 0:
    #     paths = np.concatenate(paths, axis=0)
    #     va_labels = np.concatenate(va_labels, axis=0)
    #
    # origin = np.array(['affwild2' for _ in range(len(paths))])
    # test_out_data = {'label': va_labels, 'path': paths, 'origin': origin}

    return train_out_data, val_out_data


def merge_two_datasets():
    data_aff_wild2, data_aff_wild2_val = read_aff_wild2()
    # downsample x 5 the training set in aff_wild training set
    aff_wild_train_labels = data_aff_wild2['label']
    aff_wild_train_paths = data_aff_wild2['path']
    length = len(aff_wild_train_labels)
    index = [True if i % 5 == 0 else False for i in range(length)]
    aff_wild_train_labels = aff_wild_train_labels[index]
    aff_wild_train_paths = aff_wild_train_paths[index]
    aff_wild_train_expr = data_aff_wild2['expr'][index]
    aff_wild_train_au = data_aff_wild2['au'][index]
    origin = data_aff_wild2['origin'][index]
    data_aff_wild2 = {'label': aff_wild_train_labels, 'path': aff_wild_train_paths,
                      'expr': aff_wild_train_expr, 'origin': origin, 'au': aff_wild_train_au}
    # downsample x 5 the training set in aff_wild
    data_VA = pickle.load(open(args.VA_pkl, 'rb'))
    data_VA = {**data_VA['Training_Set'], **data_VA['Validation_Set']}
    labels = []
    paths = []
    for video in data_VA.keys():
        data = data_VA[video]
        labels.append(np.stack([data['valence'], data['arousal']], axis=1))
        paths.append(data['path'])
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    data_VA = {'label': labels, 'path': paths, 'expr': np.full((len(paths)), -2), 'origin': np.full((len(paths)), 'VA'),
               'au': np.full((len(paths), len(AU_list)), -2)}
    data_merged = {'label': np.concatenate((data_aff_wild2['label'], data_VA['label']), axis=0),
                   'expr': np.concatenate((data_aff_wild2['expr'], data_VA['expr']), axis=0),
                   'au': np.concatenate((data_aff_wild2['au'], data_VA['au']), axis=0),
                   'path': list(data_aff_wild2['path']) + list(data_VA['path']),
                   'origin': list(data_aff_wild2['origin']) + list(data_VA['origin'])}
    print("Aff-wild2 :{}".format(len(data_aff_wild2['label'])))
    print("AFEW_VA:{}".format(len(data_VA['label'])))
    return {'Train_Set': data_merged, 'Validation_Set': data_aff_wild2_val}


def plot_distribution(data):
    all_samples = data['label']
    plt.hist2d(all_samples[:, 0], all_samples[:, 1], bins=(20, 20), cmap=plt.cm.jet)
    plt.xlabel("Valence")
    plt.ylabel('Arousal')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    data_file = merge_two_datasets()
    pickle.dump(data_file, open(args.save_path, 'wb'))
    plot_distribution(data_file['Train_Set'])
