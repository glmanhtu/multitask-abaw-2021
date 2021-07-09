import argparse
import math
import pickle
import numpy as np
import pandas as pd
import afew_va.create_annotation_files_Mixed_VA as mixed_va
import expw.create_annotation_files_Mixed_EXPR as mixed_expr
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='read two annotations files')
parser.add_argument('--aff_wild2_pkl', type=str, default='/home/mvu/Documents/datasets/mixed/affwild-2/annotations.pkl')
parser.add_argument('--affect_net_pkl', type=str,
                    default='/home/mvu/Documents/datasets/mixed/affectnet/annotations.pkl')
parser.add_argument('--save_path', type=str,
                    default='/home/mvu/Documents/datasets/mixed/affwild2_EXPR_VA_annotations.pkl')
test_set = set(mixed_va.test_set + mixed_expr.test_set)
args = parser.parse_args()
Expr_list = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
expr_mapping = [0, 4, 5, 6, 3, 2, 1]  # Map from affectnet format to Affwild2 format


def filtering(paths, exprs, vas):
    # From https://arxiv.org/pdf/2002.03399.pdf

    # to_del = []
    # for index, row in enumerate(paths):
    #     # The expression is labeled as happy, but the valence is labeled as negative
    #     if exprs[index] == 4 and vas[index][0] != -2 and vas[index][0] < 0:
    #         to_del.append(index)
    #
    #     # The expression is labeled as sad, but the valence is labeled as positive
    #     elif exprs[index] == 5 and vas[index][0] > 0:
    #         to_del.append(index)
    #
    #     # The expression is labeled as neutral, but sqrt(valence^2 + arousal^2) > 0.5
    #     elif exprs[index] == 0:
    #         if math.sqrt(vas[index][0] ** 2 + vas[index][1] ** 2) > 0.5:
    #             to_del.append(index)
    # paths = np.delete(paths, to_del, axis=0)
    # vas = np.delete(vas, to_del, axis=0)
    # exprs = np.delete(exprs, to_del, axis=0)
    # print(f'Filtered {len(to_del)} records')
    return paths, exprs, vas


def read_aff_wild2():
    total_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
    expr_data = total_data['EXPR_Set']['Training_Set']
    va_data = total_data['VA_Set']['Training_Set']
    expr_va_keys = list(set(expr_data.keys()) & set(va_data.keys()))
    paths, expr, va = [], [], []
    for key in expr_va_keys:
        if key in test_set:
            continue
        data_merge = pd.merge(expr_data[key], va_data[key], on='path')
        paths.append(data_merge['path'].values)
        expr.append(data_merge['label'].values.astype(np.float32))
        va.append(np.stack([data_merge['valence'], data_merge['arousal']], axis=1))
        assert len(paths[-1]) == len(expr[-1]) == len(va[-1])
    paths = np.concatenate(paths, axis=0)
    expr = np.concatenate(expr, axis=0)
    va = np.concatenate(va, axis=0)

    length = len(paths)
    index = [True if i % 3 == 0 else False for i in range(length)]
    paths = paths[index]
    expr = expr[index]
    va = va[index]

    return {
        'path': paths,
        'EXPR': expr,
        'VA': va
    }


def read_affect_net():
    all_data = pickle.load(open(args.affect_net_pkl, 'rb'))

    return {
        'path': np.asarray(all_data['path']),
        'EXPR': np.asarray([expr_mapping[x] for x in all_data['EXPR']]),
        'VA': np.array([[x['valence'], x['arousal']] for x in all_data['AV']])
    }


def calculate_weight(va_labels, expr_labels):
    # VA weights
    N, C = va_labels.shape
    assert C == 2
    hist, x_edges, y_edges = np.histogram2d(va_labels[:, 0], va_labels[:, 1], bins=[20, 20])
    x_bin_id = np.digitize(va_labels[:, 0], bins=x_edges) - 1
    y_bin_id = np.digitize(va_labels[:, 1], bins=y_edges) - 1
    # for value beyond the edges, the function returns len(digitize_num), but it needs to be replaced by len(edges)-1
    x_bin_id[x_bin_id == 20] = 20 - 1
    y_bin_id[y_bin_id == 20] = 20 - 1
    va_weights = []
    for x, y in zip(x_bin_id, y_bin_id):
        assert hist[x, y] != 0
        va_weights += [1 / hist[x, y]]

    mapping = {}
    for label in expr_labels:
        if label not in mapping:
            mapping[label] = 0
        mapping[label] += 1

    expr_weights = [1.0 / mapping[x] for x in expr_labels]

    weights = np.array(va_weights) + np.array(expr_weights)
    return weights / np.max(weights)


def merge_two_datasets():
    # data_affect = read_affect_net()
    data_aff_wild2 = read_aff_wild2()
    # paths = np.concatenate([data_affect['path'], data_aff_wild2['path']], axis=0)
    # exprs = np.concatenate([data_affect['EXPR'], data_aff_wild2['EXPR']], axis=0)
    # vas = np.concatenate([data_affect['VA'], data_aff_wild2['VA']], axis=0)
    # paths = data_affect['path']
    # exprs = data_affect['EXPR']
    # vas = data_affect['VA']
    paths = data_aff_wild2['path']
    exprs = data_aff_wild2['EXPR']
    vas = data_aff_wild2['VA']

    # origin_data_size = len(paths)
    #
    # while len(paths) / float(origin_data_size) > 0.7:
    #     weights = calculate_weight(vas, exprs)
    #     sort_idx_weights = np.argsort(weights)
    #     n_to_del = int(0.01 * len(paths))
    #     idx_to_del = sort_idx_weights[:n_to_del]
    #     paths = np.delete(paths, idx_to_del, axis=0)
    #     vas = np.delete(vas, idx_to_del, axis=0)
    #     exprs = np.delete(exprs, idx_to_del, axis=0)

    paths, exprs, vas = filtering(paths, exprs, vas)

    plot_distribution_va(vas)
    plot_distribution_expr(exprs)

    aff_wild2 = []
    for path in paths:
        if 'affwild' in path:
            aff_wild2.append(path)
    percent_affwil2 = len(aff_wild2) / float(len(paths))
    print(f'Percent of AffWild 2: {percent_affwil2}')
    print(f'Percent of AffectNet: {1 - percent_affwil2}')

    return {
        'path': paths,
        'VA': vas,
        'EXPR': exprs
    }


def plot_distribution_va(all_samples):
    plt.hist2d(all_samples[:, 0], all_samples[:, 1], bins=(20, 20), cmap=plt.cm.jet)
    plt.xlabel("Valence")
    plt.ylabel('Arousal')
    plt.colorbar()
    plt.show()


def plot_distribution_expr(all_samples):
    histogram = np.zeros(len(Expr_list))
    for i in range(len(Expr_list)):
        find_true = sum(all_samples == i)
        histogram[i] = find_true / all_samples.shape[0]
    plt.bar(np.arange(len(Expr_list)), histogram)
    plt.xticks(np.arange(len(Expr_list)), Expr_list)
    plt.show()


if __name__ == '__main__':
    data = merge_two_datasets()
    print('Total records: %d' % len(data['path']))
    pickle.dump(data, open(args.save_path, 'wb'))
