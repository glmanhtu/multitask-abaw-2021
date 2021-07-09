import argparse
import csv
import glob
import json
import os
import pickle
import re

from tqdm import tqdm

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--annot_dir', type=str, default = '/home/mvu/Documents/datasets/affwild-2/annotations',
                    help='annotation dir')

parser.add_argument('--tnt_dir', type=str, default='/home/mvu/Documents/datasets/affwild-2/aff2_processed')
parser.add_argument('--annotations', type=str, default='/home/mvu/Documents/datasets/mixed/mixed_VA_annotations.pkl')

args = parser.parse_args()

total_data = pickle.load(open(args.annotations, 'rb'))
# expr_data = total_data['EXPR_Set']['Training_Set']

def load_keymap():
    results = {}
    annotation_files = glob.glob(os.path.join(args.tnt_dir, '*.json'))
    for file in annotation_files:
        with open(file) as f:
            meta = json.load(f)
        original_file_name = os.path.splitext(meta['original_video'])[0]
        current_file_name = os.path.basename(file)
        position_search = re.search(r'(\d+_AU\d[t|v|_]_EX\d[t|v|_]_VA\d[t|v|_])(_\w+)?\.\w+\.json', current_file_name)
        key = original_file_name
        if position_search.group(2) is not None and position_search.group(2) != '_main':
            key = key + position_search.group(2)
        name = current_file_name.split('.')[0]
        if name not in results:
            results[name] = key
        else:
            raise Exception('Double key found: ' + key)
    return results

def load_annotations():
    keymap = load_keymap()
    result = {'Training_Set': {}, 'Validation_Set': {}}
    for mode in result.keys():
        annotation_dir = os.path.join(args.annot_dir, 'VA_Set', mode)
        for key in keymap.keys():
            label_file = os.path.join(annotation_dir, f'{keymap[key]}.txt')
            if not os.path.isfile(label_file):
                continue
            csv_file = csv.DictReader(open(label_file))
            records = [r for r in csv_file]
            result[mode][key] = records
    return result


def validate_va(va_data, all_labels):
    for path, va in tqdm(zip(va_data['path'], va_data['label']), desc='Validating VA Set'):
        if 'affwild-2' not in path:
            continue
        gt_labels = all_labels[path.split('/')[-2]]
        frame_id = int(path.split('/')[-1].split('.')[0])
        gt_label = gt_labels[frame_id]
        assert float(gt_label['valence']) == va[0]
        assert float(gt_label['arousal']) == va[1]


def validate_expr(va_data, annotation_dir):
    for video in tqdm(va_data, desc='Validating EXPR Set'):
        label_file = os.path.join(annotation_dir, f'{video}.txt')
        csv_file = csv.DictReader(open(label_file))
        records = [r for r in csv_file]

        for index, record in va_data[video].iterrows():
            path = record['path']
            frame_id = int(path.split('/')[-1].split('.')[0])
            label = record['label']
            gt_record = records[frame_id]
            assert label == int(gt_record['Neutral'])


labels = load_annotations()
validate_va(total_data['Training_Set'], labels['Training_Set'])
validate_va(total_data['Validation_Set'], labels['Validation_Set'])
