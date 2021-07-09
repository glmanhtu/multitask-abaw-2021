import argparse
import csv
import os
import pickle

from tqdm import tqdm

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--annot_dir', type=str, default = '/home/mvu/Documents/datasets/affwild-2/annotations',
                    help='annotation dir')
parser.add_argument('--annotations', type=str, default='/home/mvu/Documents/datasets/mixed/affwild-2/annotations.pkl')

args = parser.parse_args()

total_data = pickle.load(open(args.annotations, 'rb'))
# expr_data = total_data['EXPR_Set']['Training_Set']

def validate_va(va_data, annotation_dir):
    for video in tqdm(va_data, desc='Validating VA Set'):
        label_file = os.path.join(annotation_dir, f'{video}.txt')
        csv_file = csv.DictReader(open(label_file))
        records = [r for r in csv_file]

        for index, record in va_data[video].iterrows():
            path = record['path']
            frame_id = int(path.split('/')[-1].split('.')[0])
            valence = record['valence']
            arousal = record['arousal']
            gt_record = records[frame_id]
            assert valence == float(gt_record['valence'])
            assert arousal == float(gt_record['arousal'])


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


validate_va(total_data['VA_Set']['Training_Set'], os.path.join(args.annot_dir, 'VA_Set', 'Training_Set'))
validate_va(total_data['VA_Set']['Validation_Set'], os.path.join(args.annot_dir, 'VA_Set', 'Validation_Set'))
validate_expr(total_data['EXPR_Set']['Training_Set'], os.path.join(args.annot_dir, 'EXPR_Set', 'Training_Set'))
validate_expr(total_data['EXPR_Set']['Validation_Set'], os.path.join(args.annot_dir, 'EXPR_Set', 'Validation_Set'))
