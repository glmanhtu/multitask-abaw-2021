import argparse
import os
import pickle
import subprocess
import tempfile

import numpy as np
import pandas as pd
from sox import Transformer

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action='store_true',
                    help='whether to visualize the distribution')
parser.add_argument('--annot_dir', type=str, default='/home/mvu/Documents/datasets/affwild-2/annotations',
                    help='annotation dir')
parser.add_argument('--data_dir', type=str, default='/home/mvu/Documents/datasets/affwild-2/cropped_aligned')
parser.add_argument('--audio_dir', type=str, default='/home/mvu/Documents/datasets/affwild-2/audios')
parser.add_argument('--origin_videos_dir', type=str, default='/home/mvu/Documents/datasets/affwild-2/videos')
parser.add_argument('--save_path', default='/home/mvu/Documents/datasets/mixed/affwild-2/annotations_wshared.pkl')
parser.add_argument('--aff_wild2_pkl', type=str, default='/home/mvu/Documents/datasets/mixed/affwild-2/annotations.pkl')
parser.add_argument('--disable_test_set', action='store_true', help='Disable using test set')


args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


def set_metadata(meta_data, video, key, value):
    if video not in meta_data:
        meta_data[video] = {}
    if key not in meta_data[video]:
        meta_data[video][key] = None
    if meta_data[video][key] is not None:
        assert meta_data[video][key] == value
    else:
        meta_data[video][key] = value


def main():
    affwild2_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
    data_file = {'Metadata': {}, 'VideoTimestamps': {}}
    for video in os.listdir(args.origin_videos_dir):
        video_file = os.path.join(args.origin_videos_dir, video)
        video_ts_file = os.path.join(tempfile.mkdtemp(), 'ts.txt')
        mkvfile = os.path.join(tempfile.mkdtemp(), 'ts.mkv')
        command = 'mkvmerge -o ' + mkvfile + ' ' + video_file
        subprocess.call(command, shell=True)
        command = 'mkvextract ' + mkvfile + ' timestamps_v2 0:' + video_ts_file
        subprocess.call(command, shell=True)
        with open(video_ts_file, 'r') as f:
            time_stamps = np.genfromtxt(f)
        data_file['VideoTimestamps'][os.path.splitext(video)[0]] = time_stamps

        # Extract audio

        name = os.path.basename(video).split('.')[0]
        audio_file = os.path.join(args.audio_dir, f'{name}.wav')
        command = 'ffmpeg -i ' + video_file + ' -ar 44100 -ac 1 -y ' + audio_file
        subprocess.call(command, shell=True)
        assert os.path.exists(audio_file)
        set_metadata(data_file['Metadata'], name, 'audio_file', audio_file)

        os.remove(mkvfile)
        os.remove(video_ts_file)

    # Collect cross tasks annotations
    AU_list = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    data_file = {**data_file, 'VA_Set': {}, 'EXPR_Set': {}, 'AU_Set': {}}
    for mode in ['Train_Set', 'Validation_Set']:
        va_videos = set(affwild2_data['VA_Set'][mode].keys())
        expr_videos = set(affwild2_data['EXPR_Set'][mode].keys())
        au_videos = set(affwild2_data['AU_Set'][mode].keys())
        expr_va_shared_videos = va_videos.intersection(expr_videos)
        au_va_shared_videos = va_videos.intersection(au_videos)
        au_expr_shared_videos = expr_videos.intersection(au_videos)

        data_file['VA_Set'][mode] = {}
        for video in va_videos:
            va_data = affwild2_data['VA_Set'][mode][video]
            if video in expr_va_shared_videos:
                expr_data = affwild2_data['EXPR_Set'][mode][video]
                va_data = pd.merge(va_data, expr_data, how='left', on=['path', 'frames_ids']).fillna(value=-2)
            else:
                va_data['label'] = -2

            if video in au_va_shared_videos:
                au_data = affwild2_data['AU_Set'][mode][video]
                va_data = pd.merge(va_data, au_data, how='left', on=['path', 'frames_ids']).fillna(value=-2)
            else:
                for key in AU_list:
                    va_data[key] = -2

            data_file['VA_Set'][mode][video] = va_data

        data_file['EXPR_Set'][mode] = {}
        for video in expr_videos:
            expr_data = affwild2_data['EXPR_Set'][mode][video]
            if video in expr_va_shared_videos:
                va_data = affwild2_data['VA_Set'][mode][video]
                expr_data = pd.merge(expr_data, va_data, how='left', on=['path', 'frames_ids']).fillna(value=-2)
            else:
                expr_data['valence'] = -2
                expr_data['arousal'] = -2
            if video in au_expr_shared_videos:
                au_data = affwild2_data['AU_Set'][mode][video]
                expr_data = pd.merge(expr_data, au_data, how='left', on=['path', 'frames_ids']).fillna(value=-2)
            else:
                for key in AU_list:
                    expr_data[key] = -2
            data_file['EXPR_Set'][mode][video] = expr_data

        data_file['AU_Set'][mode] = {}
        for video in au_videos:
            au_data = affwild2_data['AU_Set'][mode][video]
            if video in au_va_shared_videos:
                va_data = affwild2_data['VA_Set'][mode][video]
                au_data = pd.merge(au_data, va_data, how='left', on=['path', 'frames_ids']).fillna(value=-2)
            else:
                au_data['valence'] = -2
                au_data['arousal'] = -2
            if video in au_expr_shared_videos:
                expr_data = affwild2_data['EXPR_Set'][mode][video]
                au_data = pd.merge(au_data, expr_data, how='left', on=['path', 'frames_ids']).fillna(value=-2)
            else:
                au_data['label'] = -2
            data_file['AU_Set'][mode][video] = au_data

    pickle.dump(data_file, open(args.save_path, 'wb'))


if __name__ == '__main__':
    main()
