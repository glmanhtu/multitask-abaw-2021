import argparse
import csv
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from utils.face_extractor import CentralCrop


def show_landmarks(image, batch_landmarks, rect=None):
    """Show image with landmarks"""
    plt.figure()
    plt.imshow(image)
    for idx, landmarks in enumerate(batch_landmarks):
        # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        plt.annotate(idx, (landmarks[0], landmarks[1]))

    if rect is not None:
        plt.gca().add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor='none'))
    plt.axis('off')
    plt.ioff()
    # plt.pause(0.05)
    # plt.clf()
    plt.show()


def read_data(img_dir, annotation_file, annotations, image_dir, is_train=False, save_img=False):
    keys = ['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks', 'expression',
            'valence', 'arousal']
    cropper = CentralCrop(112)
    with open(annotation_file) as f:
        for idx, row in enumerate(csv.DictReader(f, fieldnames=keys)):
            if idx == 0 and is_train:
                continue
            if int(row['expression']) > 6:
                # We ignore 7: Contempt, 8: None, 9: Uncertain, 10: No-Face because of unable to merge with affwild2
                continue
            cropped_img_path = os.path.join(image_dir, row['subDirectory_filePath'])
            if save_img:
                img_path = os.path.join(img_dir, row['subDirectory_filePath'])
                img = cv2.imread(img_path)
                landmarks = np.asarray(row['facial_landmarks'].split(';')).astype(np.float).reshape((-1, 2))
                img, landmarks = cropper(img, landmarks)
                # rect = np.asarray([row['face_x'], row['face_y'], row['face_width'], row['face_height']]).astype(np.int)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # show_landmarks(img, landmarks)
                os.makedirs(os.path.dirname(cropped_img_path), exist_ok=True)
                cv2.imwrite(cropped_img_path, img)
            annotations['path'].append(cropped_img_path)
            annotations['AV'].append({'valence': float(row['valence']), 'arousal': float(row['arousal'])})
            annotations['EXPR'].append(int(row['expression']))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='save annotations')
    parser.add_argument('--vis', action='store_true',
                        help='whether to visualize the distribution')
    parser.add_argument('--annot_dir', type=str, default='/home/mvu/Documents/datasets/mixed/affectnet',
                        help='annotation dir')
    parser.add_argument("--dataset_dir", type=str, default='/home/mvu/Documents/datasets/affectnet')

    args = parser.parse_args()

    data_file_path = os.path.join(args.annot_dir, 'annotations.pkl')
    output_image_dir = os.path.join(args.annot_dir, 'images')
    all_annotations = {'AV': [], 'EXPR': [], 'path': []}
    train_img_dir = os.path.join(args.dataset_dir, 'Manually_Annotated_compressed', 'Manually_Annotated_Images')
    train_annotations = os.path.join(args.dataset_dir, 'Manually_Annotated_file_lists', 'training.csv')
    read_data(train_img_dir, train_annotations, all_annotations, output_image_dir, is_train=True)

    val_img_dir = os.path.join(args.dataset_dir, 'Manually_Annotated_compressed', 'Manually_Annotated_Images')
    val_annotations = os.path.join(args.dataset_dir, 'Manually_Annotated_file_lists', 'validation.csv')
    read_data(val_img_dir, val_annotations, all_annotations, output_image_dir)

    with open(data_file_path, 'wb') as f:
        pickle.dump(all_annotations, f)




