import argparse
import os
import pickle

from .create_annotation_files_Mixed_EXPR_VA import expr_mapping
from .create_annotations import read_data

if __name__ == '__main__':

    Expr_list = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    AffectNetExpr = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    img_a = 'Disgust'
    idx = AffectNetExpr.index(img_a)
    idx_affwild = expr_mapping[idx]
    assert Expr_list[idx_affwild] == img_a



    parser = argparse.ArgumentParser(description='save annotations')
    parser.add_argument('--vis', action='store_true',
                        help='whether to visualize the distribution')
    parser.add_argument('--annot_dir', type=str, default='/home/mvu/Documents/datasets/mixed/affectnet',
                        help='annotation dir')
    parser.add_argument("--dataset_dir", type=str, default='/home/mvu/Documents/datasets/affectnet')
    parser.add_argument('--mixed_EXPR_AU', type=str, default='/home/mvu/Documents/datasets/mixed/mixed_EXPR_VA_annotations.pkl',
                        help='annotation dir')
    args = parser.parse_args()

    output_image_dir = os.path.join(args.annot_dir, 'images')
    all_annotations = {'AV': [], 'EXPR': [], 'path': []}
    train_img_dir = os.path.join(args.dataset_dir, 'Manually_Annotated_compressed', 'Manually_Annotated_Images')
    train_annotations = os.path.join(args.dataset_dir, 'Manually_Annotated_file_lists', 'training.csv')
    read_data(train_img_dir, train_annotations, all_annotations, output_image_dir, is_train=True, save_img=False)

    val_img_dir = os.path.join(args.dataset_dir, 'Manually_Annotated_compressed', 'Manually_Annotated_Images')
    val_annotations = os.path.join(args.dataset_dir, 'Manually_Annotated_file_lists', 'validation.csv')
    read_data(val_img_dir, val_annotations, all_annotations, output_image_dir, save_img=False)

    file_map = {}
    for idx, file in enumerate(all_annotations['path']):
        file_map[file] = {
            'valence': all_annotations['AV'][idx]['valence'],
            'arousal': all_annotations['AV'][idx]['arousal'],
            'expr.png': all_annotations['EXPR'][idx]
        }

    data_mixed = total_data = pickle.load(open(args.mixed_EXPR_AU, 'rb'))
    for idx, file in enumerate(data_mixed['path']):
        if 'affectnet' not in file:
            continue
        origin = file_map[file]
        va = data_mixed['VA'][idx]
        expr = data_mixed['EXPR'][idx]

        assert va[0] == origin['valence']
        assert va[1] == origin['arousal']
        assert expr == expr_mapping[origin['expr.png']]

