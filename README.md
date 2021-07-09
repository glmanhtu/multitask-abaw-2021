# Multitask Multi-database Emotion Recognition

This is the repository containing the solution for 2nd ABAW 2021 Competition

Pretrained model: to be updated

## Train a new model

### Step 1: Download all required datasets
- For AffWild 2 database, download the cropped and aligned version from the organizer

### Step 2: Create annotation files for the training
- Create Affwild2 dataset file
    - Go to `create_annotation_files/affwild2` folder
    - Update the configurations in the `create_train_val_annotation_file.py` file according to your machine
    - Run `python create_train_val_annotation_file.py`
    - Update the configurations in the `create_train_val_with_shared_annotation.py` file according to your machine
    - Run `python create_train_val_with_shared_annotation.py`
- Create Mixed VA dataset file
    - Go to `create_annotation_files/afew_va` folder
    - Update the configurations in the `read_annotation_and_align_faces.py` file according to your machine
    - Run `python read_annotation_and_align_faces.py`
    - Update the configurations in the `create_annotation_files_Mixed_VA.py` file according to your machine
    - Run `python create_annotation_files_Mixed_VA.py`
- Create Mixed EXPR dataset file
    - Go to `create_annotation_files/expw` folder
    - Update the configurations in the `create_annotations.py` file according to your machine
    - Run `python create_annotations.py`
    - Update the configurations in the `create_annotation_files_Mixed_EXPR.py` file according to your machine
    - Run `python create_annotation_files_Mixed_EXPR.py`


### Step 3: Train teacher CNN model
- Go to `multitask_cnn` folder
- Update the path file: `__init__.py`
- Run `python train.py --force_balance --name teacher_image_size_112_b64 --batch_size 64 --image_size 112`
  
### Step 4: Train student CNN model
- Run `python train.py --force_balance --name student_image_size_112_b64 --batch_size 64 --image_size 112 --pretrained_teacher_model /path/to/pretrained_teacher_model.pth`

### Step 5: Train teacher GRU model
- Go to `multitask_cnn_rnn` folder
- Update the path file: `__init__.py`
- Run `python train.py --name rnn_teacher_image_size_112_b64_seq32 --batch_size 16 --seq_len 32 --rnn_split_size 7 --image_size 112 --pretrained_resnet50_model /path/to/pretrained_cnn_student_model.pth`

### Step 6: Train student GRU model

- Run `python train.py --name rnn_teacher_image_size_112_b64_seq32 --batch_size 16 --seq_len 32 --rnn_split_size 7 --image_size 112 --pretrained_resnet50_model /path/to/pretrained_cnn_student_model.pth --pretrained_teacher_model /path/to/pretrained_rnn_teacher_model.pth --use_shared_annotations`
