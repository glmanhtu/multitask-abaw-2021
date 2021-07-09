import math
import pickle

import numpy as np
import os
import torch
import torchaudio
from PIL import Image

from path import PATH
from utils.audio_transform import ComposeWithInvert, AmpToDB, Normalize
from .dataset import DatasetBase

PRESET_VARS = PATH()


class DatasetMixedVA(DatasetBase):
    def __init__(self, opt, train_mode='Train', transform=None):
        super(DatasetMixedVA, self).__init__(opt, train_mode, transform)
        self._name = 'dataset_Mixed_VA'
        self._train_mode = train_mode
        if transform is not None:
            self._transform = transform
        self._read_dataset_paths()
        if self._opt.audio_mode:
            self._meta_data = self.read_audio_video_metadata()

            # audio params
            self.window_size = 20e-3
            self.window_stride = 20e-3
            self.sample_rate = 44100
            num_fft = 2 ** math.ceil(math.log2(self.window_size * self.sample_rate))
            window_fn = torch.hann_window
            n_mels = self._opt.audio_n_features
            self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=n_mels,
                                                                        n_fft=num_fft,
                                                                        win_length=int(
                                                                            self.window_size * self.sample_rate),
                                                                        hop_length=int(self.window_stride
                                                                                       * self.sample_rate),
                                                                        window_fn=window_fn)

            self.audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])

    def read_audio_video_metadata(self):
        metadata = {}
        for video in self._meta_data.keys():
            metadata[video] = {}
            data = self._meta_data[video]
            audio_file = data['audio_file']
            audio_metadata = torchaudio.info(audio_file)
            metadata[video] = {**data}
            metadata[video]['total_audio_frames'] = audio_metadata.num_frames
        return metadata

    def _get_all_label(self):
        return self._data['label']

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        images = []
        labels = []
        img_paths = []
        exprs = []
        df = self.sample_seqs[index]
        for i, row in df.iterrows():
            img_path = row['path']

            with Image.open(img_path) as im:
                image = im.convert('RGB')
            label = row[PRESET_VARS.Aff_wild2.categories['VA']].values.astype(np.float32)
            exprs.append(row['label'])
            images.append(image)
            labels.append(label)
            img_paths.append(img_path)

        images = self._transform(images)

        # pack data
        sample = {'image': torch.stack(images, dim=0),
                  'label': np.array(labels).astype(np.float32),
                  'sub': np.array(exprs).astype(np.long),
                  'path': img_paths,
                  'index': index
                  }
        if self._opt.audio_mode:
            audio_indexes, waveform = self._read_audio(df)
            sample['audio_indexes'] = audio_indexes
            sample['audio_features'] = waveform.squeeze().transpose(0, 1)
            sample['audio_length'] = waveform.shape[2]
        return sample

    def _read_audio(self, seq):
        frames_ids = seq['frames_ids'].tolist()
        from_frame, to_frame = frames_ids[0], frames_ids[-1]
        image_file = seq['path'].iloc[0]
        img_path = os.path.normpath(image_file).split(os.path.sep)
        video_name = img_path[-2]
        meta_data = self._meta_data[video_name]
        timestamp_name = video_name.replace('_right', '').replace('_left', '')
        timestamps = self._timestamps[timestamp_name] / 1000
        sr = self.sample_rate
        numb_frames = int(timestamps[to_frame] * sr) - int(timestamps[from_frame] * sr)
        frame_offset = int(timestamps[from_frame] * sr)
        seq_timestamps = timestamps[from_frame:to_frame + 1]
        waveform, sample_rate = torchaudio.load(meta_data['audio_file'], frame_offset=frame_offset,
                                                num_frames=numb_frames)
        waveform = self.audio_transform(waveform)
        waveform = self.audio_spec_transform(waveform)

        weight = seq_timestamps - min(seq_timestamps)
        weight = weight / max(weight)
        audio_indexes = weight * (waveform.shape[2] - 1)

        audio_indexes = np.around(audio_indexes).astype(np.int)
        return audio_indexes, waveform

    def _read_dataset_paths(self):
        self._data = self._read_path_label(PRESET_VARS.Aff_wild2.data_file)
        # sample them
        seq_len = self._opt.seq_len
        self.sample_seqs = []
        if self._train_mode == 'Train':
            N = seq_len // 2
        else:
            N = seq_len
        for video in self._data.keys():
            data = self._data[video]
            current_idx = 0
            while True:
                start, end = current_idx, current_idx + seq_len
                if end >= len(data):
                    start, end = len(data) - seq_len, len(data)
                new_df = data.iloc[start:end]

                if len(new_df) == seq_len:
                    frames_ids = new_df['frames_ids'].tolist()
                    discard_flag = False
                    for frame_id in range(1, len(frames_ids)):
                        # For the sequences that has missing frames, we discard it
                        if frames_ids[frame_id] - frames_ids[frame_id - 1] > 1:
                            discard_flag = True
                            break
                    if not discard_flag:
                        self.sample_seqs.append(new_df)
                current_idx += N
                if current_idx >= len(data):
                    break
        self._ids = np.arange(len(self.sample_seqs))
        self._dataset_size = len(self._ids)

    def __len__(self):
        return self._dataset_size

    def _read_path_label(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        if self._opt.audio_mode:
            self._meta_data = data['Metadata']
            self._timestamps = data['VideoTimestamps']
        data = data['VA_Set']
        # read frames ids
        if self._train_mode == 'Train':
            data = data['Train_Set']
        elif self._train_mode == 'Validation':
            data = data['Validation_Set']
        elif self._train_mode == 'Test':
            data = data['Test_Set']
        else:
            raise ValueError("train mode must be in : Test, Train, Validation")
        return data
