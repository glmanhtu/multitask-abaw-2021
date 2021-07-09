import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_test import DatasetTest
from models.models import ModelsFactory
from options.test_options import TestOptions
from path import PATH

# RuntimeError: received 0 items of ancdata ###########################
torch.multiprocessing.set_sharing_strategy("file_system")
#########################################################################

PRESET_VARS = PATH()


class Tester:
    def __init__(self):
        self._opt = TestOptions().parse()
        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        test_data_file = PRESET_VARS.Aff_wild2.test_data_file
        self.test_data_file = pickle.load(open(test_data_file, 'rb'))
        self.save_dir = self._opt.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self._test()

    def _test(self):
        self._model.set_eval()
        val_transforms = self._model.resnet50_GRU.backbone.backbone.compose_transforms
        outputs_record = {}
        estimates_record = {}
        frames_ids_record = {}
        self._model.resnet50_GRU.load_state_dict(torch.load(self._opt.teacher_model_path))
        for task in self._opt.tasks:
            task = task + "_Set"
            task_data_file = self.test_data_file[task]['Test_Set']
            outputs_record[task] = {}
            estimates_record[task] = {}
            frames_ids_record[task] = {}
            for i_video, video in enumerate(task_data_file.keys()):
                video_data = task_data_file[video]
                test_dataset = DatasetTest(self._opt, video_data, transform=val_transforms)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self._opt.batch_size,
                                                              shuffle=False,
                                                              num_workers=int(self._opt.n_threads_test),
                                                              drop_last=False)
                track = self.test_one_video(test_dataloader, task=task[:-4])
                outputs_record[task][video] = track['outputs']
                estimates_record[task][video] = track['estimates']
                frames_ids_record[task][video] = track['frames_ids']
                print("Task {} Current {}/{}".format(task[:-4], i_video, len(task_data_file.keys())))
                save_path = '{}/{}.txt'.format(task, video)
                self.save_to_file(track['frames_ids'], track['estimates'], save_path, task=task[:-4])

    def save_to_file(self, frames_ids, predictions, save_path, task= 'AU'):
        save_path = os.path.join(self.save_dir, save_path)
        save_dir = os.path.dirname(os.path.abspath(save_path))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        categories = PRESET_VARS.Aff_wild2.categories[task]
        # filtered out repeated frames
        mask = np.zeros_like(frames_ids, dtype=bool)
        mask[np.unique(frames_ids, return_index=True)[1]] = True
        frames_ids = frames_ids[mask]
        predictions = predictions[mask]
        assert len(frames_ids) == len(predictions)
        with open(save_path, 'w') as f:
            f.write(",".join(categories)+"\n")
            for i, line in enumerate(predictions):
                if isinstance(line, np.ndarray):
                    digits = []
                    for x in line:
                        if isinstance(x, float):
                            digits.append("{:.4f}".format(x))
                        elif isinstance(x, np.int64):
                            digits.append(str(x))
                    line = ','.join(digits)+'\n'
                elif isinstance(line, np.int64):
                    line = str(line)+'\n'
                if i == len(predictions)-1:
                    line = line[:-1]
                f.write(line)

    def test_one_video(self, data_loader, task='AU'):
        track_val = {'outputs': [], 'estimates': [], 'frames_ids': []}
        for i_val_batch, val_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            # evaluate model
            wrapped_v_batch = {task: val_batch}
            self._model.set_input(wrapped_v_batch, input_tasks = [task])
            outputs, _ = self._model.forward(return_estimates=False, input_tasks = [task])
            estimates, _ = self._model.forward(return_estimates=True, input_tasks = [task])
            # store the predictions and labels
            B, N, C = outputs[task][task].shape
            track_val['outputs'].append(outputs[task][task].reshape(B*N, C))
            track_val['frames_ids'].append(np.array([np.array(x) for x in val_batch['frames_ids']]).reshape(B*N, -1).squeeze())
            track_val['estimates'].append(estimates[task][task].reshape(B*N, -1).squeeze())
             
        for key in track_val.keys():
            track_val[key] = np.concatenate(track_val[key], axis=0)
        # assert len(track_val['frames_ids']) -1 == track_val['frames_ids'][-1]
        return track_val


if __name__ == "__main__":
    Tester()
