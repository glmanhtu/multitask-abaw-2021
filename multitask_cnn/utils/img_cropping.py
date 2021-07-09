import cv2
import numpy as np


def get_landmark_most_points(landmarks):
    min_x, min_y, max_x, max_y = 9999, 9999, 0, 0
    for landmark in landmarks:
        if min_x > landmark[0]:
            min_x = landmark[0]
        if max_x < landmark[0]:
            max_x = landmark[0]
        if min_y > landmark[1]:
            min_y = landmark[1]
        if max_y < landmark[1]:
            max_y = landmark[1]
    return min_x, min_y, max_x, max_y


class CentralCrop(object):
    """Crop the image in a sample.
        Make sure the head is in the central of image
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size, gap_percent=0.05, showing_top=0.65):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.percent = gap_percent
        self.showing_top = showing_top

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        min_x, min_y, max_x, max_y = get_landmark_most_points(landmarks)

        # if max_x > w:
        #     different = max_x - w
        #     min_x -= different
        # if max_y > h:
        #     different = max_y - h
        #     min_y -= different

        max_face = max(max_y - min_y, max_x - min_x)
        gap = min((max_y - min_y), (max_x - min_x)) * self.percent
        distance = int(max_face + gap * 2)
        if distance > min(w, h):
            distance = min(w, h)

        x = int(min_x - gap)
        if x < 0:
            x = 0
        if x + distance < max_x:
            x = int(max_x - distance)
        if x + distance > w:
            x = w - distance
        y = int((min_y - gap) * self.showing_top)
        if y < 0:
            y = 0
        if y + distance < max_y:
            y = int(max_y - distance)
        if y + distance > h:
            y = h - distance

        image = image[y: y + distance, x: x + distance].copy()

        assert image.shape[0] == image.shape[1]

        landmarks = landmarks - np.array([x, y])

        if new_w > image.shape[1]:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)

        landmarks *= np.array([new_w, new_h]) / float(distance)

        sample['image'] = image
        sample['landmarks'] = landmarks

        # image_debug_utils.show_landmarks(sample['image'], landmarks)
        return sample
