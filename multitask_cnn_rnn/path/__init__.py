import collections

Dataset_Info = collections.namedtuple("Dataset_Info", 'data_file categories test_data_file')


class PATH(object):
    def __init__(self, opt=None):
        self.Aff_wild2 = Dataset_Info(data_file='/home/mvu/Documents/datasets/mixed/affwild-2/annotations_wshared.pkl',
                                      test_data_file='/home/mvu/Documents/datasets/mixed/affwild-2/test_set.pkl',
                                      categories={'AU': ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'],
                                                  'EXPR': ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness',
                                                           'Sadness', 'Surprise'],
                                                  'VA': ['valence', 'arousal']})
        self.MODEL_DIR = '/home/mvu/Documents/pretrained'
