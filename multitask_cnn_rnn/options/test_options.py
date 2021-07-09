from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_train = False
        self._parser.add_argument('--teacher_model_path',default = '', type=str,  help='the model to be evaluated')
        self._parser.add_argument("--save_dir", type=str, default = "Predictions")

        self.is_train = False

