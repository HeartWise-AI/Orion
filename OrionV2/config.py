# config.py
import yaml
import torch.nn as nn
import torch.optim as optim
import copy


class Config:
    def __init__(self, config_file="config.yaml", split="TRAIN"):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if split == "TRAIN":        
            self._initialize_train_config(config)
        elif split == "TEST":
            self._initialize_test_config(config)

    def _initialize_train_config(self, config):
        # Dataset and Loader Configurations
        self.dataset_origin = config['dataset_origin']
        self.fps = config['fps']
        self.how_many = config['how_many']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.num_epochs = config['num_epochs']
        self.in_memory = config['in_memory']

        # Experiment Configurations
        self.dataset_path = config['dataset_path']
        self.separator = config['separator']
        self.use_wandb = config['use_wandb']
        self.model_name = config['model_name']
        self.pretrained = config['pretrained']
        self.problem_type = config['problem_type']
        self.classes = config['classes']
        self.label_column = config['label_column']
        self.filename_column = config['filename_column']
        self.split_column = config['split_column']
        self.fps_column = config['fps_column']
        self.label_normalized = config['label_normalized']
        self.threshold = config['threshold']
        self.modulo_5 = config['modulo_5']
        self.num_frames = config['num_frames']
        self.frames_size = config['frames_size']

        # Training Configurations
        self.criterion_name = config['criterion_name']
        self.criterion = self._get_criterion(self.criterion_name)
        self.optimizer_name = config['optimizer_name']
        self.learning_rate = config['learning_rate']
        self.optimizer = None
        self.scheduler = None

        # Scheduler Configurations
        self.lr_scheduler = config.get('lr_scheduler', {})
    
    def _initialize_test_config(self, config):
        # Dataset and Loader Configurations
        self.dataset_path = config["INFERENCE_DATASET_PATH"]
        self.dataset_origin = config["INFERENCE_DATASET_ORIGIN"]
        self.fps = config["INFERENCE_FPS"]
        self.batch_size = config["INFERENCE_BATCH_SIZE"]
        self.num_workers = config["INFERENCE_NUM_WORKERS"]

        # Experiment Configurations
        self.model_path = config["INFERENCE_MODEL_PATH"]
        self.separator = config["INFERENCE_SEPARATOR"]
        self.model_name = config["INFERENCE_MODEL_NAME"]
        self.problem_type = config["INFERENCE_PROBLEM_TYPE"]
        self.classes = config["INFERENCE_CLASSES"]
        self.label_column = config["INFERENCE_LABEL_COLUMN"]
        self.filename_column = config["INFERENCE_FILENAME_COLUMN"]
        self.split_column = config["INFERENCE_SPLIT_COLUMN"]
        self.fps_column = config["INFERENCE_FPS_COLUMN"]
        self.label_normalized = config["INFERENCE_LABEL_NORMALIZED"]
        self.threshold = config["INFERENCE_THRESHOLD"]
        self.modulo_5 = config["INFERENCE_MODULO_5"]
        self.num_frames = config["INFERENCE_NUM_FRAMES"]
        self.frames_size = config["INFERENCE_FRAMES_SIZE"]
        self.in_memory = config["INFERENCE_IN_MEMORY"]
        self.how_many = config["INFERENCE_HOW_MANY"]
        self.stride = config["INFERENCE_STRIDE"]
        self.pretrained = False

    def _get_criterion(self, criterion_name):
        if criterion_name == "L1Loss":
            return nn.L1Loss()
        if criterion_name == "SmoothL1Loss":
            return nn.SmoothL1Loss()
        elif criterion_name == "MSELoss":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")

    def _initialize_optimizer(self, model):
        if self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "RAdam":
            self.optimizer = optim.RAdam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _initialize_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_scheduler['factor'],
            patience=self.lr_scheduler['patience'],
            verbose=self.lr_scheduler['verbose']
        )

    def _initialize_model_path(self, directory):
        self.model_path = directory

    def save(self, file_path):
        with open(file_path, 'w') as file:
            dict_to_save = copy.deepcopy(self.__dict__)
            keys_to_remove = {'criterion', 'optimizer', 'scheduler'} 
            dict_to_save = {k: v for k, v in dict_to_save.items() if k not in keys_to_remove}
            yaml.dump(dict_to_save, file)