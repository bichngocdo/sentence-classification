import importlib.resources as pkg_resources
import json
import logging
import math
import os
import random
from collections import OrderedDict
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as _default_module_optim
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import update_config, init_obj
from .dataset import ScientificPaperDataset
from .modules import SentenceClassifier
from .trainer import Trainer

MAX_INT = 4294967296

logger = logging.getLogger(__name__)


class Experiment(object):
    """
    The ``Experiment`` class helps to perform experiments with sentence classification models.
    It reads the configuration file, initializes components
    (data loaders, model, optimizer, criterion, metric, trainer),
    provides methods to perform training and evaluation.
    """

    def __init__(
            self,
            config: Dict = None,
            data: DataFrame = None,
            resume: str = None,
            model=None,
            trainer=None,
            criterion=None,
            metric=None,
            optimizer=None,
            device=None,
            default_config_file: str = 'default_config.json'
    ):
        super().__init__()

        # Configuration
        if not resume:
            with pkg_resources.open_text(__package__, default_config_file) as text:
                default_config = json.load(text, object_hook=OrderedDict)
            if config is None:
                config = default_config
            else:
                config = update_config(default_config, config)
            self.config = config
        else:
            with open(os.path.join(resume, 'config.json'), 'r') as f:
                config = json.load(f, object_hook=OrderedDict)
            self.config = config

        # Logger
        self.default_logger(**config['logger'])

        # Random seeds
        self.set_random_seeds()

        # Device
        if device is None:
            device = self.default_device()
        logger.info(f'Use device: {device}')
        self.device = device

        # Paths
        self.set_paths()

        # Data, datasets, data loaders
        self.data = data
        self.train_data, self.dev_data, self.test_data = self.split_data(config['data'])

        config['dataset']['labels'] = list(self.data['label'].cat.categories)

        self.train_dataset = ScientificPaperDataset(self.train_data.copy())
        self.dev_dataset = ScientificPaperDataset(self.dev_data.copy())
        self.test_dataset = ScientificPaperDataset(self.test_data.copy())

        self.train_data_loader = DataLoader(self.train_dataset,
                                            batch_size=config['data_loaders']['train_batch_size'],
                                            shuffle=True)
        self.dev_data_loader = DataLoader(self.dev_dataset,
                                          batch_size=config['data_loaders']['dev_batch_size'],
                                          shuffle=False)
        self.test_data_loader = DataLoader(self.test_dataset,
                                           batch_size=config['data_loaders']['test_batch_size'],
                                           shuffle=False)

        # Model
        if model is None:
            model = self.default_model(config['model'])
        self.model = model

        # Optimizer, criterion, metrics, trainer...
        if optimizer is None:
            optimizer = self.default_optimizer(config['optimizer'])
        self.optimizer = optimizer

        if criterion is None:
            criterion = self.default_criterion()
        self.criterion = criterion

        if metric is None:
            metric = config['metric']
        self.metric = metric

        if trainer is None:
            trainer = self.default_trainer()
        self.trainer = trainer

        if resume:
            self.resume(resume)

    def set_random_seeds(self):
        seeds_config = self.config['seeds']
        if seeds_config['py_seed'] is None:
            seeds_config['py_seed'] = random.randint(0, MAX_INT)
        if seeds_config['np_seed'] is None:
            seeds_config['np_seed'] = random.randint(0, MAX_INT)
        if seeds_config['torch_seed'] is None:
            seeds_config['torch_seed'] = random.randint(0, MAX_INT)
        random.seed(seeds_config['py_seed'])
        np.random.seed(seeds_config['np_seed'])
        torch.manual_seed(seeds_config['torch_seed'])
        logger.info(f"Python random seed: {seeds_config['py_seed']}")
        logger.info(f"NumPy random seed: {seeds_config['np_seed']}")
        logger.info(f"Torch random seed: {seeds_config['torch_seed']}")

    def set_paths(self):
        paths_config = self.config['paths']
        output_dir = paths_config['output_dir']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            logger.info(f'Create output directory: {output_dir}')

        if 'experiment_dir' not in paths_config:
            current_time = datetime.now().strftime('%y%m%d_%H-%M-%S')
            if self.config['dataset']['name'] is not None:
                experiment_name = '{}_{}'.format(self.config['dataset']['name'], current_time)
            else:
                experiment_name = current_time
            experiment_dir = os.path.join(output_dir, experiment_name)
            if not os.path.exists(experiment_dir):
                os.mkdir(experiment_dir)
                logger.info(f'Create experiment directory: {experiment_dir}')
            log_dir = os.path.join(experiment_dir, 'log')

            paths_config['experiment_dir'] = experiment_dir
            paths_config['log_dir'] = log_dir

    def default_logger(self, **kwargs):
        logging.basicConfig(format=kwargs['format'], level=kwargs['level'])

    def default_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def split_data(self, config):
        size = len(self.data)
        dev_size = config['dev_ratio'] if isinstance(config['dev_ratio'], int) \
            else math.ceil(config['dev_ratio'] * size)
        test_size = config['test_ratio'] if isinstance(config['test_ratio'], int) \
            else math.ceil(config['test_ratio'] * size)
        rs1, rs2 = config['random_states']
        if rs1 is None:
            rs1 = random.randint(0, MAX_INT)
        if rs2 is None:
            rs2 = random.randint(0, MAX_INT)
        config['random_states'] = [rs1, rs2]
        train_dev_data, test_data = train_test_split(
            self.data,
            test_size=test_size, shuffle=True, stratify=self.data['label'],
            random_state=rs1,
        )
        train_data, dev_data = train_test_split(
            train_dev_data,
            test_size=dev_size, shuffle=True, stratify=train_dev_data['label'],
            random_state=rs2,
        )
        return train_data, dev_data, test_data

    def default_model(self, config):
        config['output_dim'] = self.data['label'].nunique()
        model = SentenceClassifier(**config)
        model.to(self.device)
        return model

    def default_optimizer(self, config):
        params = [{'params': self.model.sentence_embedder.parameters()},
                  {'params': self.model.projection_layer.parameters(), 'lr': config['learning_rate_top']}]
        return init_obj(config['type'], _default_module_optim, params, lr=config['learning_rate'])

    def default_criterion(self):
        labels = self.data['label'].cat.codes
        label_counts = np.bincount(labels)
        label_weights = len(labels) / (len(label_counts) * label_counts)
        label_weights = torch.tensor(label_weights, dtype=torch.float, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=label_weights)
        return criterion

    def default_trainer(self):
        return Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            metric=self.metric,
            config=self.config,
            device=self.device,
        )

    def resume(self, path):
        logger.info('Load checkpoint from: {}'.format(path))
        self.trainer.load_checkpoint(path)

    def train(self):
        self.trainer.train(
            self.train_data_loader,
            self.dev_data_loader,
            **self.config['trainer'],
        )

    def eval(self):
        self.trainer.eval(
            self.dev_data_loader,
            verbose=self.config['trainer']['verbose'],
        )

    def test(self):
        self.trainer.eval(
            self.test_data_loader,
            verbose=self.config['trainer']['verbose'],
        )
