import json
import logging
import os
from collections import OrderedDict, defaultdict
from timeit import default_timer as timer

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

from .system import SentenceClassificationSystem
from .utils import AverageMeter, batch_to_device

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    This class performs the training process of sentence classification models.
    """

    def __init__(
            self,
            model, optimizer, criterion, metric,
            config, device,
    ):
        super().__init__()

        self.model = model
        self.system = SentenceClassificationSystem(model, max_sequence_length=config['system']['max_sequence_length'])
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric

        self.config = config

        self.device = device
        self.save_dir = config['paths']['experiment_dir']
        self.log_dir = config['paths']['log_dir']

        self.iteration = 0
        self.best_meter = float('-inf')

        self.summary_writer = None

        self.train_meters = OrderedDict({
            'time': AverageMeter(default='sum'),
            'loss': AverageMeter(default='avg'),
        })
        self.eval_meters = OrderedDict({
            'time': AverageMeter(default='sum'),
            'time_scoring': AverageMeter(default='sum'),
            'loss': AverageMeter(default='avg'),
        })

    def train(
            self,
            train_data_loader,
            dev_data_loader,
            num_epochs: int = 20,
            interval: int = 100,
            verbose: bool = True,
    ):
        num_batches = len(train_data_loader)
        num_iterations = num_epochs * num_batches
        logger.info('Total no. iterations: {}'.format(num_iterations))
        train_iter = iter(train_data_loader)

        with SummaryWriter(self.log_dir) as self.summary_writer:
            if verbose:
                print('Epoch 0, iteration 0:')
            self._validate(dev_data_loader, verbose)

            while self.iteration < num_iterations:
                self.iteration += 1
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    batch = next(train_iter)

                self._train_one_batch(batch)

                if self.iteration % interval == 0 or self.iteration == num_iterations:
                    if verbose:
                        epoch = (self.iteration - 1) // num_batches + 1
                        print('Epoch {}, iteration {}:'.format(epoch, self.iteration))
                        self._verbose('train')
                        self._reset_meters('train')
                    self._validate(dev_data_loader, verbose)

    def _verbose(self, name: str):
        meters = getattr(self, '{}_meters'.format(name))
        for k, v, in meters.items():
            if hasattr(v, 'result'):
                v = v.result()
            print('  {}_{} = {}'.format(name, k, v))

    def _reset_meters(self, name: str):
        meters = getattr(self, '{}_meters'.format(name))
        for meter in meters.values():
            if hasattr(meter, 'reset'):
                meter.reset()

    def _write_summary(self):
        for k, v, in self.eval_meters.items():
            if k == 'time' or k.startswith('time_'):
                continue
            if hasattr(v, 'result'):
                v = v.result()
            self.summary_writer.add_scalar('{}/eval'.format(k), v, self.iteration)

    def _train_one_batch(self, batch) -> None:
        self.model.train()

        start = timer()

        inputs = batch['sentence']
        targets = batch['label']
        batch_to_device(batch, self.device)

        batch_size = len(inputs)

        self.optimizer.zero_grad()

        encoded_inputs = self.system.tokenize(inputs)
        encoded_inputs = batch_to_device(encoded_inputs, self.device)
        targets = targets.to(self.device)

        extended_inputs = {
            'encoded_inputs': encoded_inputs,
            'xmin': batch['xmin'], 'xmax': batch['xmax'], 'ymin': batch['ymin'], 'ymax': batch['ymax'],
            'position': batch['position']
        }

        outputs = self.model(extended_inputs)

        loss = self.criterion(outputs['logit'], targets)
        loss.backward()
        self.optimizer.step()

        end = timer()

        self.train_meters['loss'].update(loss.item(), batch_size)
        self.summary_writer.add_scalar('loss/train', loss, self.iteration)

        self.train_meters['time'].update(end - start)

    def _validate(
            self,
            dev_data_loader,
            verbose=True,
    ) -> None:
        self.eval(dev_data_loader, verbose=verbose)
        self._write_summary()
        self.save_checkpoint()
        meter = self.eval_meters[self.metric]
        if self.iteration > 0 and meter > self.best_meter:
            self.best_meter = meter
            if verbose:
                print('Save best model')
            self.save_best_model()

    def eval(
            self,
            data_loader,
            verbose: bool = True,
    ):
        self._reset_meters('eval')
        labels = list(data_loader.dataset.labels)

        all_outputs = defaultdict(list)
        for batch in data_loader:
            batch_outputs = self._eval_one_batch(batch)
            for k, v in batch_outputs.items():
                if v is not None:
                    all_outputs[k].append(v)

        outputs = {}
        for k, v in all_outputs.items():
            outputs[k] = torch.cat(v)

        start = timer()

        prediction = outputs['prediction'].tolist()
        p, r, f1, _ = precision_recall_fscore_support(labels, prediction, average='macro')
        for name, metric in zip(['p', 'r', 'f1'], [p, r, f1]):
            self.eval_meters['macro_{}'.format(name)] = metric
        self.eval_meters['acc'] = np.mean(np.equal(labels, prediction))

        p, r, f1, _ = precision_recall_fscore_support(labels, prediction, average=None)
        for k, score in enumerate(f1):
            label = self.config['dataset']['labels'][k]
            self.eval_meters['f1_{}'.format(label)] = score

        end = timer()
        self.eval_meters['time_scoring'].update(end - start)

        if verbose:
            self._verbose('eval')

        return outputs

    def _eval_one_batch(self, batch):
        self.model.eval()

        start = timer()

        with torch.no_grad():
            inputs = batch['sentence']
            targets = batch['label']
            batch_to_device(batch, self.device)

            batch_size = len(inputs)

            encoded_inputs = self.system.tokenize(inputs)
            encoded_inputs = batch_to_device(encoded_inputs, self.device)
            targets = targets.to(self.device)

            extended_inputs = {
                'encoded_inputs': encoded_inputs,
                'xmin': batch['xmin'], 'xmax': batch['xmax'], 'ymin': batch['ymin'], 'ymax': batch['ymax'],
                'position': batch['position']
            }

            outputs = self.model(extended_inputs)

            loss = self.criterion(outputs['logit'], targets)

        end = timer()

        self.eval_meters['loss'].update(loss.item(), batch_size)
        self.eval_meters['time'].update(end - start)

        return outputs

    def save_checkpoint(self, dir=None):
        if dir is None:
            dir = self.save_dir
        ckpt_path = os.path.join(dir, 'checkpoint.pth')
        config_path = os.path.join(dir, 'config.json')
        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
        }, ckpt_path)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def load_checkpoint(self, dir=None):
        if dir is None:
            dir = self.save_dir
        ckpt_path = os.path.join(dir, 'checkpoint.pth')
        checkpoint = torch.load(ckpt_path)
        self.iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_best_model(self, dir=None):
        if dir is None:
            dir = self.save_dir
        ckpt_path = os.path.join(dir, 'best.pth')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_best_model(self, dir=None):
        if dir is None:
            dir = self.save_dir
        ckpt_path = os.path.join(dir, 'best.pth')
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint)
