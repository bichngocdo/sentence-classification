import argparse
import json
import logging
import os
from collections import OrderedDict

import pandas as pd

from sentclf.experiment import Experiment

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_experiment.py',
        description='Experiment script for scientific paper sentence classification (SPSC)'
    )
    subparsers = parser.add_subparsers()

    ##########################################################################
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(command='train')

    parser_train.add_argument('--data_file', type=str, required=True,
                              help='data file in csv format')
    parser_train.add_argument('--config', type=str, required=False,
                              help='config file')
    parser_train.add_argument('--device', type=str, required=False,
                              help='device')

    ##########################################################################
    parser_train = subparsers.add_parser('eval')
    parser_train.set_defaults(command='eval')

    parser_train.add_argument('--data_file', type=str, required=True,
                              help='data file in csv format')
    parser_train.add_argument('--model', type=str, required=True,
                              help='model dir')
    parser_train.add_argument('--device', type=str, required=False,
                              help='device')
    parser_train.add_argument('--restore_best', action='store_true',
                              help='restore the best model')

    ##########################################################################
    args = parser.parse_args()
    print(args)

    if args.command == 'train':
        data = pd.read_csv(args.data_file)
        data['label'] = data['label'].astype('category')
        data.info()

        if args.config is not None:
            with open(args.config, 'r') as f:
                config = json.load(f, object_hook=OrderedDict)
        else:
            config = None

        experiment = Experiment(
            config,
            data=data,
            device=args.device,
        )
        experiment.train()

    elif args.command == 'eval':
        data = pd.read_csv(args.data_file)
        data['label'] = data['label'].astype('category')
        data.info()

        config_path = os.path.join(args.model, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f, object_hook=OrderedDict)

        experiment = Experiment(
            config,
            data=data,
            resume=args.model,
            device=args.device,
        )

        if args.restore_best:
            experiment.trainer.load_best_model()

        print('Dev results:')
        experiment.eval()

        print('Test results:')
        experiment.test()
