import pandas as pd

from sentclf.experiment import Experiment

if __name__ == '__main__':
    data_path = 'data'

    data = pd.read_csv('SPSC.csv')
    data['sentence'] = data['sentence'].astype('str')
    data['label'] = data['label'].astype('category')
    data.info()

    config = {
        'dataset': {
            'name': 'Uizard',
        },
        'data_loaders': {
            'train_batch_size': 5,
            'dev_batch_size': 10,
            'test_batch_size': 10,
        },
        'model': {
            'pretrained_model': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
            'pooling_mode': 'mean',
            'dropout_prob': 0.1,
            'use_extended_features': True,
        },
        'optimizer': {
            'type': 'Adam',
            'learning_rate': 5e-4,
            'learning_rate_top': 5e-4,
        },
        'trainer': {
            'num_epochs': 1,
            'interval': 5,
        },
        'seeds': {
            'py_seed': 42,
            'np_seed': 4242,
            'torch_seed': 424242,
        },
        'paths': {
            'output_dir': 'model'
        },
        'logger': {
            'level': 'INFO',
        }
    }

    experiment = Experiment(
        config,
        data=data,
    )

    print('===================Training===================')
    experiment.train()
    print('==================Evaluation==================')
    experiment.trainer.load_best_model()
    experiment.eval()
    print('===================Testing====================')
    experiment.test()
