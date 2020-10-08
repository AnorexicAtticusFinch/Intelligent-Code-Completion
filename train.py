from pathlib import PurePath
from typing import Callable
import torch
import torch.nn as nn
from labml import lab, experiment, tracker, monit
from labml.configs import option
from labml.utils.pytorch import get_modules
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import TrainValidConfigs, Mode

class Configs(TrainValidConfigs):
    
    device = DeviceConfigs()
    model: Module
    text: TextDataset
    n_tokens: int
    tokenizer: Callable
    rhn_depth = 1
    is_save_models = True

    batch_size = 2 # Batch size: 5
    epochs = 32 # Number of epochs: 32
    dropout = 0.2 # Dropout rate: 0.2
    d_model = 512 # Embedding size (Number of features for a word): 300
    rnn_size = 512 # Size of hidden layer: 512
    n_layers = 2 # Number of layers: 2
    seq_len = 512 # Time steps (Memory of characters): 512
    inner_iterations = 100 # Number of iterations: 100

    def run(self):
        for _ in self.training_loop:
            self.run_step()

@option(Configs.optimizer)
def _optimizer(c):
    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.optimizer = 'Adam'
    optimizer.d_model = c.d_model
    return optimizer

class SimpleAccuracyFunc(Module):

    def __call__(self, output, target):
        pred = output.argmax(dim=-1)
        return pred.eq(target).sum().item() / target.shape[1]

@option(Configs.accuracy_func)
def simple_accuracy():
    return SimpleAccuracyFunc()

class CrossEntropyLoss(Module):

    def __init__(self, n_tokens):
        super().__init__()

        self.n_tokens = n_tokens
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        return self.loss(outputs.view(-1, self.n_tokens), targets.view(-1))

@option(Configs.loss_func)
def _loss_func(c):
    return CrossEntropyLoss(c.n_tokens)

@option(Configs.model)
def lstm_model(c):
    from lstm import LSTM_Model
    model = LSTM_Model(n_tokens=c.n_tokens, embedding_size=c.d_model, hidden_size=c.rnn_size, n_layers=c.n_layers)
    return model.to(c.device)

def character_tokenizer(x):
    return list(x)

@option(Configs.tokenizer)
def character():
    return character_tokenizer

@option(Configs.n_tokens)
def _n_tokens(c):
    return c.text.n_tokens

class SourceCodeDataset(TextDataset):

    def __init__(self, path, tokenizer):
        with monit.section('Load data'):
            train = self.load(path / 'train.py')
            valid = self.load(path / 'valid.py')

        super().__init__(path, tokenizer, train, valid, '')

@option(Configs.text)
def source_code(c):
    return SourceCodeDataset(lab.get_data_path(), c.tokenizer)

@option(Configs.train_loader)
def train_loader(c):
    return SequentialDataLoader(text=c.text.train, dataset=c.text, batch_size=c.batch_size, seq_len=c.seq_len)

@option(Configs.valid_loader)
def train_loader(c):
    return SequentialDataLoader(text=c.text.valid, dataset=c.text, batch_size=c.batch_size, seq_len=c.seq_len)

def main():
    conf = Configs()
    conf.model = 'lstm_model'
    experiment.create(name='code_completion', comment='lstm model')
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
        'device.cuda_device': 1
    }, 'run')
    experiment.add_pytorch_models(get_modules(conf))
    # experiment.load('d5ba7f56d88911eaa6629b54a83956dc')
    with experiment.start():
        conf.run()

if __name__ == '__main__':
    main()