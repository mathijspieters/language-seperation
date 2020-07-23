import os
import pandas as pd
import numpy as np
import torch
from collections import Counter
import sklearn.metrics as metrics
from sklearn.preprocessing import normalize
import math
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(__file__)

data_folder = 'data/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = os.path.dirname(__file__)
DATA_FOLDER = 'data/'
SST_FOLDER = 'sst'
GLOVE_FOLDER = 'embeddings/glove'
FASTTEXT_FOLDER = 'embeddings/fastText'


def parse_CoNLL(sentence):
    """
    Parse a CoNLL formatted sentence
    """
    lines = sentence.split('\n')
    lines = [line.split('\t') for line in lines]
    data = {'uid': int(lines[0][1]), 'sentiment': lines[0][2],
            'tokens': " ".join([line[0] for line in lines[1:] if len(line) > 1]),
            'lang_id': " ".join([line[1] for line in lines[1:] if len(line) > 1])}

    return data


def process_sentimix(language='hindi'):
    """
    Process the sentimix data
    """

    assert language in ['hindi', 'spanish'], 'Language must be either hindi or spanish'

    sentimix_file_train = 'train_conll_hinglish.txt' if language == 'hindi' else 'train_conll_spanglish.txt'

    with open(os.path.join(script_dir, data_folder, sentimix_file_train), 'r') as f:
        doc_train = f.read()

    sentences_train = doc_train.split('\n\n')
    processed_sentences_train = [parse_CoNLL(sentence) for sentence in sentences_train if len(sentence) > 0]

    df = pd.DataFrame(processed_sentences_train)

    train, test = train_test_split(df, test_size=0.1)

    print(Counter(train.sentiment))
    print(Counter(test.sentiment))

    train = train.copy()
    test = test.copy()
    train['splitset_label'] = 'train'
    test['splitset_label'] = 'test'

    df = pd.concat([train, test])

    return df


def process_hatespeech():
    """

    Process hate speech dataset
    """

    file = 'hate_speech.tsv'

    df = pd.read_csv(os.path.join(script_dir, data_folder, file), sep='\t', names=['tokens', 'label'])

    train, test = train_test_split(df, test_size=0.1)

    train = train.copy()
    test = test.copy()
    train['splitset_label'] = 'train'
    test['splitset_label'] = 'test'

    df = pd.concat([train, test])

    return df


def calculate_class_weights(original_dataloader):
    """
    Calculate the inverse relative frequency of the classes in the dataloader.
    Can be used as an input for a loss function.
    Alternative for balanced dataloader
    """

    labels = []
    for _, t in original_dataloader:
        labels = [*labels, *t.cpu().numpy()]

    freqs = Counter(labels).most_common()
    values = np.zeros((len(freqs)))
    for i, f in freqs:
        values[i] = 1. / f

    return normalize(values[:, np.newaxis], axis=0).ravel()


def print_dataset_statistics(dataloader):
    """
    Print the dataset statistics
    """
    lengths_words = []
    lengths_tokens = []
    for i, t, _ in dataloader:
        i = (i != 0).sum(1)
        t = (t != 0).sum(1).sum(1).float()
        lengths_words = [*lengths_words, *i.cpu().numpy()]
        lengths_tokens = [*lengths_tokens, *t.cpu().numpy()]

    print("Mean number of words per sentences: %.3f ± %.3f" % (np.mean(lengths_words), np.std(lengths_words)))
    print("Mean number of tokens per sentences: %.3f ± %.3f" % (np.mean(lengths_tokens), np.std(lengths_tokens)))


def print_parameters(model):
    """
    Print model parameters
    """
    total_param = 0
    total_train_param = 0
    for _, p in model.named_parameters():
        total_param += np.prod(p.shape)
        total_train_param += np.prod(p.shape) if p.requires_grad else 0

    print("Total parameters: {}".format(total_param))
    if total_param != total_train_param:
        print("Total trainable parameters: {}".format(total_train_param))


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Initialize for LSTM layer
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def print_mean_gradient(model, param=None):
    for name, p in model.named_parameters():
        if p.requires_grad:
            if param is None or (param is not None and name == param):
                if p.grad is None:
                    print("{:20s} {} mean gradient=NONE".format(name, p.shape))
                else:
                    print("{:20s} {} mean gradient={} var gradient={}".format(name, p.shape, p.grad.mean(), p.grad.var()))


def initialize_model_(model):
    """
    Initialize the model using Glorot
    Article: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("emb_"):
            print("{:10s} {:20s} {} requires gradient={}".format("unchanged", name, p.shape, p.requires_grad))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {} requires gradient={}".format("xavier_n", name, p.shape, p.requires_grad))
            xavier_uniform_n_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {} requires gradient={}".format("zeros", name, p.shape, p.requires_grad))
            torch.nn.init.constant_(p, 0.)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {} requires gradient={}".format("xavier", name, p.shape, p.requires_grad))
            torch.nn.init.xavier_uniform_(p)
        else:
            print("{:10s} {:20s} {} requires gradient={}".format("unchanged", name, p.shape, p.requires_grad))


def args_to_results_path(args):
    """
    Create path for model, if model is only evaluated return the specified path.
    For every new experiment a new folder is created, using an incremental number
    """
    path = os.path.join('results', args.dataset)

    if not os.path.exists(os.path.join(path, '0')):
        path = os.path.join(path, '0')
    else:
        files = os.listdir(path)
        previous_counts = [int(m) for m in files if m.isdigit()]
        count = max(previous_counts)
        path = os.path.join(path, str(count + 1))

    os.makedirs(path)

    save_args(path, args)

    return path


def save_args(path, args):
    """
    Save the train arguments to the specified path
    """
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write("%s=%s\n" % (key, str(value)))


def save_model(model, path):
    """
    Save the model to the specified path, using an incremental model name.
    This prevents overwriting the previously saved model.
    """

    files = os.listdir(path)
    previous_counts = [int(m.split('-')[1][:-3]) for m in files if m[:5] == 'model']
    count = max(previous_counts) + 1 if len(previous_counts) > 0 else 0

    torch.save(model.state_dict(), os.path.join(path, 'model-%d.pt' % count))


def load_model(model, path, n=0):
    """
    Load the saved model
    """
    path = os.path.join(path, 'model-%s.pt' % str(n))

    if not os.path.exists(path):
        raise NotADirectoryError('There is no model at the specified path: %s' % path)

    missing, unexpected = model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
    print("Loaded model: %s" % path)
    if len(missing):
        print("Missing keys: ", missing)
    if len(unexpected):
        print("Unexpected keys: ", unexpected)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Source: https://github.com/ufoym/imbalanced-dataset-sampler
    
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.labels[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
