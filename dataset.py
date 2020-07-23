import torch
from torch.utils import data
import numpy as np

def sort_batch(batch, targets, lengths, annotations=None):
    """
    Sort a minibatch by length from long to short,
    such that pack_padded_sequences(...) can be used
    """
    _, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]

    if annotations is not None:
        return seq_tensor, target_tensor, annotations[perm_idx]

    return seq_tensor, target_tensor

def pad_and_sort_batch(data):
    """
    data should be a list of tuples (input, target)
    Returns a sorted and padded batch
    """
    batch_size = len(data)
    batch_split = list(zip(*data))

    if len(batch_split) == 3:
        inputs, targets, language = batch_split[0], batch_split[1], batch_split[2]
    else:
        inputs, targets = batch_split[0], batch_split[1]
        language = None

    lengths = [len(x) for x in inputs]
    max_length = max(lengths)

    padded_inputs = np.zeros((batch_size, max_length), dtype=int)
    for i, l in enumerate(lengths):
        padded_inputs[i, 0:l] = inputs[i][0:l]

    if len(batch_split) == 3:
        padded_language = np.zeros((batch_size, max_length), dtype=int)
        for i, l in enumerate(lengths):
            padded_language[i, 0:l] = language[i][0:l]
        language = torch.tensor(padded_language)
        
    return sort_batch(torch.tensor(padded_inputs), torch.tensor(targets).view(-1), torch.tensor(lengths), language)


def sort_batch_tokens(batch, batch_tokens, targets, lengths, annotations=None):
    """
    Sort a minibatch by length from long to short,
    such that pack_padded_sequences(...) can be used
    """
    _, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    token_tensor = batch_tokens[perm_idx]
    target_tensor = targets[perm_idx]

    if annotations is not None:
        return seq_tensor, token_tensor, target_tensor, annotations[perm_idx]

    return seq_tensor, token_tensor, target_tensor


def pad_and_sort_batch_tokens(data):
    """
    data should be a list of tuples (input, target)
    Returns a sorted and padded batch
    """
    batch_size = len(data)
    batch_split = list(zip(*data))

    if len(batch_split) == 4:
        inputs, inputs_tokens, targets, language = batch_split[0], batch_split[1], batch_split[2], batch_split[3]
    else:
        inputs, inputs_tokens, targets, language = batch_split[0], batch_split[1], batch_split[2], None

    lengths = [len(x) for x in inputs]
    max_length = max(lengths)

    padded_inputs = np.zeros((batch_size, max_length), dtype=int)
    for i, l in enumerate(lengths):
        padded_inputs[i, 0:l] = inputs[i][0:l]

    max_length_tokens = max([max([len(t) for t in x]) for x in inputs_tokens])
    padded_inputs_tokens = np.zeros((batch_size, max_length, max_length_tokens), dtype=int)

    for i in range(len(inputs)):
        for j, l_t in enumerate([len(x) for x in inputs_tokens[i]]):
            padded_inputs_tokens[i, j, 0:l_t] = inputs_tokens[i][j][0:l_t]

    if len(batch_split) == 4:
        lengths_language = [len(x) for x in language]
        padded_language = np.zeros((batch_size, max(lengths_language)), dtype=int)
        for i, l in enumerate(lengths_language):
            padded_language[i, 0:l] = language[i][0:l]

        language = torch.tensor(padded_language)

    return sort_batch_tokens(torch.tensor(padded_inputs), torch.tensor(padded_inputs_tokens),
                             torch.tensor(targets).view(-1), torch.tensor(lengths), language)


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DataLoaderDevice():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for batch in self.dataloader:
            yield to_device(batch, self.device)
            
    def __len__(self):
        return len(self.dataloader)
        
        
class Dataset(data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, data, labels, language=None):
        'Initialization'
        self.data = data
        self.labels = labels
        self.language = language

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        X = self.data[index]
        y = self.labels[index]

        if self.language is not None:
            return X, y, self.language[index]

        return X, y


class Dataset_tokens(data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, data, data_tokens, labels, language=None):
        'Initialization'
        self.data = data
        self.data_tokens = data_tokens
        self.labels = labels
        self.language = language

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        X = self.data[index]
        X_words = self.data_tokens[index]
        y = self.labels[index]

        if self.language is not None:
            return X, X_words, y, self.language[index]

        return X, X_words, y