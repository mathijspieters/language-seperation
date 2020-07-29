import vocabulary
import dataset
import utils
import torch
import string
import os

SCRIPT_DIR = os.path.dirname(__file__)
DATA_FOLDER = 'data/'

def preprocess(s, lower=True):
    s = s.translate(str.maketrans('', '', string.punctuation))

    if lower:
        s = s.lower()

    return s

def load_sentimix(dataloader_params, language='spanish', use_balanced_loader=True, binary=False, allowed_words=None):
    df = utils.process_sentimix(language=language)

    labels_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    if binary:
        labels_map = {'negative': 0, 'positive': 1}

    if language == 'spanish':
        lang_id_dict = {'lang1': 0, 'lang2': 1}
        id2lang = ['Eng', 'Spa', 'Unk']
    else:
        lang_id_dict = {'Eng': 0, 'Hin': 1}
        id2lang = ['Eng', 'Hin']

    vocab = vocabulary.Vocabulary(allowed_words=allowed_words)
    for text in df['tokens']:
        for word in preprocess(str(text)).split():
            vocab.count_token(word)

    vocab.build()

    vocab.num_labels = len(labels_map)
    vocab._i2l = list(labels_map.keys())
    vocab._l2i = labels_map
    vocab._id2lang = id2lang

    if binary:
        df = df[df.sentiment != 'neutral']

    tokenizer = lambda x: vocab.sentence2IDs(x)

    labels_train = list(df.loc[df.splitset_label == 'train', 'sentiment'].map(labels_map).astype(int))
    data_train = [tokenizer(preprocess(s)) for s in df.loc[df.splitset_label == 'train', 'tokens']]
    training_set = dataset.Dataset(data_train, labels_train)

    labels_test = list(df.loc[df.splitset_label == 'test', 'sentiment'].map(labels_map).astype(int))
    data_test = [tokenizer(preprocess(s)) for s in df.loc[df.splitset_label == 'test', 'tokens']]
    language_test = [[lang_id_dict.get(t, 2) for t in s.split()] for s in
                     df.loc[df.splitset_label == 'test', 'lang_id']]

    test_set = dataset.Dataset(data_test, labels_test, language_test)

    if use_balanced_loader:
        dataloader_params['shuffle'] = False
        train_generator = torch.utils.data.DataLoader(training_set,
                                                      sampler=utils.ImbalancedDatasetSampler(training_set),
                                                      **dataloader_params,
                                                      collate_fn=dataset.pad_and_sort_batch)
    else:
        train_generator = torch.utils.data.DataLoader(training_set,
                                                      **dataloader_params,
                                                      collate_fn=dataset.pad_and_sort_batch)

    test_generator = torch.utils.data.DataLoader(test_set,
                                                 **dataloader_params,
                                                 collate_fn=dataset.pad_and_sort_batch)

    return train_generator, test_generator, vocab


def load_sentimix_tokens(dataloader_params, language='spanish', use_balanced_loader=False, binary=False,
                         allowed_words=None):

    df = utils.process_sentimix(language=language)

    labels_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    if binary:
        labels_map = {'negative': 0, 'positive': 1}

    if language == 'spanish':
        lang_id_dict = {'lang1': 0, 'lang2': 1}
        id2lang = ['Eng', 'Spa', 'Unk']
    else:
        lang_id_dict = {'Eng': 0, 'Hin': 1}
        id2lang = ['Eng', 'Hin', 'Unk']

    vocab = vocabulary.Vocabulary(allowed_words=allowed_words)
    for text in df['tokens']:
        for word in preprocess(str(text)).split():
            vocab.count_token(word)

    vocab.build(vocab_size=10000)

    vocab_tokens = vocabulary.Vocabulary_tokens()
    for text in df['tokens']:
        for word in preprocess(str(text)).split():
            for token in word:
                vocab_tokens.count_token(token)

    vocab_tokens.build(vocab_size=60)

    vocab.num_labels = len(labels_map)
    vocab._i2l = list(labels_map.keys())
    vocab._l2i = labels_map
    vocab._id2lang = id2lang

    if binary:
        df = df[df.sentiment != 'neutral']

    tokenizer = lambda x: vocab.sentence2IDs(x)
    tokenizer_words = lambda x: vocab_tokens.word2IDs(x)

    labels_train = list(df.loc[df.splitset_label == 'train', 'sentiment'].map(labels_map).astype(int))
    data_train = [tokenizer(preprocess(s)) for s in df.loc[df.splitset_label == 'train', 'tokens']]
    data_train_tokens = [[tokenizer_words(w) for w in vocab.split_sentence(preprocess(s))] for s in
                         df.loc[df.splitset_label == 'train', 'tokens']]

    training_set = dataset.Dataset_tokens(data_train, data_train_tokens, labels_train)

    labels_test = list(df.loc[df.splitset_label == 'test', 'sentiment'].map(labels_map).astype(int))
    data_test = [tokenizer(preprocess(s)) for s in df.loc[df.splitset_label == 'test', 'tokens']]
    data_test_tokens = [[tokenizer_words(w) for w in vocab.split_sentence(preprocess(s))] for s in
                        df.loc[df.splitset_label == 'test', 'tokens']]
    language_test = [[lang_id_dict.get(t, 2) for t in s.split()] for s in
                     df.loc[df.splitset_label == 'test', 'lang_id']]

    test_set = dataset.Dataset_tokens(data_test, data_test_tokens, labels_test, language_test)

    if use_balanced_loader:
        dataloader_params['shuffle'] = False
        train_generator = torch.utils.data.DataLoader(training_set,
                                                      sampler=utils.ImbalancedDatasetSampler(training_set),
                                                      **dataloader_params,
                                                      collate_fn=dataset.pad_and_sort_batch_tokens)
    else:
        train_generator = torch.utils.data.DataLoader(training_set,
                                                      **dataloader_params,
                                                      collate_fn=dataset.pad_and_sort_batch_tokens)

    test_generator = torch.utils.data.DataLoader(test_set,
                                                 **dataloader_params,
                                                 collate_fn=dataset.pad_and_sort_batch_tokens)

    return train_generator, test_generator, vocab, vocab_tokens