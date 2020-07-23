from collections import Counter, OrderedDict
import re
import os
import pickle

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOW_TOKEN = '<BOW>'
EOW_TOKEN = '<EOW>'

class OrderedCounter(Counter, OrderedDict):
    """
    Counter that remembers the order elements are first seen
    """

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """
    A vocabulary, assigns IDs to tokens
    """

    def __init__(self, allowed_words=None):
        'Initialization'
        self.freqs = OrderedCounter()
        self._w2i = {}
        self._i2w = []
        self._i2l = []
        self._l2i = {}
        self.num_labels = 0
        self.prefix = 'vocab'
        self.allowed_words = allowed_words

    def count_token(self, t):
        'Increment count of token'
        self.freqs[t] += 1

    def add_token(self, t):
        'Add token to the vocabulary'
        self._w2i[t] = len(self._w2i)
        self._i2w.append(t)

    def __len__(self):
        return len(self._i2w)

    def w2i(self, w):
        'Return ID of word w'
        return self._w2i[w] if w in self._w2i else self._w2i[UNK_TOKEN]

    def i2w(self, i):
        'Return word corresponding to the ID'
        return self._i2w[i] if i < len(self._i2w) else UNK_TOKEN

    def split_sentence(self, s):
        return re.findall(r"[\w']+|[.,!?;]", s)

    def sentence2IDs(self, sentence):
        'Return the list of indices corresponding to the sentence'
        return [self.w2i(w) for w in self.split_sentence(sentence)]

    def IDs2sentence(self, indices):
        'Return the sentence corresponding to the list of indices'
        return " ".join([self.i2w(i) for i in indices])

    def build(self, min_freq=0, vocab_size=-1):
        'Create the vocabulary after the tokens have been counted'
        self.add_token(PAD_TOKEN)  # reserve 0 for <pad>
        self.add_token(UNK_TOKEN)  # reserve 1 for <unk>

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        tok_freq = tok_freq[:vocab_size-2]
        for tok, freq in tok_freq:
            if freq >= min_freq:
                if not self.allowed_words or tok in self.allowed_words:
                    self.add_token(tok)
                
    def build_from_list(self, word_list):
        'Create the vocabulary from the provided word_list'
        self.add_token(PAD_TOKEN)  # reserve 0 for <pad>
        self.add_token(UNK_TOKEN)  # reserve 1 for <unk>
        
        for tok in word_list:
            self.add_token(tok)

    def save_to_file(self, path):
        """Save Vocabulary to file, useful for replicating results without needing the full dataset"""
        with open(os.path.join(path, '%s_i2w.pkl' % self.prefix), 'wb') as fp:
            pickle.dump(self._i2w, fp)

        with open(os.path.join(path, '%s_i2l.pkl' % self.prefix), 'wb') as fp:
            pickle.dump(self._i2l, fp)

    def build_from_file(self, path):
        with open(os.path.join(path, '%s_i2w.pkl' % self.prefix), 'rb') as fp:
            self._i2w = pickle.load(fp)

        for i, w in enumerate(self._i2w):
            self._w2i[w] = i

        with open(os.path.join(path, '%s_i2l.pkl' % self.prefix), 'rb') as fp:
            self._i2l = pickle.load(fp)

        for i, l in enumerate(self._i2l):
            self._l2i[l] = i

        self.num_labels = len(self._l2i)


class Vocabulary_tokens(Vocabulary):
    def __init__(self, allowed_words=None):
        super(Vocabulary_tokens, self).__init__(allowed_words=allowed_words)
        self.prefix = 'token_vocab'

    def build(self, min_freq=0, vocab_size=-1):
        'Create the vocabulary after the tokens have been counted'
        self.add_token(PAD_TOKEN)  # reserve 0 for <pad>
        self.add_token(UNK_TOKEN)  # reserve 1 for <unk>
        self.add_token(BOW_TOKEN)
        self.add_token(EOW_TOKEN)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        tok_freq = tok_freq[:vocab_size-2]
        for tok, freq in tok_freq:
            if freq >= min_freq:
                if not self.allowed_words or tok in self.allowed_words:
                    self.add_token(tok)

    def word2IDs(self, word):
        return [self.w2i(BOW_TOKEN)] + [self.w2i(t) for t in word] + [self.w2i(EOW_TOKEN)]

    def build_from_list(self, word_list):
        'Create the vocabulary from the provided word_list'
        self.add_token(PAD_TOKEN)  # reserve 0 for <pad>
        self.add_token(UNK_TOKEN)  # reserve 1 for <unk>
        self.add_token(BOW_TOKEN)
        self.add_token(EOW_TOKEN)

        for tok in word_list:
            self.add_token(tok)


