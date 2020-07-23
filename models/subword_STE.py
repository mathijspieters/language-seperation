import torch
import torch.nn as nn
from models.bernoulli import STE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.rcnn import RCNNEncoder
from models.lstm import LSTMEncoder

class Subword_STE(nn.Module):
    """
    TODO: explain
    """

    def __init__(self,
                 embedding_dim: int = 300,
                 embedding_dim_tokens: int = 30,
                 hidden_dim: int = 200,
                 hidden_dim_tokens: int = 30,
                 vocab_size: int = 0,
                 vocab_size_tokens: int = 0,
                 dropout_prob: float = 0.5,
                 bidirectional: bool = True,
                 encoder: str = 'lstm',
                 share_emb: bool = True
                 ):
        super(Subword_STE, self).__init__()

        self.vocab_size_tokens = vocab_size_tokens
        self.share_emb = share_emb

        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.emb_token_l1 = nn.Embedding(vocab_size_tokens, embedding_dim_tokens, padding_idx=0)
        if not share_emb:
            self.emb_token_l2 = nn.Embedding(vocab_size_tokens, embedding_dim_tokens, padding_idx=0)

        self.encoder = RCNNEncoder(embedding_dim, hidden_dim, bidirectional=bidirectional) if encoder == 'cnn' else\
                        LSTMEncoder(embedding_dim, hidden_dim, bidirectional=bidirectional)

        self.bernoulli_gate = STE(hidden_dim * (int(bidirectional) + 1), 1)
        self.dropout = nn.Dropout(dropout_prob)

        self.token_encoder_l1 = nn.LSTM(embedding_dim_tokens, hidden_dim_tokens, bidirectional=False)
        self.token_encoder_l2 = nn.LSTM(embedding_dim_tokens, hidden_dim_tokens, bidirectional=False)

        self.token_prob_l1 = nn.Linear(hidden_dim_tokens, vocab_size_tokens)
        self.token_prob_l2 = nn.Linear(hidden_dim_tokens, vocab_size_tokens)


    def get_z(self, sentence):
        mask = (sentence != 0)
        lengths = mask.sum(1)
        embeds = self.emb(sentence)
        embeds = self.dropout(embeds)

        out, _ = self.encoder(embeds, mask, lengths)

        out_z = self.bernoulli_gate.layer(out)

        out_z = self.bernoulli_gate.sigmoid(out_z)

        return out_z


    def generate_token(self, max_length=10, l1=True, temperature=1, prefix=None):
        start_token = torch.LongTensor([[2]]).to(device)

        if l1:
            enc = self.token_encoder_l1
            prob = self.token_prob_l1
            embedder = self.emb_token_l1
        else:
            enc = self.token_encoder_l2
            prob = self.token_prob_l2
            embedder = self.emb_token_l1 if self.share_emb else self.emb_token_l2

        emb = embedder(start_token)
        output = []

        out, (h_n, c_n) = enc(emb)
        word_probs = prob(out)

        for i in range(max_length):
            word_probs = torch.nn.functional.softmax(temperature * word_probs, dim=2)
            prob_dist = torch.distributions.Categorical(word_probs)
            pred = prob_dist.sample()

            if prefix is not None and i < len(prefix):
                pred = torch.LongTensor([[prefix[i]]]).to(device)

            if pred.item() == 3:
                break
            emb = embedder(pred).reshape(1, 1, -1)
            out, (h_n, c_n) = enc.forward(emb, (h_n, c_n))
            word_probs = prob(out)

            output.append(pred.item())

        return output

    def word_prob(self, tokens, l1=True):
        start_token = torch.LongTensor([[2]]).to(device)

        if l1:
            enc = self.token_encoder_l1
            prob = self.token_prob_l1
            embedder = self.emb_token_l1
        else:
            enc = self.token_encoder_l2
            prob = self.token_prob_l2
            embedder = self.emb_token_l1 if self.share_emb else self.emb_token_l2

        emb = embedder(start_token)

        out, (h_n, c_n) = enc(emb)
        word_probs = prob(out)

        log_prob = 0

        for i in range(len(tokens)):
            token = torch.LongTensor([[tokens[i]]]).to(device)

            word_probs = torch.nn.functional.softmax(word_probs, dim=2)
            prob_dist = torch.distributions.Categorical(word_probs)

            prob_value = prob_dist.log_prob(token)

            log_prob += prob_value

            emb = embedder(token).reshape(1, 1, -1)
            out, (h_n, c_n) = enc.forward(emb, (h_n, c_n))
            word_probs = prob(out)

        return log_prob.item() / len(tokens)


    def forward(self, sentence, sentence_tokens):
        mask = (sentence != 0)
        lengths = mask.sum(1)
        embeds = self.emb(sentence)

        embeds = self.dropout(embeds)

        out, _ = self.encoder(embeds, mask, lengths)

        out_z = self.bernoulli_gate(out)

        batch_size, max_sentence_length, max_word_length = sentence_tokens.shape

        sentence_tokens = sentence_tokens.reshape(batch_size * max_sentence_length, max_word_length)

        token_embeds_l1 = token_embeds_l2 = self.emb_token_l1(sentence_tokens)
        if not self.share_emb:
            token_embeds_l2 = self.emb_token_l2(sentence_tokens)

        token_embeds_l1 = self.dropout(token_embeds_l1)
        token_embeds_l2 = self.dropout(token_embeds_l2)

        out_l1, _ = self.token_encoder_l1(token_embeds_l1)
        out_l2, _ = self.token_encoder_l2(token_embeds_l2)

        prob_l1 = self.token_prob_l1(out_l1)
        prob_l2 = self.token_prob_l2(out_l2)

        prob_l1 = prob_l1.reshape(batch_size, max_sentence_length, max_word_length, self.vocab_size_tokens)
        prob_l2 = prob_l2.reshape(batch_size, max_sentence_length, max_word_length, self.vocab_size_tokens)

        return out_z, prob_l1, prob_l2


    def get_loss(self, token_inputs, z, p_l1, p_l2):
        z_sample = z.unsqueeze(-1)

        langauge_probs = z_sample * p_l1 + (1-z_sample) * p_l2

        batch_size, max_sentence_length, max_word_length, vocab_size_token = langauge_probs.shape

        langauge_probs = langauge_probs.reshape(batch_size * max_sentence_length, max_word_length, vocab_size_token)
        langauge_probs = langauge_probs[:, :-1, :]

        token_inputs = token_inputs.reshape(batch_size * max_sentence_length, max_word_length)
        token_inputs = token_inputs[:, 1:]

        token_mask = (token_inputs != 0)

        distributions = torch.distributions.Categorical(logits=langauge_probs)

        loss = distributions.log_prob(token_inputs)

        loss = torch.where(token_mask, loss, torch.Tensor([0]).to(device))

        final_loss = -loss.sum(-1).mean()

        token_pred = langauge_probs.argmax(dim=-1)
        correct = (token_pred == token_inputs).float()

        correct = torch.where(token_mask, correct, torch.Tensor([0]).to(device))
        acc = correct.sum() / token_mask.sum()

        optional = {'acc':acc.item()}

        return final_loss, optional
