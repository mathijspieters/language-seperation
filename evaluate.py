from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np


def generate_words(model, vocab_tokens):
    for t in [1, 3, 5]:
        words = []
        for _ in range(3):
            word = model.generate_token(l1=True, temperature=t)
            words.append("".join([vocab_tokens.i2w(i) for i in word]))

        print("TEMP=%d   %s" % (t, " ".join(words)))

    for t in [1, 3, 5]:
        words = []
        for _ in range(3):
            word = model.generate_token(l1=False, temperature=t)
            words.append("".join([vocab_tokens.i2w(i) for i in word]))

        print("TEMP=%d   %s" % (t, " ".join(words)))


def determine_ROC(model, sorted_z_values, vocab, vocab_tokens):
    with open('data/en-2012/en.txt', 'r') as f:
        doc = f.read()
        words_eng = doc.split('\n')

    with open('data/es-2012/es.txt', 'r') as f:
        doc = f.read()
        words_spa = doc.split('\n')

    words_eng_dict = dict([w.split(' ') for w in words_eng if len(w.split(' ')) == 2])
    words_spa_dict = dict([w.split(' ') for w in words_spa if len(w.split(' ')) == 2])

    words_eng_dict = dict((k, int(v)) for k, v in words_eng_dict.items())
    words_spa_dict = dict((k, int(v)) for k, v in words_spa_dict.items())

    all_words = set(words_eng_dict.keys()) | set(words_spa_dict.keys())
    all_words = dict([(w, 'eng' if words_eng_dict.get(w, 0) > words_spa_dict.get(w, 0) else 'spa') for w in all_words])
    overlap = set(words_eng_dict.keys()) & set(words_spa_dict.keys())
    words_eng = [w for w in words_eng_dict.keys() if w not in overlap]
    words_spa = [w for w in words_spa_dict.keys() if w not in overlap]

    pred_values = []
    true_values = []
    words = []

    for word in words_spa[:1000]:
        tokens = [vocab_tokens.w2i(w) for w in word]
        prob_l1 = np.exp(model.word_prob(tokens, l1=True))
        prob_l2 = np.exp(model.word_prob(tokens, l1=False))
        norm = (prob_l1 + prob_l2)

        prob_l1 /= norm
        prob_l2 /= norm

        pred_values.append(prob_l1)
        true_values.append(0)
        words.append(word)

    for word in words_eng[:1000]:
        tokens = [vocab_tokens.w2i(w) for w in word]
        prob_l1 = np.exp(model.word_prob(tokens, l1=True))
        prob_l2 = np.exp(model.word_prob(tokens, l1=False))
        norm = (prob_l1 + prob_l2)

        prob_l1 /= norm
        prob_l2 /= norm

        pred_values.append(prob_l1)
        true_values.append(1)
        words.append(word)

    pred_values = np.array(pred_values)
    fpr, tpr, _ = roc_curve(true_values, pred_values)
    roc_auc = auc(fpr, tpr)

    fpr1, tpr1, _ = roc_curve(true_values, 1-pred_values)
    roc_auc1 = auc(fpr1, tpr1)

    print("ROC=%.3f" % max(roc_auc, roc_auc1))

    N_values = [50, 100, 500, 1000, 2000]

    for N in N_values:
        true = []
        pred = []

        for (k, v) in sorted_z_values[:N]:
            w = vocab.i2w(k)
            if w in all_words:
                true.append(int(all_words[w] == 'eng'))
                pred.append(v)

        for (k, v) in sorted_z_values[::-1][:N]:
            w = vocab.i2w(k)
            if w in all_words:
                true.append(int(all_words[w] == 'eng'))
                pred.append(v)

        fpr, tpr, _ = roc_curve(true, pred)
        roc_auc = auc(fpr, tpr)

        fpr1, tpr1, _ = roc_curve(true, 1-np.array(pred))
        roc_auc1 = auc(fpr1, tpr1)

        print("N=%4d ROC=%.3f" % (N, max(roc_auc, roc_auc1)))