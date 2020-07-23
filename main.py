import utils
import torch
import numpy as np
import torch.optim as optim
from collections import defaultdict, Counter

from tqdm import tqdm

from arguments import args

from models.subword_STE import Subword_STE

import dataloader
import dataset
import evaluate

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(dev)

args.model = 'subword_STE'

PATH = utils.args_to_results_path(args)

def eval(model, test_generator, vocab, vocab_tokens):
    model.eval()

    z_values_dict = defaultdict(list)
    l_mu_counter = defaultdict(list)

    languages = vocab._id2lang

    for batch_inputs, batch_inputs_tokens, batch_targets, *batch_language in test_generator:
        z = model.get_z(batch_inputs)

        lengths = (batch_inputs != 0).sum(1)
        if len(batch_language) > 0:

            for inputs, z_values, length, language in zip(batch_inputs, z, lengths, batch_language[0]):
                for i_, z_, l_ in zip(inputs[:length], z_values[:length], language[:length]):
                    lang = languages[l_.item()]
                    z_values_dict[i_.item()].append(z_.item())
                    l_mu_counter[lang].append(z_.item())

        else:
            for inputs, z_values, length in zip(batch_inputs, z, lengths):
                for i_, z_ in zip(inputs[:length], z_values[:length]):
                    z_values_dict[i_.item()].append(z_.item())

    for key in l_mu_counter:
        print("%s - %.4f - %.4f- %d" % (key,
                                        np.mean(l_mu_counter[key]),
                                        np.mean([round(x) for x in l_mu_counter[key]]),
                                        len(l_mu_counter[key])))

    mean_z_values_dict = Counter()
    for key in z_values_dict:
        mean_z_values_dict[key] = np.mean(z_values_dict[key])

    sorted_z_values = mean_z_values_dict.most_common()
    for (k, v) in sorted_z_values[:20]:
        print(vocab.i2w(k), round(v, 4))

    print()
    for (k, v) in sorted_z_values[::-1][:20]:
        print(vocab.i2w(k), round(v, 4))

    evaluate.generate_words(model, vocab_tokens)

    evaluate.determine_ROC(model, sorted_z_values, vocab, vocab_tokens)


def train(model, train_generator, test_generator, vocab, vocab_tokens, epochs=args.epochs):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    step = 0

    for epoch in range(epochs):

        epoch_loss = 0
        epoch_step = 0
        extra_keys = set()
        summary_dict = defaultdict(float)

        if epoch % 5 == 0 or dev == 'cpu':
            eval(model, test_generator, vocab, vocab_tokens)
            utils.save_model(model, PATH)

        model.train()


        with tqdm(total=len(train_generator), leave=False) as t:
            for batch_inputs_sentence, batch_inputs_tokens, batch_targets in train_generator:

                optimizer.zero_grad()
                z, p_l1, p_l2 = model(batch_inputs_sentence, batch_inputs_tokens)

                loss, optional_loss = model.get_loss(batch_inputs_tokens, z, p_l1, p_l2)

                for k in optional_loss:
                    extra_keys.add(k)
                    summary_dict[k] += optional_loss[k]

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                summary_dict['loss'] += loss.item()
                epoch_loss += loss.item()

                step += 1
                epoch_step += 1

                t.set_description("Epoch: %2d    Loss: %.3f   %s" % (epoch, epoch_loss / epoch_step,
                                                                     "   ".join(
                                                                         ["%s: %.4f " % (
                                                                         k, summary_dict[k] / epoch_step)
                                                                          for k in extra_keys])))
                t.update()

            if dev == 'cuda':
                print("Epoch: %2d    Loss: %.3f   %s" % (epoch, epoch_loss / epoch_step,
                                                         "   ".join(
                                                             ["%s: %.4f " % (k, summary_dict[k] / epoch_step)
                                                              for k in extra_keys])))


def main():
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 1}

    if args.dataset == 'sentimix':
        train_generator, test_generator, vocab, vocab_tokens = dataloader.load_sentimix_tokens(params,
                                                                                               binary=True,
                                                                                               use_balanced_loader=False,
                                                                                               language=args.language)
    elif args.dataset == 'hatespeech':
        train_generator, test_generator, vocab, vocab_tokens = dataloader.load_hatespeech_tokens(params,
                                                                                               use_balanced_loader=False)

    elif args.dataset == 'sst':
        train_generator, test_generator, dev_generator, vocab, vocab_tokens = dataloader.load_SST_tokens(params)
    else:
        exit("Unknown dataset")


    train_generator = dataset.DataLoaderDevice(train_generator, DEVICE)
    test_generator = dataset.DataLoaderDevice(test_generator, DEVICE)

    model = Subword_STE(vocab_size=len(vocab),
                        vocab_size_tokens=len(vocab_tokens),
                        embedding_dim=100,
                        hidden_dim=200,
                        share_emb=False)

    utils.initialize_model_(model)

    print("Language: ", args.language)
    print("Vocab size: %d" % len(vocab))
    print("Tokens: %s" % " ".join(vocab_tokens._i2w))

    vocab.save_to_file(PATH)
    vocab_tokens.save_to_file(PATH)

    if args.print_stats:
        utils.print_parameters(model)
        print('Data loaded!\nTrain batches: %d\nTest batches: %d\nVocab size: %d\nNumber of labels %d' %
              (len(train_generator), len(test_generator), len(vocab), vocab.num_labels))

        utils.print_dataset_statistics(train_generator)
        print(model)

    model = model.to(DEVICE)

    train(model, train_generator, test_generator, vocab, vocab_tokens)


if __name__ == '__main__':
    main()
