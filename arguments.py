import argparse

def str2bool(v):
    """
    Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse?rq=1
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--dataset', type=str, default='sentimix')
parser.add_argument('--language', type=str, default='spanish')
parser.add_argument('--augmentation', type=str2bool, nargs='?', default=True)
parser.add_argument('--train', type=str2bool, nargs='?', default=True)
parser.add_argument('--load_dir', type=str, default='0')
parser.add_argument('--model_dir', type=str, default='0')
parser.add_argument('--restore', type=str2bool, nargs='?', default=False)
parser.add_argument('--comet_ml', type=str2bool, nargs='?', default=True)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--save_every', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=10e-4)
parser.add_argument('--weight_decay', type=float, default=10e-6)
parser.add_argument('--z_diff_alpha', type=float, default=10)
parser.add_argument('--rnnlm', type=str2bool, nargs='?', default=False)

parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--fix_emb', type=str2bool, nargs='?', default=False)
parser.add_argument('--load_emb', type=str2bool, nargs='?', default=True)

parser.add_argument('--dropout_prob', type=float, default=0.5)
parser.add_argument('--max_grad_norm', type=float, default=5.0)

parser.add_argument('--print_stats', type=str2bool, nargs='?', default=True)
args, unknown = parser.parse_known_args()