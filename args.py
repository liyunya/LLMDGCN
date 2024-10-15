import argparse
import torch

def parameter_parser():
    ap = argparse.ArgumentParser(description="LLMDGCN.")

    ap.add_argument('--dataset', type=str, default='citeseer', help='Dataset.')
    ap.add_argument('--few_shot_perclass', type=int, default=0, help='Number of labeled nodes each class.')
    ap.add_argument('--pseudo_labels_perclass', type=int, default=15, help='Number of high-confidence pseudo labels nodes for per class')
    ap.add_argument('--virtual_node_perclass', type=int, default=20, help='Number of virtual nodes for per class')
    ap.add_argument('--thred_conf', type=float, default=0.9, help='Threshold value of selecting training nodes.')
    ap.add_argument('--thred_sim', type=float, default=0.7, help='Threshold value of predicting edges.')
    ap.add_argument('--hidden_size', type=int, default=32, help='Dimension of hidden embeddings.')
    ap.add_argument('--train_iter', type=int, default=5, help='Maximum number of iterations.')
    ap.add_argument('--topk', type=int, default=10, help='Number of high-confidence nodes for per class.')
    ap.add_argument('--num_layers', type=int, default=2, help='Number pf layers for train gcn.')
    ap.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    ap.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    ap.add_argument('--weight_decay', type=float, default=0.0005, help='L2 regularization weight.')
    ap.add_argument('--epoch', type=int, default=500, help='Number of epochs for trainning gcn.')
    ap.add_argument('--seed', type=int, default=0, help='Random seed.')
    ap.add_argument('--gpu_id', type=str, default='0', help='Id of gpu.')
    ap.add_argument('--no_cuda', action='store_false', default=True, help='Using CUDA or not. Default is True (Using CUDA).')
    ap.add_argument('--repeats', type=int, default=5, help='Repeat.')

    args, _ = ap.parse_known_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_id) if args.no_cuda and torch.cuda.is_available() else 'cpu')

    return args