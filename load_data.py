import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from copy import deepcopy
import torch


class Dataset:
    def __init__(self, name):
        self.dataset_name = name


def get_dataset(device, dataset, path):
    data = Dataset(dataset)

    idx_labels = np.genfromtxt(
        "{}idx2label".format(path), dtype=np.dtype(str))
    labels = idx_labels[:, -1]
    if dataset == 'citeseer':
        class_map = {x: i for i, x in enumerate(['Agents', 'ML', 'IR', 'DB', 'HCI', 'AI'])}
        data_labels = np.array([class_map[l] for l in labels])
    elif dataset == 'cora':
        class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                             'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning',
                                             'Theory'])}
        data_labels = np.array([class_map[l] for l in labels])
    else:
        data_labels = np.array(labels, dtype=int)

    idx = idx_labels[:, 0]
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}edges".format(path), dtype=np.dtype(str))
    data_edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    degrees = np.genfromtxt('{}degrees'.format(path), dtype=str)[:, 1]
    data_degrees = np.array(degrees, dtype=float).reshape((degrees.shape[0], 1))

    data.labels = torch.LongTensor(data_labels).to(device)
    data.edges = data_edges
    data.num_nodes = torch.tensor(len(data_labels)).to(device)
    data.degrees = torch.FloatTensor(data_degrees).to(device)

    return data

def get_pred_train_id(pesudo_label_perclass, path, num_class, train_set):
    pred_conf = torch.load('{}pred_conf.pt'.format(path))
    preds = pred_conf['pred']
    confs = pred_conf['conf']
    preds_class = [[] for _ in range(num_class)]
    for i, (pd, cf) in enumerate(zip(preds, confs)):
        if i in train_set:
            preds_class[pd].append((i, cf.item()))
    index_topk = []
    conf_topk = []
    for pred_conf in preds_class:
        pred_conf = sorted(pred_conf, key=lambda x: x[1], reverse=True)
        for index, conf in pred_conf[:pesudo_label_perclass]:
            index_topk.append(index)
            conf_topk.append(conf)
    return index_topk, preds, confs


def get_virtual_nodes(path):
    virtual_raw_embed = torch.load('{}virtual_embed.pt'.format(path))
    return virtual_raw_embed


def load_data(args, seed):
    path = 'dataset/' + args.dataset +'/'
    # load data
    data = get_dataset(args.device, args.dataset, path)
    np.random.seed(seed)
    num_class = data.labels.max() + 1
    # split data
    node_id = [[] for _ in range(num_class)]
    few_shot_id_perclass = [[] for _ in range(num_class)]
    for i in range(data.labels.shape[0]):
        node_id[data.labels[i]].append(i)
    train_id = []
    remain_id = []
    for i in range(num_class):
        np.random.shuffle(node_id[i])
        train_id = train_id + node_id[i][:args.few_shot_perclass]
        few_shot_id_perclass[i] = node_id[i][:args.few_shot_perclass]
        remain_id = remain_id + node_id[i][args.few_shot_perclass:]
    data.train_id_initial = torch.LongTensor(train_id).to(args.device)
    data.label_onehot_train = F.one_hot(data.labels[data.train_id_initial], num_classes=num_class).float()

    np.random.shuffle(remain_id)
    data.train_set = torch.LongTensor(remain_id[1500:]).to(args.device)
    data.val_id = torch.LongTensor(remain_id[0:500]).to(args.device)
    data.test_id = torch.LongTensor(remain_id[500:1500]).to(args.device)

    # get pseudo labels, confidences and high-confidence nodes
    pred_train_id, pred, conf = get_pred_train_id(args.pseudo_labels_perclass, path, num_class, data.train_set)
    pred = pred.to(args.device)
    data.pred = pred

    train_id = train_id + pred_train_id
    pred_train_id = torch.LongTensor(pred_train_id).to(args.device)

    raw_embed = torch.load('{}raw_embed.pt'.format(path)).to(args.device)

    if args.virtual_node_perclass>0:
        # get virtual nodes
        virtual_raw_embed = get_virtual_nodes(path).to(args.device)
        M = int(virtual_raw_embed.shape[0])
        virtual_nodes_labels = []
        for i in range(num_class):
            virtual_nodes_labels = virtual_nodes_labels + [i] * args.virtual_node_perclass

        data.train_id = torch.LongTensor(train_id + list(range(data.num_nodes, data.num_nodes + M))).to(args.device)
        data.labels = torch.cat((data.labels, torch.tensor(virtual_nodes_labels).to(args.device)))
        data.raw_embed = torch.cat((raw_embed, virtual_raw_embed)).to(args.device)
        data.conf = torch.cat((conf, torch.ones((M)))).to(args.device)

        v_number = data.num_nodes + M
    else:
        data.train_id = torch.LongTensor(train_id).to(args.device)
        data.text_embed = raw_embed
        data.conf = conf.to(args.device)
        v_number = data.num_nodes

    data.pseudo_labels = deepcopy(data.labels)
    data.pseudo_labels[pred_train_id] = pred[pred_train_id]
    data.conf[data.train_id_initial] = torch.ones((data.train_id_initial.shape[0])).to(args.device)

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(v_number)]).to(args.device)
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(v_number)]).to(args.device)
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(v_number)]).to(args.device)

    adj = sp.coo_matrix((np.ones(data.edges.shape[0]), (data.edges[:,0], data.edges[:, 1])),
                             shape=(data.num_nodes, data.labels.shape[0]), dtype=np.float32)
    # initial adj
    data.adj_initial = torch.FloatTensor(adj.todense()).to(args.device)
    # initial edge set
    data.edges_initial = deepcopy(data.edges)
    return data
