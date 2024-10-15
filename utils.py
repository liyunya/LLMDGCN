import torch
import numpy as np
from copy import deepcopy

def enlarge_training_set(data, args):
    # load the output of the last iteration of GCN
    logits = torch.load('./logits/{}/logits.pt'.format(args.dataset))
    confs, preds = torch.max(logits, dim=1)

    # get high-confidence nodes
    preds_class = [[] for _ in range(data.labels.max() + 1)]
    for i, (pd, cf) in enumerate(zip(preds, confs)):
        if i in data.train_set:
            preds_class[pd].append((i, cf.item()))
    index_topk = []
    conf_topk = []
    for pred_conf in preds_class:
        pred_conf = sorted(pred_conf, key=lambda x: x[1], reverse=True)
        for index, conf in pred_conf[:args.topk]:
            index_topk.append(index)
            conf_topk.append(conf)
    index_topk_new = []
    conf_topk_new = []
    for i in range(len(index_topk)):
        if conf_topk[i]>args.thred_conf and index_topk[i] not in data.train_id:
            index_topk_new.append(index_topk[i])
            conf_topk_new.append(conf_topk[i])

    if len(index_topk_new)==0:
        return 0

    data.train_id = torch.cat((data.train_id, torch.LongTensor(index_topk_new).to(args.device)))

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.labels.shape[0])]).to(args.device)
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.labels.shape[0])]).to(args.device)
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.labels.shape[0])]).to(args.device)

    # replace pseudo labels and confidences
    data.pseudo_labels[index_topk_new] = preds[index_topk_new]
    data.conf[index_topk_new] = torch.FloatTensor(conf_topk_new).to(args.device)
    return 1

def edge_prediction(data, args):
    # load the output of the last iteration of GCN
    logits = torch.load('./logits/{}/logits.pt'.format(args.dataset))

    # compute the inter-category probability matrix
    B = deepcopy(logits)
    B[data.train_id_initial] = data.label_onehot_train
    H = torch.mm(B[:data.num_nodes].t(), data.adj_initial)
    H = torch.mm(H, B)
    H = H/torch.sum(H, dim=1, keepdim=True)

    # obtain the number of missing edges
    Edge_remain = data.degrees[:data.num_nodes] * torch.mm(logits[:data.num_nodes], H) - torch.mm(data.adj_initial, logits)
    Edge_remain[Edge_remain<0] = 0.
    Edge_remain = torch.round(Edge_remain)
    Edge_remain = Edge_remain.type(torch.int8)

    embedding = data.raw_embed
    pred = torch.argmax(logits,dim=1)
    train_idx_class = [[] for _ in range(data.labels.max().item() + 1)]
    for i in range(pred.shape[0]):
        train_idx_class[pred[i]].append(i)

    value_list = []
    index_list = []
    for idx in train_idx_class:
        # similarity
        dot = torch.mm(embedding[:data.num_nodes, :], embedding[torch.LongTensor(idx).to(args.device)].t())*(~(data.adj_initial[:,torch.LongTensor(idx).to(args.device)]==1))
        norm_1 = torch.norm(embedding[:data.num_nodes, :], dim=1)
        norm_2 = torch.norm(embedding[torch.LongTensor(idx).to(args.device)], dim=1)
        sim = dot/(norm_1.unsqueeze(1)*norm_2.unsqueeze(0))

        value, index = torch.sort(sim, dim=1, descending=True)
        value_list.append(value)
        index_list.append(index)

    new_edge = np.ones((0, 2), dtype=int)
    for i in range(data.num_nodes):
        for c in range(data.labels.max() + 1):
            ref_num = Edge_remain[i][c]
            idx_list = (index_list[c][i])[:ref_num]
            vl_list = (value_list[c][i])[:ref_num]
            idx_list = idx_list[vl_list>args.thred_sim]
            idx_edge = torch.LongTensor(train_idx_class[c]).to(args.device)[idx_list]
            new_edge = np.vstack((new_edge, np.c_[i*np.ones(idx_list.shape, dtype=int), idx_edge.cpu().numpy()]))

    data.edges = np.vstack((data.edges_initial, new_edge))

