from gcn import GCN
from torchmetrics.functional import accuracy
import torch
import torch.nn.functional as F
import numpy as np


def train(args, data):

    num_class = data.labels.max().item() + 1

    edges = np.c_[data.edges[:,1],data.edges[:,0]]
    edges = np.vstack((edges, np.fliplr(edges)))
    edges = np.unique(edges, axis=0).transpose()
    edges = torch.LongTensor(edges).to(args.device)

    #init model
    model = GCN(
            in_size=data.raw_embed.shape[1],
            hidden_size=args.hidden_size,
            out_size=num_class,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc, best_test_acc, best_epoch = 0., 0., 0
    best_logits = None

    t_acc = []
    v_acc = []
    t_loss = []

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        logits = model(data.raw_embed, edges)

        train_loss = F.nll_loss(torch.log(logits[data.train_mask]), data.pseudo_labels[data.train_mask], reduction='none')
        train_loss = (train_loss * data.conf[data.train_mask]).mean()
        train_acc = accuracy(logits[data.train_mask], data.pseudo_labels[data.train_mask], task='multiclass', num_classes=num_class)
        train_loss.backward()
        optimizer.step()
        t_acc.append(round(train_acc.item(), 4))
        t_loss.append(round(train_loss.item(), 4))

        model.eval()
        logits = model(data.raw_embed, edges)
        val_acc = accuracy(logits[data.val_mask], data.pseudo_labels[data.val_mask], task='multiclass', num_classes=num_class)
        test_acc = accuracy(logits[data.test_mask], data.labels[data.test_mask], task='multiclass', num_classes=num_class)
        v_acc.append(round(val_acc.item(), 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_logits = logits

    torch.save(best_logits.detach(), './logits/{}/logits.pt'.format(args.dataset))

    return best_test_acc,best_val_acc