import torch
import numpy as np
import os
from utils import enlarge_training_set, edge_prediction
from args import parameter_parser
from train_gcn import train
from load_data import load_data

args = parameter_parser()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def node_classification():
    test_acc_list = []
    for i in range(args.repeats):
        print("----------repeat:{}-------\n".format(i))

        #load data
        data = load_data(args, seed=i)

        test_acc_iter_list = []
        val_acc_iter_list = []
        for i in range(args.train_iter):
            if i > 0:
                # enlarge the training set
                flag = enlarge_training_set(data, args)
                if not flag:
                    break
                # supplement the edge set
                else:
                    edge_prediction(data, args)
            # train gcn
            test_acc, val_acc = train(args, data)

            test_acc_iter_list.append(round(test_acc.item(), 4))
            val_acc_iter_list.append(round(val_acc.item(), 4))

        # select the best result
        max_idx = np.argmax(np.array(val_acc_iter_list))
        print("Test_acc: {}\n".format(test_acc_iter_list[max_idx]))
        test_acc_list.append(test_acc_iter_list[max_idx])

    print('Result test acc list: {}\n'.format(test_acc_list))
    print('Std: {}\n'.format(np.array(test_acc_list).std()*100))
    print('Avg test acc: {:.6f}\n'.format(sum(test_acc_list) / args.repeats))


if __name__ == '__main__':
    dictory_logits = './logits/{}/'.format(args.dataset)
    if not os.path.exists(dictory_logits):
        os.makedirs(dictory_logits)
    node_classification()