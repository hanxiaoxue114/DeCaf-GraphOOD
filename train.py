import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, contains_self_loops
from torch_scatter import scatter
import pickle
import time
import csv

from logger import Logger, SimpleLogger
from dataset_ood_valid import get_all_datasets
from correct_smooth import double_correlation_autoscale, double_correlation_fixed
from data_utils import normalize, gen_normalized_adjs, evaluate_ood_valid, to_sparse_tensor, load_fixed_splits, evaluate
# from parse_ood import parse_method, parser_add_main_args
from casual_models import *
from utils import eval_acc, eval_rocauc, accuracy
from gnn_models import *

# NOTE: for consistent data splits, see data_utils.rand_train_idx
np.random.seed(0)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device = torch.device(device)

### Parse args ###

# parser_add_main_args(parser)
# args = parser.parse_args()
# print(args)

def run():
    print (dataname)
    print (gnn)
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser.add_argument('--dataname', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='label')
    parser.add_argument('--shift', type=str, default=shift)
    # parser.add_argument('--sub_dataset', type=str, default='DE')
    parser.add_argument('--inductive', type=bool, default=True)
    parser.add_argument('--gnn_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--method', '-m', type=str, default='h2gcn')
    parser.add_argument('--gnn', type=str, default='GCN')
    parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    if gnn == 'GAT':
        parser.add_argument('--gat_heads', type=int, default=8,
                            help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')
    # parser.add_argument('--print_prop', action='store_true',
    #                     help='print proportions of predicted class')
    parser.add_argument('--train_prop', type=float, default=.7,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.3,
                        help='validation label proportion')
    # parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--bn', action='store_true', help=bn)

    # m: COMO net
    parser.add_argument('--sin_layer', type=int, default=2, help='num layers of mlps')
    parser.add_argument('--sin_hidden', type=int, default=hidden, help='num layers of mlps')
    parser.add_argument('--sin_lr', type=float, default=5e-4, help='num layers of gnns')
    parser.add_argument('--sin_wd', type=float, default=1e-5, help='num layers of gnns')
    parser.add_argument('--sin_dropout', type=float, default=0.5, help='drop out of causal model')
    parser.add_argument('--num_steps_global', type=int, default=1, help='num layers of gnns')
    parser.add_argument('--num_steps_propensity', type=int, default=1, help='num layers of gnns')

    args = parser.parse_args()

    # if args.cpu:
    #     device = torch.device('cpu')
    train_datasets, valid_datasets, test_datasets = get_all_datasets(dataname, ood_type, args)
    n_test = len(test_datasets)
    args.device = device

    predictions_gnn = [[0 for i in range(args.runs)] for j in range(n_test)]
    predictions_ite = [[0 for i in range(args.runs)] for j in range(n_test)]
    predictions_both = [[0 for i in range(args.runs)] for j in range(n_test)]

    if not average_test_datasets:
        test_acc_gnn_list = [[] for i in range(n_test)]
        test_acc_ite_list = [[] for i in range(n_test)]
        test_acc_both_list = [[] for i in range(n_test)]
        test_bacc_gnn_list = [[] for i in range(n_test)]
        test_bacc_ite_list = [[] for i in range(n_test)]
        test_bacc_both_list = [[] for i in range(n_test)]
        test_f1_gnn_list = [[] for i in range(n_test)]
        test_f1_ite_list = [[] for i in range(n_test)]
        test_f1_both_list = [[] for i in range(n_test)]
        test_ll_gnn_list = [[] for i in range(n_test)]
        test_ll_ite_list = [[] for i in range(n_test)]
        test_ll_both_list = [[] for i in range(n_test)]
    else:
        test_acc_gnn_list = []
        test_acc_ite_list = []
        test_acc_both_list = []
        test_bacc_gnn_list = []
        test_bacc_ite_list = []
        test_bacc_both_list = []
        test_f1_gnn_list = []
        test_f1_ite_list = []
        test_f1_both_list = []
        test_ll_gnn_list = []
        test_ll_ite_list = []
        test_ll_both_list = []

    for run in range(args.runs):
        curr_time = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        best_model_path = f'pretrained/{curr_time}.pt'
        train_datasets, valid_datasets, test_datasets = get_all_datasets(dataname, ood_type, args)

        for i in range(len(train_datasets)):
            if len(train_datasets[i].label.shape) == 1:
                train_datasets[i].label = train_datasets[i].label.unsqueeze(1)
            train_datasets[i].label = train_datasets[i].label.to(device)

        for i in range(len(valid_datasets)):
            if len(valid_datasets[i].label.shape) == 1:
                valid_datasets[i].label = valid_datasets[i].label.unsqueeze(1)
            valid_datasets[i].label = valid_datasets[i].label.to(device)

        for i in range(len(test_datasets)):
            # print (test_datasets[i])
            if len(test_datasets[i].label.shape) == 1:
                test_datasets[i].label = test_datasets[i].label.unsqueeze(1)
            test_datasets[i].label = test_datasets[i].label.to(device)



        # infer the number of classes for non one-hot and one-hot labels
        c = int(train_datasets[0].label.max().item() + 1)
        d = train_datasets[0].graph['node_feat'].shape[1]

        train_samples_per_class = [0 for i in range(c)]
        for i in range(len(train_datasets)):
            for cl in range(c):
                # print ((train_datasets[i].label[train_datasets[i].idx]))
                train_samples_per_class[cl] += (train_datasets[i].label[train_datasets[i].idx] == cl).nonzero().size()[0]

        train_samples_per_class = 1/torch.tensor(train_samples_per_class)
        train_samples_per_class = (train_samples_per_class/train_samples_per_class.sum()*c).to(device)
        # print (train_samples_per_class)


        for i in range(len(train_datasets)):
            n = train_datasets[i].graph['num_nodes']
            print(f"num nodes {n} | num classes {c} | num node feats {d}")
        for i in range(len(valid_datasets)):
            n = valid_datasets[i].graph['num_nodes']
            print(f"num nodes {n} | num classes {c} | num node feats {d}")
        for i in range(len(test_datasets)):
            n = test_datasets[i].graph['num_nodes']
            print(f"num nodes {n} | num classes {c} | num node feats {d}")

        if c == 2: c=1
        args.label_dim = c
        args.feature_dim = d
        # whether or not to symmetrize
        if not args.directed and dataname != 'ogbn-proteins':
            for i in range(len(train_datasets)):
                train_datasets[i].graph['edge_index'] = to_undirected(train_datasets[i].graph['edge_index'])
            for i in range(len(valid_datasets)):
                valid_datasets[i].graph['edge_index'] = to_undirected(valid_datasets[i].graph['edge_index'])
            for i in range(len(test_datasets)):
                test_datasets[i].graph['edge_index'] = to_undirected(test_datasets[i].graph['edge_index'])

        for i in range(len(train_datasets)):
            train_datasets[i].graph['edge_index'], train_datasets[i].graph['node_feat'] = \
                train_datasets[i].graph['edge_index'].to(device), train_datasets[i].graph['node_feat'].to(device)

        for i in range(len(valid_datasets)):
            valid_datasets[i].graph['edge_index'], valid_datasets[i].graph['node_feat'] = \
                valid_datasets[i].graph['edge_index'].to(device), valid_datasets[i].graph['node_feat'].to(device)

        for i in range(len(test_datasets)):
            test_datasets[i].graph['edge_index'], test_datasets[i].graph['node_feat'] = \
                test_datasets[i].graph['edge_index'].to(device), test_datasets[i].graph['node_feat'].to(device)



        ### Load method ###
        if args.gnn == 'GCN':
            gnn_model = GCN(in_channels=d,
                        hidden_channels=args.gnn_hidden,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.bn).to(device)
        if args.gnn == 'GAT':
            gnn_model = GAT(in_channels=d,
                        hidden_channels=args.gnn_hidden,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        heads=args.gat_heads).to(device)
        if args.gnn == 'SGC':
            gnn_model = SGC(in_channels=d,
                        hidden_channels=args.gnn_hidden,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.bn).to(device)
        if args.gnn == 'H2GCN':
            gnn_model = H2GCN(in_channels=d,
                        hidden_channels=args.gnn_hidden,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout).to(device)

        # gnn_model = parse_method(args, train_datasets + test_datasets, n, c, d, device)

        # using rocauc as the eval function
        def activation(x, dim=1):
            if c<2: return F.sigmoid(x)
            return F.log_softmax(x, dim)
        if c<2:
            criterion = nn.BCEWithLogitsLoss(weight=train_samples_per_class[1])
            # criterion = nn.BCEWithLogitsLoss()
            eval_func = eval_rocauc
        else:
            criterion = nn.NLLLoss(weight=train_samples_per_class)
            # criterion = nn.NLLLoss()
            eval_func = eval_acc
        # criterion = nn.NLLLoss(weight=train_samples_per_class)
        # criterion = nn.NLLLoss()
        gnn_model.train()
        # print('MODEL:', gnn_model)
        gnn_model.reset_parameters()
        # optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float('-inf')

        for epoch in range(args.epochs):
            loss = 0.
            for i, dataset in enumerate(train_datasets):
                if args.gnn == 'H2GCN':
                    gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i)
                # if epoch == 0: dataset.get_idx_split(shuffle_train_valid=shuffle_train_valid)
                gnn_model.train()
                optimizer.zero_grad()
                out = gnn_model(dataset)
                if args.rocauc or dataname in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
                    if len(dataset.label.shape) or dataset.label.shape[1] == 1:
                        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                    else:
                        true_label = dataset.label

                    loss += criterion(out[dataset.idx], true_label.squeeze(1)[dataset.idx].to(torch.float))
                else:
                    true_label = dataset.label
                    if len(dataset.label.shape)==2 and dataset.label.shape[1] == 1:
                        true_label = true_label.squeeze(1)
                    if c==1:
                        true_label = true_label.to(dtype=torch.float)
                    out = activation(out)
                    loss += criterion(out[dataset.idx], true_label[dataset.idx])
            loss.backward()
            optimizer.step()
            results = evaluate_ood_valid(gnn_model, train_datasets, valid_datasets, test_datasets, eval_func, args=args, gnn=args.gnn)
            # acc, bacc, f1 = accuracy(prediction[test_nodes], labels[test_nodes])
            # val_acc, _, _, _= evaluate(gnn_model, train_datasets, args.gnn, mode='valid')
            # logger.add_result(run, result[:-1])

            if results[1] > best_val:
                best_val = results[1]
                # if dataname != 'ogbn-proteins':
                #     best_out = F.softmax(result[-1], dim=1)
                # else:
                #     best_out = result[-1]
                torch.save(gnn_model.state_dict(), best_model_path)

        gnn_model.load_state_dict(torch.load(best_model_path))

        # get hidden embeddings
        neighbors_list = []
        neighbors_valid_list = []
        features_list = []
        features_valid_list = []
        labels_list = []
        labels_valid_list = []
        n_train_nodes, n_valid_nodes, n_test_nodes = 0, 0, 0

        for i, dataset in enumerate(train_datasets):
            if args.gnn == 'H2GCN':
                gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i)
            old_edge_index = dataset.graph['edge_index']
            edge_index = remove_self_loops(dataset.graph['edge_index'])[0]
            dataset.graph['edge_index'] = edge_index
            neighbors_list.append(gnn_model(dataset, return_hidden=True).detach()[dataset.idx])
            dataset.graph['edge_index'] = old_edge_index
            features_list.append(dataset.graph['node_feat'][dataset.idx])
            labels_list.append(dataset.label[dataset.idx])
            n_train_nodes += dataset.idx.size(0)

        for i, dataset in enumerate(valid_datasets):
            if args.gnn == 'H2GCN':
                gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i, mode='valid')
            old_edge_index = dataset.graph['edge_index']
            edge_index = remove_self_loops(dataset.graph['edge_index'])[0]
            dataset.graph['edge_index'] = edge_index
            neighbors_valid_list.append(gnn_model(dataset, return_hidden=True).detach()[dataset.idx])
            dataset.graph['edge_index'] = old_edge_index
            features_valid_list.append(dataset.graph['node_feat'][dataset.idx])
            labels_valid_list.append(dataset.label[dataset.idx])
            n_valid_nodes += dataset.idx.size(0)

        neighbors = torch.cat(neighbors_list+neighbors_valid_list)
        features = torch.cat(features_list+features_valid_list)
        labels = torch.cat(labels_list+labels_valid_list)

        train_nodes = torch.arange(n_train_nodes)
        valid_nodes = torch.arange(n_train_nodes, n_train_nodes+n_valid_nodes)

        neighbors_test_list = []
        features_test_list = []
        labels_test_list = []

        for i, dataset in enumerate(test_datasets):
            # print (dataset.graph['node_feat'].size())
            if args.gnn == 'H2GCN':
                gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i, mode='test')
            old_edge_index = dataset.graph['edge_index']
            edge_index = remove_self_loops(dataset.graph['edge_index'])[0]
            dataset.graph['edge_index'] = edge_index
            neighbors_test_list.append(gnn_model(dataset, return_hidden=True).detach())
            dataset.graph['edge_index'] = old_edge_index
            features_test_list.append(dataset.graph['node_feat'])
            labels_test_list.append(dataset.label)

        # neighbors_test = torch.cat(neighbors_test_list)
        # features_test = torch.cat(features_test_list)
        # labels_test = torch.cat(labels_test_list)

        sin_model = SIN(args, features, neighbors, labels, train_nodes, valid_nodes)
        sin_model.train_m_model(args)

        sin_model.train_and_test(args)
        ites_train_list = []
        for i in range(len(train_datasets)):
            print ('train', i)
            neighbor, feature, label = neighbors_list[i], features_list[i], labels_list[i]
            ites_train_list.append(sin_model.get_ite(args, neighbor, feature, label))
        ites_valid_list = []
        for i in range(len(valid_datasets)):
            print ('valid', i)
            neighbor, feature, label = neighbors_valid_list[i], features_valid_list[i], labels_valid_list[i]
            ites_valid_list.append(sin_model.get_ite_test(args, neighbor, feature, label))
        ites_test_list = []
        for i in range(len(test_datasets)):
            print ('test', i)
            neighbor, feature, label = neighbors_test_list[i], features_test_list[i], labels_test_list[i]
            ites_test_list.append(sin_model.get_ite_test(args, neighbor, feature, label))

        sin_r_model = SIN_r(args, features, neighbors, labels, train_nodes, valid_nodes)
        sin_r_model.train_m_model(args)

        sin_r_model.train_and_test(args)

        ites_r_train_list = []
        for i in range(len(train_datasets)):
            print ('train', i)
            neighbor, feature, label = neighbors_list[i], features_list[i], labels_list[i]
            ites_r_train_list.append(sin_r_model.get_ite(args, neighbor, feature, label))

        ites_r_valid_list = []
        for i in range(len(valid_datasets)):
            print ('valid', i)
            neighbor, feature, label = neighbors_valid_list[i], features_valid_list[i], labels_valid_list[i]
            ites_r_valid_list.append(sin_r_model.get_ite_test(args, neighbor, feature, label))

        ites_r_test_list = []
        for i in range(len(test_datasets)):
            print ('test', i)
            neighbor, feature, label = neighbors_test_list[i], features_test_list[i], labels_test_list[i]
            ites_r_test_list.append(sin_r_model.get_ite_test(args, neighbor, feature, label))

        # logger.print_statistics(run)
        if average_test_datasets:
            test_acc, test_bacc, test_f1, test_ll = 0., 0., 0., 0.
        for i, dataset in enumerate(test_datasets):
            if args.gnn == 'H2GCN':
                gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i, mode='test')
            labels = dataset.label
            test_nodes = dataset.idx
            if c>1:
                prediction = nn.LogSoftmax(dim=1)(gnn_model(dataset))
                prediction = torch.exp(prediction)
                # prediction = nn.Softmax(dim=1)(gnn_model(dataset))
                # prediction = nn.LogSoftmax(dim=1)(prediction)
            else:
                prediction = activation(gnn_model(dataset))
            # test_acc += eval_acc(prediction, labels[test_nodes])
            # print (prediction[test_nodes])
            acc, bacc, f1, ll = accuracy(prediction[test_nodes], labels[test_nodes])
            predictions_gnn[i][run] = (prediction[test_nodes].detach(), labels[test_nodes])
            if average_test_datasets:
                test_acc += acc
                test_bacc += bacc
                test_f1 += f1
                test_ll += ll
            else:
                test_acc_gnn_list[i].append(acc)
                test_bacc_gnn_list[i].append(bacc)
                test_f1_gnn_list[i].append(f1)
                test_ll_gnn_list[i].append(ll)

                print ("gnn only, test dataset ",i)
                print (f"test acc: {acc}")
                print (f"test bacc: {bacc}")
                print (f"test f1: {f1}")
                print (f"test ll: {ll}")
                print ()

        if average_test_datasets:
            test_acc /= len(test_datasets)
            test_bacc /= len(test_datasets)
            test_f1 /= len(test_datasets)
            test_ll /= len(test_datasets)
            test_acc_gnn_list.append(test_acc)
            test_bacc_gnn_list.append(test_bacc)
            test_f1_gnn_list.append(test_f1)
            test_ll_gnn_list.append(test_ll)

            print ("gnn only")
            print (f"test acc: {test_acc}")
            print (f"test bacc: {test_bacc}")
            print (f"test f1: {test_f1}")
            print (f"test ll: {test_ll}")
            print ()

        def norm(p):
            min = torch.min(p, axis=1).values.unsqueeze(1)
            p = p-min+1e-10
            sum = torch.sum(p, axis=1).unsqueeze(1)
            return p/sum

        alpha = [i*0.1 for i in range(11)]
        beta = [i*0.1 for i in range(11)]
        best_valid_acc = 0.
        best_a, best_b = 0, 0
        for a in alpha:
            for b in beta:
                a_, b_ = a/(a+b+1e-10), b/(a+b+1e-10)
                valid_acc, valid_bacc, valid_f1, valid_ll = 0., 0., 0., 0.
                for i, dataset in enumerate(valid_datasets):
                    if args.gnn == 'H2GCN':
                        gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i, mode='valid')
                    labels = dataset.label
                    valid_nodes =  dataset.idx
                    if c>1:
                        prediction = a_*ites_valid_list[i] + b_*ites_r_valid_list[i]
                        prediction = norm(prediction)
                        # prediction = torch.exp(prediction)
                        # prediction = nn.Softmax(dim=1)(prediction)
                        # prediction = activation(prediction)
                        # prediction = nn.Softmax(dim=1)(prediction)
                    else:
                        prediction = a_*ites_valid_list[i] + b_*ites_r_valid_list[i]
                        prediction = activation(prediction)
                    # test_acc += eval_acc(prediction, labels[test_nodes])

                    acc, bacc, f1, ll = accuracy(prediction, labels[valid_nodes])
                    valid_acc += acc
                    valid_bacc += bacc
                    valid_f1 += f1
                    valid_ll += ll

                valid_f1 /= len(train_datasets)
                if valid_bacc > best_valid_acc:
                    best_valid_acc = valid_bacc
                    best_a, best_b = a_, b_

        test_acc, test_bacc, test_f1, test_ll = 0., 0., 0., 0.
        for i, dataset in enumerate(test_datasets):
            if args.gnn == 'H2GCN':
                gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i, mode='test')
            labels = dataset.label
            test_nodes = dataset.idx
            if c>1:
                prediction = best_a*ites_test_list[i] + best_b*ites_r_test_list[i]
                prediction = norm(prediction)
                # prediction = activation(prediction)
                # prediction = torch.exp(prediction)
                # prediction = nn.Softmax(dim=1)(prediction)
                # prediction = nn.Softmax(dim=1)(prediction)
            else:
                prediction = best_a*ites_test_list[i] + best_b*ites_r_test_list[i]
                prediction = activation(prediction)

            if torch.isnan(prediction).sum()> 0:
                nan_idx = torch.isnan(prediction).nonzero()
                prediction[nan_idx] = 0

            acc, bacc, f1, ll = accuracy(prediction[test_nodes], labels[test_nodes])
            predictions_ite[i][run] = (prediction[test_nodes].detach(), labels[test_nodes])
            if average_test_datasets:
                test_acc += acc
                test_bacc += bacc
                test_f1 += f1
                test_ll += ll
            else:
                test_acc_ite_list[i].append(acc)
                test_bacc_ite_list[i].append(bacc)
                test_f1_ite_list[i].append(f1)
                test_ll_ite_list[i].append(ll)

                print ("ite only, test dataset ",i)
                print (f"test acc: {acc}")
                print (f"test bacc: {bacc}")
                print (f"test f1: {f1}")
                print (f"test ll: {ll}")
                print ()

        if average_test_datasets:
            test_acc /= len(test_datasets)
            test_bacc /= len(test_datasets)
            test_f1 /= len(test_datasets)
            test_ll /= len(test_datasets)
            test_acc_ite_list.append(test_acc)
            test_bacc_ite_list.append(test_bacc)
            test_f1_ite_list.append(test_f1)
            test_ll_ite_list.append(test_ll)

            print ("ite only")
            print (f"test acc: {test_acc}")
            print (f"test bacc: {test_bacc}")
            print (f"test f1: {test_f1}")
            print (f"test ll: {test_ll}")
            print ()


        alpha = [i*0.1 for i in range(11)]
        beta = [i*0.1 for i in range(11)]
        best_valid_f1 = 0.
        best_a, best_b = 0, 0
        for a in alpha:
            for b in beta:
                a_, b_ = a/(a+b+1), b/(a+b+1)
                valid_acc, valid_bacc, valid_f1, valid_ll = 0., 0., 0., 0.
                for i, dataset in enumerate(valid_datasets):
                    if args.gnn == 'H2GCN':
                        gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i, mode='valid')
                    labels = dataset.label
                    valid_nodes = dataset.idx
                    if c>1:
                        prediction = nn.Softmax(dim=1)(gnn_model(dataset))
                        # prediction = (1-a_-b_)*torch.exp(prediction)[valid_nodes] + a_*ites_valid_list[i] + b_*ites_r_valid_list[i]
                        # prediction = gnn_model(dataset)
                        prediction = (1-a_-b_)*prediction[valid_nodes] + a_*ites_valid_list[i] + b_*ites_r_valid_list[i]
                        # prediction = nn.Softmax(dim=1)(prediction)
                        prediction = norm(prediction)
                    else:
                        prediction = gnn_model(dataset)
                        prediction = (1-a_-b_)*prediction[valid_nodes] + a_*ites_valid_list[i] + b_*ites_r_valid_list[i]
                        prediction = activation(prediction)
                    acc, bacc, f1, ll = accuracy(prediction, labels[valid_nodes])
                    valid_acc += acc
                    valid_bacc += bacc
                    valid_f1 += f1
                    valid_ll += ll
                valid_f1 /= len(train_datasets)
                if valid_bacc > best_valid_acc:
                    best_valid_acc = valid_bacc
                    best_a, best_b = a_, b_

        test_acc, test_bacc, test_f1, test_ll = 0., 0., 0., 0.
        for i, dataset in enumerate(test_datasets):
            if args.gnn == 'H2GCN':
                gnn_model.switch_graph(dataset.graph['node_feat'].size()[0], dataset.graph['edge_index'], i, mode='test')
            labels = dataset.label
            test_nodes = dataset.idx
            if c > 1:
                prediction = nn.Softmax(dim=1)(gnn_model(dataset))
                # prediction = (1-best_a-best_b)*torch.exp(prediction) + best_a*ites_test_list[i] + best_b*ites_r_test_list[i]
                # prediction = gnn_model(dataset)
                prediction = (1-a_-b_)*prediction + a_*ites_test_list[i] + b_*ites_r_test_list[i]
                # prediction = nn.Softmax(dim=1)(prediction)
                prediction = norm(prediction)
            else:
                prediction = gnn_model(dataset)
                prediction = (1-best_a-best_b)*prediction + best_a*ites_test_list[i] + best_b*ites_r_test_list[i]
                prediction = activation(prediction)

            if torch.isnan(prediction).sum()> 0:
                nan_idx = torch.isnan(prediction).nonzero()
                prediction[nan_idx] = 0
            acc, bacc, f1, ll = accuracy(prediction[test_nodes], labels[test_nodes])
            predictions_both[i][run] = (prediction[test_nodes].detach(), labels[test_nodes])

            if average_test_datasets:
                test_acc += acc
                test_bacc += bacc
                test_f1 += f1
                test_ll += ll
            else:
                test_acc_both_list[i].append(acc)
                test_bacc_both_list[i].append(bacc)
                test_f1_both_list[i].append(f1)
                test_ll_both_list[i].append(ll)

                print ("ite+gnn, test dataset ",i)
                print (f"test acc: {acc}")
                print (f"test bacc: {bacc}")
                print (f"test f1: {f1}")
                print (f"test ll: {ll}")
                print ()

        if average_test_datasets:
            test_acc /= len(test_datasets)
            test_bacc /= len(test_datasets)
            test_f1 /= len(test_datasets)
            test_ll /= len(test_datasets)
            test_acc_both_list.append(test_acc)
            test_bacc_both_list.append(test_bacc)
            test_f1_both_list.append(test_f1)
            test_ll_both_list.append(test_ll)

            print ("gnn+ite")
            print (f"test acc: {test_acc}")
            print (f"test bacc: {test_bacc}")
            print (f"test f1: {test_f1}")
            print (f"test ll: {test_ll}")
            print ()


    if not average_test_datasets:
        test_acc_gnn_avg, test_acc_gnn_std = [], []
        test_acc_ite_avg, test_acc_ite_std = [], []
        test_acc_both_avg, test_acc_both_std = [], []

        test_bacc_gnn_avg, test_bacc_gnn_std = [], []
        test_bacc_ite_avg, test_bacc_ite_std = [], []
        test_bacc_both_avg, test_bacc_both_std = [], []

        test_f1_gnn_avg, test_f1_gnn_std = [], []
        test_f1_ite_avg, test_f1_ite_std = [], []
        test_f1_both_avg, test_f1_both_std = [], []

        test_ll_gnn_avg, test_ll_gnn_std = [], []
        test_ll_ite_avg, test_ll_ite_std = [], []
        test_ll_both_avg, test_ll_both_std = [], []

        for i in range(n_test):
            test_acc_gnn_avg.append(np.mean(test_acc_gnn_list[i]))
            test_acc_gnn_std.append(np.std(test_acc_gnn_list[i]))
            test_acc_ite_avg.append(np.mean(test_acc_ite_list[i]))
            test_acc_ite_std.append(np.std(test_acc_ite_list[i]))
            test_acc_both_avg.append(np.mean(test_acc_both_list[i]))
            test_acc_both_std.append(np.std(test_acc_both_list[i]))

            test_bacc_gnn_avg.append(np.mean(test_bacc_gnn_list[i]))
            test_bacc_gnn_std.append(np.std(test_bacc_gnn_list[i]))
            test_bacc_ite_avg.append(np.mean(test_bacc_ite_list[i]))
            test_bacc_ite_std.append(np.std(test_bacc_ite_list[i]))
            test_bacc_both_avg.append(np.mean(test_bacc_both_list[i]))
            test_bacc_both_std.append(np.std(test_bacc_both_list[i]))

            test_f1_gnn_avg.append(np.mean(test_f1_gnn_list[i]))
            test_f1_gnn_std.append(np.std(test_f1_gnn_list[i]))
            test_f1_ite_avg.append(np.mean(test_f1_ite_list[i]))
            test_f1_ite_std.append(np.std(test_f1_ite_list[i]))
            test_f1_both_avg.append(np.mean(test_f1_both_list[i]))
            test_f1_both_std.append(np.std(test_f1_both_list[i]))

            test_ll_gnn_avg.append(np.mean(test_ll_gnn_list[i]))
            test_ll_gnn_std.append(np.std(test_ll_gnn_list[i]))
            test_ll_ite_avg.append(np.mean(test_ll_ite_list[i]))
            test_ll_ite_std.append(np.std(test_ll_ite_list[i]))
            test_ll_both_avg.append(np.mean(test_ll_both_list[i]))
            test_ll_both_std.append(np.std(test_ll_both_list[i]))

        result_rows = []
        for i in range(n_test):
            result_rows.append([f'{i+1}', ood_type, inductive, gnn, d, c, hidden, sin_dropout, num_head, bn,
                            f'{test_acc_gnn_avg[i]:.2f} ± {test_acc_gnn_std[i]:.2f}',
                            f'{test_acc_ite_avg[i]:.2f} ± {test_acc_ite_std[i]:.2f}',
                            f'{test_acc_both_avg[i]:.2f} ± {test_acc_both_std[i]:.2f}',
                            f'{test_bacc_gnn_avg[i]:.2f} ± {test_bacc_gnn_std[i]:.2f}',
                            f'{test_bacc_ite_avg[i]:.2f} ± {test_bacc_ite_std[i]:.2f}',
                            f'{test_bacc_both_avg[i]:.2f} ± {test_bacc_both_std[i]:.2f}',
                            f'{test_f1_gnn_avg[i]:.2f} ± {test_f1_gnn_std[i]:.2f}',
                            f'{test_f1_ite_avg[i]:.2f} ± {test_f1_ite_std[i]:.2f}',
                            f'{test_f1_both_avg[i]:.2f} ± {test_f1_both_std[i]:.2f}',
                            f'{test_ll_gnn_avg[i]:.2f} ± {test_ll_gnn_std[i]:.2f}',
                            f'{test_ll_ite_avg[i]:.2f} ± {test_ll_ite_std[i]:.2f}',
                            f'{test_ll_both_avg[i]:.2f} ± {test_ll_both_std[i]:.2f}'
                            ])

    else:
        test_acc_gnn_avg, test_acc_gnn_std = np.mean(test_acc_gnn_list), np.std(test_acc_gnn_list)
        test_acc_ite_avg, test_acc_ite_std = np.mean(test_acc_ite_list), np.std(test_acc_ite_list)
        test_acc_both_avg, test_acc_both_std = np.mean(test_acc_both_list), np.std(test_acc_both_list)

        test_bacc_gnn_avg, test_bacc_gnn_std = np.mean(test_bacc_gnn_list), np.std(test_bacc_gnn_list)
        test_bacc_ite_avg, test_bacc_ite_std = np.mean(test_bacc_ite_list), np.std(test_bacc_ite_list)
        test_bacc_both_avg, test_bacc_both_std = np.mean(test_bacc_both_list), np.std(test_bacc_both_list)

        test_f1_gnn_avg, test_f1_gnn_std = np.mean(test_f1_gnn_list), np.std(test_f1_gnn_list)
        test_f1_ite_avg, test_f1_ite_std = np.mean(test_f1_ite_list), np.std(test_f1_ite_list)
        test_f1_both_avg, test_f1_both_std = np.mean(test_f1_both_list), np.std(test_f1_both_list)

        test_ll_gnn_avg, test_ll_gnn_std = np.mean(test_ll_gnn_list), np.std(test_ll_gnn_list)
        test_ll_ite_avg, test_ll_ite_std = np.mean(test_ll_ite_list), np.std(test_ll_ite_list)
        test_ll_both_avg, test_ll_both_std = np.mean(test_ll_both_list), np.std(test_ll_both_list)

        result_rows = []

        result_rows.append([f'avg', ood_type, inductive, gnn, d, c, hidden, sin_dropout, num_head, bn,
                        f'{test_acc_gnn_avg:.2f} ± {test_acc_gnn_std:.2f}',
                        f'{test_acc_ite_avg:.2f} ± {test_acc_ite_std:.2f}',
                        f'{test_acc_both_avg:.2f} ± {test_acc_both_std:.2f}',
                        f'{test_bacc_gnn_avg:.2f} ± {test_bacc_gnn_std:.2f}',
                        f'{test_bacc_ite_avg:.2f} ± {test_bacc_ite_std:.2f}',
                        f'{test_bacc_both_avg:.2f} ± {test_bacc_both_std:.2f}',
                        f'{test_f1_gnn_avg:.2f} ± {test_f1_gnn_std:.2f}',
                        f'{test_f1_ite_avg:.2f} ± {test_f1_ite_std:.2f}',
                        f'{test_f1_both_avg:.2f} ± {test_f1_both_std:.2f}',
                        f'{test_ll_gnn_avg:.2f} ± {test_ll_gnn_std:.2f}',
                        f'{test_ll_ite_avg:.2f} ± {test_ll_ite_std:.2f}',
                        f'{test_ll_both_avg:.2f} ± {test_ll_both_std:.2f}'
                        ])
        if save_each_run:
            result_row = []
            for test_f1_gnn in test_f1_gnn_list:
                result_row.append(f'{test_f1_gnn:.2f}')
            result_rows.append(result_row)
            result_row = []
            for test_f1_gnn in test_f1_ite_list:
                result_row.append(f'{test_f1_gnn:.2f}')
            result_rows.append(result_row)
            result_row = []
            for test_f1_gnn in test_f1_both_list:
                result_row.append(f'{test_f1_gnn:.2f}')
            result_rows.append(result_row)

    # result_row = [dataname, ood_type, shift, inductive, gnn, hidden, bn, f'{test_acc_gnn_avg:.2f} ± {test_acc_gnn_std:.2f}', f'{test_acc_ite_avg:.2f} ± {test_acc_ite_std:.2f}', f'{test_acc_both_avg:.2f} ± {test_acc_both_std:.2f}']
    split = 'inductive' if inductive else 'transductive'
    os.makedir('results_v1/', exist_ok=True) 
    if save_each_run:
        result_file = f'results_v1/results_each_run/{dataname}_{ood_type}_{split}.csv'
        os.makedir('results_v1/results_each_run/', exist_ok=True) 
    else:
        result_file = f'results_v1/{dataname}_{ood_type}_{split}.csv'
    
    fields = ['dataset', 'ood_type', 'inductive', 'GNN', 'input', 'output', 'hidden', 'sin_dropout', 'head', 'bn', 'acc_gnn', 'acc_ite', 'acc_ite+gnn', 'bacc_gnn', 'bacc_ite', 'bacc_ite+gnn', 'f1_gnn', 'f1_ite', 'f1_ite+gnn', 'll_gnn', 'll_ite', 'll_ite+gnn']

    if not os.path.exists(result_file):
        with open(result_file, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)

    with open(result_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(len(result_rows)):
            csvwriter.writerow(result_rows[i])

    torch.save([predictions_gnn, predictions_ite, predictions_both], f'results_v1/predictions_{dataname}_{ood_type}_{split}_{gnn}_{hidden}_{sin_dropout}.pt')


average_test_datasets = False 
run()


