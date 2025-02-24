import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import time
import enlighten
import logging
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import *
from layers import *
import tracemalloc

cudaid = "cuda:0"
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")

class SIN(nn.Module):
    def __init__(self, args, features, neighbors, labels, train_nodes, valid_nodes):
        super().__init__()
        self.device = args.device
        self.features = features
        self.labels = labels
        self.neighbors = neighbors
        # self.g = g
        if args.label_dim > 1:
            self.activation = nn.LogSoftmax(dim=1)
        else: self.activation = nn.Sigmoid()
        self.out_dim = args.label_dim
        self.emb = MLP(
                dim_input = args.feature_dim,
                dim_hidden = args.sin_hidden,
                dim_output = args.sin_hidden,
                num_layers = 1,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'relu').to(self.device)
        self.emb_opt = optim.Adam(self.emb.parameters(),
                                         lr=args.sin_lr,
                                         weight_decay=args.sin_wd)

        # m(x) -> y
        self.m_net = MLP(
                dim_input = args.sin_hidden,
                dim_hidden = args.sin_hidden,
                dim_output = args.label_dim,
                num_layers = 2,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'none').to(self.device)
        self.m_net_opt = optim.Adam(self.m_net.parameters(),
                                         lr=args.sin_lr,
                                         weight_decay=args.sin_wd)

        # h(T)
        if args.gnn == 'H2GCN':
            gnn_hidden = args.gnn_hidden * (2**(args.num_layers+1)-1)
        else: gnn_hidden = args.gnn_hidden
        self.h_net = MLP(
                dim_input = gnn_hidden,
                dim_hidden = args.sin_hidden,
                num_layers = args.sin_layer,
                dim_output = args.label_dim,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'none').to(device)

        self.h_net_opt = optim.Adam(self.h_net.parameters(),
                                 lr=args.sin_lr,
                                 weight_decay=args.sin_wd)
        # e(x) -> h(T)
        self.e_net = MLP(
                dim_input = args.sin_hidden,
                dim_hidden = args.sin_hidden,
                dim_output = args.label_dim,
                num_layers = args.sin_layer-1,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'none').to(device)
        self.e_net_opt = optim.Adam(self.e_net.parameters(),
                                 lr=args.sin_lr,
                                 weight_decay=args.sin_wd)
        # g(x)
        self.g_net = MLP(
                dim_input = args.sin_hidden,
                dim_hidden = args.sin_hidden,
                dim_output = args.label_dim,
                num_layers = args.sin_layer,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'none').to(device)
        self.g_net_opt = optim.Adam(self.g_net.parameters(),
                                 lr=args.sin_lr,
                                 weight_decay=args.sin_wd)

        # train_nodes, valid_nodes, test_nodes = splits[0][0], splits[0][1], splits[0][2]
        # m_dataset_train = M_Dataset(features[train_nodes], labels[train_nodes])
        # self.m_train_loader = DataLoader(m_dataset_train, batch_size=32, shuffle=True)
        #
        # m_dataset_valid = M_Dataset(features[valid_nodes], labels[valid_nodes])
        # self.m_valid_loader = DataLoader(m_dataset_valid, batch_size=1, shuffle=False)
        #
        # m_dataset_test = M_Dataset(features[test_nodes], labels[test_nodes])
        # self.m_test_loader = DataLoader(m_dataset_test, batch_size=1, shuffle=False)

        g_dataset_train = G_Dataset(features[train_nodes], neighbors[train_nodes], labels[train_nodes])
        self.g_train_loader = DataLoader(g_dataset_train, batch_size=train_nodes.size()[0], shuffle=True)

        g_dataset_valid = G_Dataset(features[valid_nodes], neighbors[valid_nodes], labels[valid_nodes])
        self.g_valid_loader = DataLoader(g_dataset_valid, batch_size=valid_nodes.size()[0], shuffle=False)

        # g_dataset_test = G_Dataset(features[test_nodes], neighbors[test_nodes], labels[test_nodes])
        # self.g_test_loader = DataLoader(g_dataset_test, batch_size=1, shuffle=False)

    def m_loss(self, prediction, target):
        if self.out_dim == 1: c = 2
        else: c = self.out_dim
        train_samples_per_class = [0 for i in range(c)]
        for cl in range(c):
            # print ((train_datasets[i].label[train_datasets[i].train_idx]))
            train_samples_per_class[cl] += (target== cl).nonzero().size()[0]
        train_samples_per_class = 1/torch.tensor(train_samples_per_class)
        weight = (train_samples_per_class/train_samples_per_class.sum()*c).to(self.device)
        if self.out_dim == 1:
            return nn.BCELoss(weight=weight[1])(input=prediction, target=target.to(dtype=torch.float))
        return nn.NLLLoss(weight=weight)(input=prediction, target=target)

    def global_loss(self, prediction, target):
        if self.out_dim == 1: c = 2
        else: c = self.out_dim
        train_samples_per_class = [0 for i in range(c)]
        for cl in range(c):
            # print ((train_datasets[i].label[train_datasets[i].train
            train_samples_per_class[cl] += (target== cl).nonzero().size()[0]
        train_samples_per_class = 1/torch.tensor(train_samples_per_class)
        weight = (train_samples_per_class/train_samples_per_class.sum()*c).to(self.device)
        if self.out_dim  == 1:
            return nn.BCELoss(weight=weight[1])(input=prediction, target=target.to(dtype=torch.float))
        return nn.NLLLoss(weight=weight)(input=prediction, target=target)

    def e_loss(self, prediction, target):
        return F.mse_loss(input=prediction, target=target)

    def train_m_model(self, args, patience = 10, max_epochs = 100):
    #     com_early_stopping = EarlyStoppingCriterion(patience=patience, mode="min")
        best_loss = float('inf')
        counter = 0
        curr_time = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        best_model_path = f'pretrained/{curr_time}.pt'
        for epoch in range(1, max_epochs + 1):
            for batch_idx, batch in enumerate(self.g_train_loader):
                covariates, target_outcome = batch[0].to(self.device), batch[1].to(self.device)
                # print (covariates.size())
                # print (target_outcome.size())
                if covariates.size()[0] <= 1:
                    continue
                covariates = self.emb(covariates)
                # if len(target_outcome.size())>1 :
                #     target_outcome = torch.argmax(target_outcome, 1)
                if len(target_outcome.size()) > 1 and target_outcome.size()[1] == 1:
                    target_outcome = target_outcome.squeeze(1)
                if len(target_outcome.size()) > 1 and target_outcome.size()[1] > 1:
                    target_outcome = torch.argmax(target_outcome, 1)
                # else:
                #     target_outcome = F.one_hot(target_outcome, num_classes=args.label_dim).float()
                self.m_net_opt.zero_grad()
                self.emb_opt.zero_grad()
                prediction = self.activation(self.m_net(covariates))
                if self.out_dim == 1:
                    prediction = prediction.squeeze(1)
                loss = self.m_loss(prediction=prediction, target=target_outcome)
                loss.backward()
                # acc, bacc = accuracy(prediction, target_outcome)
                self.emb_opt.step()
                self.m_net_opt.step()
                if batch_idx % 10 == 0:
                    loss = loss.item()
                    # logging.warning({"epoch": epoch, "train_loss": loss, "acc": acc})

            if epoch % 1 == 0:
                self.m_net.eval()
                self.emb.eval()
                val_error = 0.0
                # val_acc = 0.
                for batch_idx, batch in enumerate(self.g_valid_loader):
                    covariates, target_outcome = batch[0].to(self.device), batch[1].to(self.device)
                    covariates = self.emb(covariates)
                    # target = F.one_hot(target_outcome, num_classes=args.label_dim).float()
                    # if len(target_outcome.size())>1:
                    #     target_outcome = torch.argmax(target_outcome, 1)
                    if len(target_outcome.size()) > 1 and target_outcome.size()[1] == 1:
                        target_outcome = target_outcome.squeeze(1)
                    if len(target_outcome.size()) > 1 and target_outcome.size()[1] > 1:
                        target_outcome = torch.argmax(target_outcome, 1)
                    prediction = self.activation(self.m_net(covariates))
                    if self.out_dim == 1:
                        prediction = prediction.squeeze(1)
                    val_error += self.m_loss(prediction=prediction, target=target_outcome).item()
                    # acc, bacc = accuracy(prediction, target_outcome)
                    # val_acc += acc
                val_error /= len(self.g_valid_loader.dataset)
                # val_acc /= len(self.g_valid_loader.dataset)
                if val_error < best_loss:

                    counter = patience
                    best_loss = val_error
                    torch.save(self.state_dict(), best_model_path)
                    # logging.warning(f"epoch {epoch} valid loss: {val_error:.6f}, valid acc: {val_acc:.6f}")
                else: counter -= 1
                if counter == 0: break
                self.m_net.train()
                self.emb.train()
        self.load_state_dict(torch.load(best_model_path))

    # def forward(self, batch_emb, batch_label, batch_neighbor, args):
    #
    #     h = self.h_net(batch_neighbor).reshape(-1, args.label_dim, 1, args.sin_hidden)
    #     g = self.g_net(batch_emb).reshape(-1, args.label_dim, args.sin_hidden, 1)
    #     e = self.e_net(batch_emb).reshape(-1, args.label_dim, 1, args.sin_hidden)
    #     m = self.m_net(batch_emb).detach()
    #
    #     diff = h - e
    #     Y = torch.matmul(diff, g).squeeze() + m
    #     # Y = (diff * g).sum(-1).view(-1) + m
    #     # return nn.LogSoftmax(dim=1)(Y)
    #     return nn.Sigmoid()(Y)


    def forward(self, batch_emb, batch_label, batch_neighbor, args, act=True):

        h = self.h_net(batch_neighbor)
        g = self.g_net(batch_emb)
        e = self.e_net(batch_emb)
        m = self.m_net(batch_emb).detach()
        # print (h.size())
        # print (g.size())
        # print (e.size())
        # print (m.size())
        # diff = h - e
        Y = g + h - g*e + m
        # Y = (diff * g).sum(-1).view(-1) + m
        if not act:
            return Y
        return self.activation(Y)


    def train_step(self, args):
        self.train()
        for batch_idx, batch in enumerate(self.g_train_loader):
            batch_feature, batch_label, batch_neighbor = batch[0], batch[1], batch[2]
            if batch_feature.size()[0] <= 1:
                continue
            # if len(batch_label.size()) == 1:
            #     batch_label_vec = F.one_hot(batch_label, num_classes=args.label_dim).float()
            # if len(batch_label.size()) > 1:
            #     batch_label = torch.argmax(batch_label, 1)
            if len(batch_label.size()) > 1 and batch_label.size()[1] == 1:
                batch_label = batch_label.squeeze(1)
            if len(batch_label.size()) > 1 and batch_label.size()[1] > 1:
                batch_label = torch.argmax(batch_label, 1)
            batch_emb = self.emb(batch_feature)
            for _ in range(args.num_steps_global):
                self.g_net_opt.zero_grad()
                self.h_net_opt.zero_grad()
                self.emb_opt.zero_grad()
                global_prediction = self.forward(batch_emb, batch_label, batch_neighbor, args)
                if self.out_dim == 1:
                    global_prediction = global_prediction.squeeze(1)
                global_loss = self.global_loss(global_prediction, batch_label)
                global_loss.backward()
                self.emb_opt.zero_grad()
                self.g_net_opt.step()
                self.h_net_opt.step()

            for _ in range(args.num_steps_propensity):
                h = self.h_net(batch_neighbor).detach()
                self.e_net_opt.zero_grad()
                e_features = self.e_net(batch_emb.detach())
                e_loss = self.e_loss(e_features, h)
                e_loss.backward()
                self.e_net_opt.step()

    def valid_step(self, args, loader):
        loss = 0.0
        val_acc = 0.0
        self.eval()
        for batch_idx, batch in enumerate(loader):
            batch_feature, batch_label, batch_neighbor = batch[0], batch[1], batch[2]
            # batch_label_vec = F.one_hot(batch_label, num_classes=args.label_dim).float()
            # if len(batch_label.size()) > 1:
            #     batch_label = torch.argmax(batch_label, 1)
            if len(batch_label.size()) > 1 and batch_label.size()[1] == 1:
                batch_label = batch_label.squeeze(1)
            if len(batch_label.size()) > 1 and batch_label.size()[1] > 1:
                batch_label = torch.argmax(batch_label, 1)
            batch_emb = self.emb(batch_feature)
            batch_size = batch_feature.size()[0]
            prediction = self.forward(batch_emb, batch_label, batch_neighbor, args)
            if self.out_dim == 1:
                prediction = prediction.squeeze(1)
            # print (prediction.size())
            # print (batch_label.size())
            loss += self.global_loss(prediction, batch_label)*batch_size
            # acc = accuracy(prediction, batch_label)
            acc, bacc, _, _ = accuracy(prediction, batch_label)
            val_acc += acc

        loss /= len(loader.dataset)
        val_acc /= len(loader.dataset)
        return loss, val_acc

    def train_and_test(self, args, patience = 10, max_epochs = 50):
        counter = patience
        curr_time = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        best_model_path = f'pretrained/{curr_time}.pt'
        best_loss = float('inf')
        for epoch in range(500):
            self.train_step(args)
            # train_loss, train_acc = self.valid_step(args,self.g_train_loader)
            valid_loss, valid_acc = self.valid_step(args,self.g_valid_loader)
            # logging.warning(f"epoch {epoch} train loss: {train_loss:.6f}, train acc: {train_acc:.6f}")
            # logging.warning(f"epoch {epoch} valid loss: {valid_loss:.6f}, valid acc: {valid_acc:.6f}")
            if valid_loss < best_loss:
                counter = patience
                best_loss = valid_loss
                torch.save(self.state_dict(), best_model_path)
            # else: counter -= 1
            if counter == 0: break
        self.load_state_dict(torch.load(best_model_path))

    def get_ite(self, args, neighbors, features, labels, num_counterfactual=10):
        n = features.size()[0]
        data = (features, labels, neighbors)
        emb = self.emb(features)
        y = torch.exp(self.forward(emb, labels, neighbors, args, act=(self.out_dim>1)))
        # y = self.forward(emb, labels, neighbors, args, act=(self.out_dim>1))
        # return y
        y_counter = torch.zeros(n, args.label_dim).to(self.device)
        for i in range(num_counterfactual):
            node_idxs = [i for i in range(n)]
            random.shuffle(node_idxs)
            shuffled_neighbors = neighbors[torch.tensor(node_idxs)]
            y_counter = y_counter + torch.exp(self.forward(emb, labels, shuffled_neighbors, args, act=(self.out_dim>1)))
            # y_counter = y_counter + self.forward(emb, labels, shuffled_neighbors, args, act=(self.out_dim>1))
        y_counter = y_counter/num_counterfactual
        ites = y-y_counter
        if self.out_dim == 1:
            ites = ites.squeeze(1)
        return ites

    def get_ite_test(self, args, neighbors, features, labels, num_counterfactual=10):
        n = features.size()[0]
        emb = self.emb(features)
        y = torch.exp(self.forward(emb, labels, neighbors, args, act=(self.out_dim>1)))
        # y = self.forward(emb, labels, neighbors, args, act=(self.out_dim>1))
        # return y
        y_counter = torch.zeros(n, args.label_dim).to(self.device)
        for i in range(num_counterfactual):
            node_idxs = [i for i in range(n)]
            random.shuffle(node_idxs)
            shuffled_neighbors = neighbors[torch.tensor(node_idxs)]
            y_counter = y_counter + torch.exp(self.forward(emb, labels, shuffled_neighbors, args, act=(self.out_dim>1)))
            # y_counter = y_counter + self.forward(emb, labels, shuffled_neighbors, args, act=(self.out_dim>1))
        y_counter = y_counter/num_counterfactual
        ites = y-y_counter
        if self.out_dim == 1:
            ites = ites.squeeze(1)
        if torch.isnan(ites).sum()> 0:
            print ('NaN!')
            nan_idx = torch.isnan(ites).nonzero()
            ites[nan_idx] = 0
            print ('still NaN?', torch.isnan(ites).sum()> 0)
        return ites

class SIN_r(nn.Module):
    def __init__(self, args, features, neighbors, labels, train_nodes, valid_nodes):
        super().__init__()
        self.device = args.device
        self.features = features
        self.labels = labels
        self.neighbors = neighbors
        # self.g = g
        if args.label_dim > 1:
            self.activation = nn.LogSoftmax(dim=1)
        else: self.activation = nn.Sigmoid()
        self.out_dim = args.label_dim

        self.emb = MLP(
                dim_input = args.feature_dim,
                dim_hidden = args.sin_hidden,
                dim_output = args.sin_hidden,
                num_layers = 1,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'relu').to(self.device)
        self.emb_opt = optim.Adam(self.emb.parameters(),
                                         lr=args.sin_lr,
                                         weight_decay=args.sin_wd)


        if args.gnn == 'H2GCN':
            gnn_hidden = args.gnn_hidden * (2**(args.num_layers+1)-1)
        else: gnn_hidden = args.gnn_hidden

        # m(x) -> y
        self.m_net = MLP(
                dim_input = gnn_hidden,
                dim_hidden = args.sin_hidden,
                dim_output = args.label_dim,
                num_layers = 2,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'none').to(self.device)
        self.m_net_opt = optim.Adam(self.m_net.parameters(),
                                         lr=args.sin_lr,
                                         weight_decay=args.sin_wd)

        # h(T)
        self.h_net = MLP(
                dim_input = args.sin_hidden,
                dim_hidden = args.sin_hidden,
                num_layers = args.sin_layer-1,
                dim_output = args.label_dim,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'none').to(device)

        self.h_net_opt = optim.Adam(self.h_net.parameters(),
                                 lr=args.sin_lr,
                                 weight_decay=args.sin_wd)
        # e(x) -> h(T)
        self.e_net = MLP(
                dim_input = gnn_hidden,
                dim_hidden = args.sin_hidden,
                dim_output = args.label_dim,
                num_layers = args.sin_layer-1,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'none').to(device)
        self.e_net_opt = optim.Adam(self.e_net.parameters(),
                                 lr=args.sin_lr,
                                 weight_decay=args.sin_wd)
        # g(x)
        self.g_net = MLP(
                dim_input = gnn_hidden,
                dim_hidden = args.sin_hidden,
                dim_output = args.label_dim,
                num_layers = args.sin_layer,
                batch_norm = True,
        #         initialiser: str,
                dropout = args.sin_dropout,
                activation = 'relu',
                leaky_relu = 0,
                output_activation = 'none').to(device)
        self.g_net_opt = optim.Adam(self.g_net.parameters(),
                                 lr=args.sin_lr,
                                 weight_decay=args.sin_wd)

        # m_dataset_train = M_Dataset(features[train_nodes], labels[train_nodes])
        # self.m_train_loader = DataLoader(m_dataset_train, batch_size=32, shuffle=True)
        #
        # m_dataset_valid = M_Dataset(features[valid_nodes], labels[valid_nodes])
        # self.m_valid_loader = DataLoader(m_dataset_valid, batch_size=1, shuffle=False)
        #
        # m_dataset_test = M_Dataset(features[test_nodes], labels[test_nodes])
        # self.m_test_loader = DataLoader(m_dataset_test, batch_size=1, shuffle=False)

        # train_nodes, valid_nodes, test_nodes = splits[0][0], splits[0][1], splits[0][2]

        g_dataset_train = G_Dataset(features[train_nodes], neighbors[train_nodes], labels[train_nodes])
        self.g_train_loader = DataLoader(g_dataset_train, batch_size=train_nodes.size()[0], shuffle=True)

        g_dataset_valid = G_Dataset(features[valid_nodes], neighbors[valid_nodes], labels[valid_nodes])
        self.g_valid_loader = DataLoader(g_dataset_valid, batch_size=valid_nodes.size()[0], shuffle=False)

        # g_dataset_test = G_Dataset(features[test_nodes], neighbors[test_nodes], labels[test_nodes])
        # self.g_test_loader = DataLoader(g_dataset_test, batch_size=1, shuffle=False)

    def m_loss(self, prediction, target):
        if self.out_dim == 1: c = 2
        else: c = self.out_dim
        train_samples_per_class = [0 for i in range(c)]
        for cl in range(c):
            # print ((train_datasets[i].label[train_datasets[i].train_idx]))
            train_samples_per_class[cl] += (target== cl).nonzero().size()[0]
        train_samples_per_class = 1/torch.tensor(train_samples_per_class)
        weight = (train_samples_per_class/train_samples_per_class.sum()*c).to(self.device)
        if self.out_dim  == 1:
            return nn.BCELoss(weight=weight[1])(input=prediction, target=target.to(dtype=torch.float))
        return nn.NLLLoss(weight=weight)(input=prediction, target=target)

    def global_loss(self, prediction, target):
        if self.out_dim == 1: c = 2
        else: c = self.out_dim
        train_samples_per_class = [0 for i in range(c)]
        for cl in range(c):
            # print ((train_datasets[i].label[train_datasets[i].train_idx]))
            train_samples_per_class[cl] += (target== cl).nonzero().size()[0]
        train_samples_per_class = 1/torch.tensor(train_samples_per_class)
        weight = (train_samples_per_class/train_samples_per_class.sum()*c).to(self.device)
        if self.out_dim == 1:
            return nn.BCELoss(weight=weight[1])(input=prediction, target=target.to(dtype=torch.float))
        # target = target.squeeze(-1)
        return nn.NLLLoss(weight=weight)(input=prediction, target=target)
        # return nn.BCELoss()(input=prediction, target=target)

    def e_loss(self, prediction, target):
        return F.mse_loss(input=prediction, target=target)

    def train_m_model( self, args, patience = 10, max_epochs = 100):
    #     com_early_stopping = EarlyStoppingCriterion(patience=patience, mode="min")
        best_loss = float('inf')
        counter = 0
        curr_time = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        best_model_path = f'pretrained/{curr_time}.pt'
        for epoch in range(1, max_epochs + 1):
            for batch_idx, batch in enumerate(self.g_train_loader):
                covariates, target_outcome = batch[2].to(self.device), batch[1].to(self.device)
                if covariates.size()[0] <= 1:
                    continue
                # covariates = self.emb(covariates)
                # target = F.one_hot(target_outcome, num_classes=args.label_dim).float()
                # if len(target_outcome.size())>1:
                #     target_outcome = torch.argmax(target_outcome, 1)
                if len(target_outcome.size()) > 1 and target_outcome.size()[1] == 1:
                    target_outcome = target_outcome.squeeze(1)
                if len(target_outcome.size()) > 1 and target_outcome.size()[1] > 1:
                    target_outcome = torch.argmax(target_outcome, 1)
                self.m_net_opt.zero_grad()
                self.emb_opt.zero_grad()
                # prediction = nn.Sigmoid()(self.m_net(covariates))
                # loss = self.m_loss(prediction=prediction, target=target)
                prediction = self.activation(self.m_net(covariates))
                if self.out_dim == 1:
                    prediction = prediction.squeeze(1)
                loss = self.m_loss(prediction=prediction, target=target_outcome)
                loss.backward()
                # acc, bacc = accuracy(prediction, target_outcome)
                self.emb_opt.step()
                self.m_net_opt.step()
                if batch_idx % 10 == 0:
                    loss = loss.item()
                    # logging.warning({"epoch": epoch, "train_loss": loss, "acc": acc})

            if epoch % 1 == 0:
                self.m_net.eval()
                self.emb.eval()
                val_error = 0.0
                val_acc = 0.
                for batch_idx, batch in enumerate(self.g_valid_loader):
                    covariates, target_outcome = batch[2].to(self.device), batch[1].to(self.device)
                    # covariates = self.emb(covariates)
                    # target = F.one_hot(target_outcome, num_classes=args.label_dim).float()
                    # if len(target_outcome.size())>1:
                    #     target_outcome = torch.argmax(target_outcome, 1)
                    if len(target_outcome.size()) > 1 and target_outcome.size()[1] == 1:
                        target_outcome = target_outcome.squeeze(1)
                    if len(target_outcome.size()) > 1 and target_outcome.size()[1] > 1:
                        target_outcome = torch.argmax(target_outcome, 1)
                    # prediction = nn.Sigmoid()(self.m_net(covariates))
                    # val_error += self.m_loss(prediction, target).item()
                    prediction = self.activation(self.m_net(covariates))
                    if self.out_dim == 1:
                        prediction = prediction.squeeze(1)
                    val_error += self.m_loss(prediction=prediction, target=target_outcome).item()
                    # acc = accuracy(prediction, target_outcome)
                    acc, bacc, _, _ = accuracy(prediction, target_outcome)
                    val_acc += acc
                val_error /= len(self.g_valid_loader.dataset)
                val_acc /= len(self.g_valid_loader.dataset)

                if val_error < best_loss:
                    counter = patience
                    best_loss = val_error
                    torch.save(self.state_dict(), best_model_path)
                    # logging.warning(f"epoch {epoch} valid loss: {val_error:.6f}, valid acc: {val_acc:.6f}")
                else: counter -= 1
                if counter == 0: break
                self.m_net.train()
                self.emb.train()
        self.load_state_dict(torch.load(best_model_path))

    # def forward(self, batch_emb, batch_label, batch_neighbor, args):
    #
    #     h = self.h_net(batch_neighbor).reshape(-1, args.label_dim, 1, args.sin_hidden)
    #     g = self.g_net(batch_emb).reshape(-1, args.label_dim, args.sin_hidden, 1)
    #     e = self.e_net(batch_emb).reshape(-1, args.label_dim, 1, args.sin_hidden)
    #     m = self.m_net(batch_emb).detach()
    #
    #     diff = h - e
    #     Y = torch.matmul(diff, g).squeeze() + m
    #     # Y = (diff * g).sum(-1).view(-1) + m
    #     # return nn.LogSoftmax(dim=1)(Y)
    #     return nn.Sigmoid()(Y)


    def forward(self, batch_emb, batch_label, batch_neighbor, args, act=True):

        h = self.h_net(batch_emb)
        g = self.g_net(batch_neighbor)
        e = self.e_net(batch_neighbor)
        m = self.m_net(batch_neighbor).detach()
        # print (h.size())
        # print (g.size())
        # print (e.size())
        # print (m.size())
        # diff = h - e
        Y = g + h - g*e + m
        # Y = (diff * g).sum(-1).view(-1) + m
        if not act:
            return Y
        return self.activation(Y)


    def train_step(self, args):
        self.train()
        for batch_idx, batch in enumerate(self.g_train_loader):
            batch_feature, batch_label, batch_neighbor = batch[0], batch[1], batch[2]
            if batch_feature.size()[0] <= 1:
                continue
            # batch_label_vec = F.one_hot(batch_label, num_classes=args.label_dim).float()
            # if len(batch_label.size()) > 1:
            #     batch_label = torch.argmax(batch_label, 1)
            if len(batch_label.size()) > 1 and batch_label.size()[1] == 1:
                batch_label = batch_label.squeeze(1)
            if len(batch_label.size()) > 1 and batch_label.size()[1] > 1:
                batch_label = torch.argmax(batch_label, 1)
            batch_emb = self.emb(batch_feature)
            for _ in range(args.num_steps_global):
                self.g_net_opt.zero_grad()
                self.h_net_opt.zero_grad()
                self.emb_opt.zero_grad()
                global_prediction = self.forward(batch_emb, batch_label, batch_neighbor, args)
                if self.out_dim == 1:
                    global_prediction = global_prediction.squeeze(1)
                global_loss = self.global_loss(global_prediction, batch_label)
                global_loss.backward()
                self.emb_opt.zero_grad()
                self.g_net_opt.step()
                self.h_net_opt.step()

            for _ in range(args.num_steps_propensity):
                h = self.h_net(batch_emb).detach()
                self.e_net_opt.zero_grad()
                e_features = self.e_net(batch_neighbor.detach())
                e_loss = self.e_loss(e_features, h)
                e_loss.backward()
                self.e_net_opt.step()

    def valid_step(self, args, loader):
        loss = 0.0
        val_acc = 0.0
        self.eval()
        for batch_idx, batch in enumerate(loader):
            batch_feature, batch_label, batch_neighbor = batch[0], batch[1], batch[2]
            # batch_label_vec = F.one_hot(batch_label, num_classes=args.label_dim).float()
            # if len(batch_label.size()) > 1:
            #     batch_label = torch.argmax(batch_label, 1)
            if len(batch_label.size()) > 1 and batch_label.size()[1] == 1:
                batch_label = batch_label.squeeze(1)
            if len(batch_label.size()) > 1 and batch_label.size()[1] > 1:
                batch_label = torch.argmax(batch_label, 1)
            batch_emb = self.emb(batch_feature)
            batch_size = batch_feature.size()[0]
            prediction = self.forward(batch_emb, batch_label, batch_neighbor, args)
            if self.out_dim == 1:
                prediction = prediction.squeeze(1)
            loss += self.global_loss(prediction, batch_label)*batch_size
            # acc = accuracy(prediction, batch_label)
            acc, bacc, _, _ = accuracy(prediction, batch_label)
            val_acc += acc
        loss /= len(loader.dataset)
        val_acc /= len(loader.dataset)
        return loss, val_acc

    def train_and_test(self, args, patience = 10, max_epochs = 50):
        counter = patience
        curr_time = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        best_model_path = f'pretrained/{curr_time}.pt'
        best_loss = float('inf')
        for epoch in range(500):
            self.train_step(args)
            train_loss, train_acc = self.valid_step(args,self.g_train_loader)
            valid_loss, valid_acc = self.valid_step(args,self.g_valid_loader)
            # logging.warning(f"epoch {epoch} train loss: {train_loss:.6f}, train acc: {train_acc:.6f}")
            # logging.warning(f"epoch {epoch} valid loss: {valid_loss:.6f}, valid acc: {valid_acc:.6f}")
            if valid_loss < best_loss:
                counter = patience
                best_loss = valid_loss
                torch.save(self.state_dict(), best_model_path)
            # else: counter -= 1
            if counter == 0: break
        self.load_state_dict(torch.load(best_model_path))

    def get_ite(self, args, neighbors, features, labels, num_counterfactual=100):
        n = features.size()[0]
        emb = self.emb(features)
        y = torch.exp(self.forward(emb, labels, neighbors, args, act=(self.out_dim>1)))
        # y = self.forward(emb, labels, neighbors, args, act=(self.out_dim>1))
        # return y
        y_counter = torch.zeros(n, args.label_dim).to(self.device)
        for i in range(num_counterfactual):
            node_idxs = [i for i in range(n)]
            random.shuffle(node_idxs)
            shuffled_emb = emb[torch.tensor(node_idxs)]
            y_counter = y_counter + torch.exp(self.forward(shuffled_emb, labels, neighbors, args, act=(self.out_dim>1)))
            # y_counter = y_counter + self.forward(shuffled_emb, labels, neighbors, args, act=(self.out_dim>1))
        y_counter = y_counter/num_counterfactual
        ites = y-y_counter
        if self.out_dim == 1:
            ites = ites.squeeze(1)
        return ites

    def get_ite_test(self, args, neighbors, features, labels, num_counterfactual=100):
        n = features.size()[0]
        emb = self.emb(features)
        y = torch.exp(self.forward(emb, labels, neighbors, args, act=(self.out_dim>1)))
        # y = self.forward(emb, labels, neighbors, args, act=(self.out_dim>1))
        # return y
        y_counter = torch.zeros(n, args.label_dim).to(self.device)
        for i in range(num_counterfactual):
            node_idxs = [i for i in range(n)]
            random.shuffle(node_idxs)
            shuffled_emb = emb[torch.tensor(node_idxs)]
            y_counter = y_counter + torch.exp(self.forward(shuffled_emb, labels, neighbors, args, act=(self.out_dim>1)))
            # y_counter = y_counter + self.forward(shuffled_emb, labels, neighbors, args, act=(self.out_dim>1))
        y_counter = y_counter/num_counterfactual
        ites = y-y_counter
        if self.out_dim == 1:
            ites = ites.squeeze(1)
        if torch.isnan(ites).sum()> 0:
            print ('NaN!')
            nan_idx = torch.isnan(ites).nonzero()
            ites[nan_idx] = 0
            print ('still NaN?', torch.isnan(ites).sum()> 0)
        return ites
