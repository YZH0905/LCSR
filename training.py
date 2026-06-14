import numpy as np
from sklearn import cluster

from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader

import torch.nn as nn
from learner.contrastive_utils import PairConLoss, Attention_loss

import matplotlib.pyplot as plt
from collections import Counter
from plabel_allocator import *
import pandas as pd
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
import os

class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer,  train_loader, semi_train_loader, semi_labels, semi_indexs, args, scheduler=None):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.semi_train_loader = semi_train_loader
        self.semi_labels = semi_labels
        self.semi_indexs = semi_indexs
        self.args = args
        self.eta = self.args.eta
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrast_loss = PairConLoss(temperature=self.args.temperature, m=self.args.m)
        self.attention_loss = Attention_loss(temperature=self.args.temperature)

        N = len(self.train_loader.dataset)
        self.a = torch.full((N, 1), 1/N).squeeze()

        self.b = torch.ones((self.args.num_classes,), dtype=torch.float64).to('cuda') / self.args.num_classes

        self.u = None
        self.v = None
        self.h = torch.FloatTensor([1])
        self.allb = [[self.b[i].item()] for i in range(self.args.classes)]
        self.label_ratios = torch.zeros(8)

        self.empty_tensor = torch.zeros(len(self.train_loader.dataset), dtype=torch.long)
        self.semi_empty_tensor = torch.zeros(len(self.semi_train_loader.dataset), dtype=torch.long)  # 或者根据需要使用 float
        self.semi_empty_index = torch.zeros(len(self.semi_train_loader.dataset), dtype=torch.long)  # 使用 long 数据类型

        self.prototypes = torch.zeros(self.args.classes, 128).to('cuda')
        self.prototypes_accum = torch.zeros(self.args.classes, 128).to('cuda')
        self.prototypes_counter = torch.zeros(self.args.classes, dtype=torch.long).to('cuda')
        self.max_p = torch.zeros(self.args.batch_size, dtype=torch.long).to('cuda')

        print(self.empty_tensor.shape)
        print(self.semi_empty_tensor.shape)
        
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")

    def soft_ce_loss(self, pred, target, step):
        tmp = target ** 2 / torch.sum(target, dim=0)
        target = tmp / torch.sum(tmp, dim=1, keepdim=True)
        return torch.mean(-torch.sum(target * (F.log_softmax(pred, dim=1)), dim=1))

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.args.max_length,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat

    def prepare_transformer_input(self, batch):
        text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
        feat1 = self.get_batch_token(text1)
        if self.args.augtype == 'explicit':
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
        else:
            input_ids = feat1['input_ids'] 
            attention_mask = feat1['attention_mask']
            
        return input_ids.cuda(), attention_mask.cuda()



    def accumulate_prototypes(self, feature_matrix, labels):
        j = 0
        for i in labels:
            self.prototypes_accum[i] += feature_matrix[j]
            self.prototypes_counter[i] += 1
            j += 1


    def update_prototypes(self):
        self.prototypes = F.normalize(self.prototypes_accum / self.prototypes_counter.unsqueeze(1), dim=1)
        # reset accumulator and counter
        self.prototypes_accum = 0 * self.prototypes_accum
        self.prototypes_counter = 0 * self.prototypes_counter

    def pro_contrastive_loss(self, features, labels):
        # 初始化损失值
        loss = torch.tensor(0.0, device=features.device)
        # 获取类别数
        num_classes = self.prototypes.shape[0]

        # 遍历每一个类别
        for c in range(num_classes):
            # 找到当前类别的样本索引
            indices = (labels == c).nonzero(as_tuple=True)[0]
            if len(indices) == 0:  # 如果该类别没有样本，跳过
                continue

            # 获取当前类别的样本特征
            class_features = features[indices]
            all_prototypes = F.normalize(self.prototypes, dim=1)
            # 获取当前类别的簇心
            prototype = all_prototypes[c].unsqueeze(0)

            # # 计算高斯核函数
            # sigma = 1.0
            # distance_squared = torch.sum((class_features - prototype) ** 2, dim=1)  # (200,)
            # gaussian_similarity = torch.exp(-distance_squared / (2 * sigma ** 2))  # (200,)

            t = 1   # temperature
            sim_p = torch.matmul(class_features, prototype.T) / t
            sim_n = torch.matmul(class_features, all_prototypes.T) / t
            numerator = torch.exp(sim_p.squeeze(1))
            denominator = torch.exp(sim_n).sum(dim=1, keepdim=True).squeeze(1)
            # per_loss = torch.log((numerator * gaussian_similarity) / denominator)
            per_loss = torch.log(numerator / denominator)
            loss += 0 - per_loss.sum()

            # distances = torch.norm(class_features - prototype, dim=1)
            # loss += distances.sum()

        # 计算平均损失
        loss = loss / features.size(0)

        return loss


    def loss_function(self, input_ids, attention_mask, selected, i, semi_batch):
        embd0, embd2, embd3 = self.model.get_embeddings(input_ids, attention_mask, task_type=self.args.augtype)

        feat0 = self.model.contrast_logits(embd0)
        feat2, feat3 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat2, feat3)
        loss = losses["loss"]
        losses['contrast'] = losses["loss"]
        self.args.tensorboard.add_scalar('loss/contrast_loss', losses['loss'].item(), global_step=i)

        sim_input_ids, sim_attention_mask = self.prepare_transformer_input(semi_batch)
        semi_label = semi_batch['label']
        sim_embd0, sim_embd1, sim_embd2 = self.model.get_embeddings(sim_input_ids, sim_attention_mask, task_type=self.args.augtype)
        P1 = self.model(sim_embd1)
        P2 = self.model(sim_embd2)
        semi_label = semi_label - 1
        semi_label = semi_label.to('cuda')
        semi_loss = self.ce_loss(P1, semi_label) + self.ce_loss(P2, semi_label)
        loss += semi_loss

        if i >= self.args.pre_step + 1:
            P2 = self.model(embd2)
            P3 = self.model(embd3)  # predicted labels before softmax
            target_label = None
            if self.args.soft == True:
                target = self.L.cuda()
                cluster_loss = self.soft_ce_loss(P2, target, i) + self.soft_ce_loss(P3, target, i)
            else:
                target_label = self.L.squeeze(0).cuda()

            if target_label != None:
                cluster_loss = self.ce_loss(P2, target_label) + self.ce_loss(P3, target_label)
            loss += cluster_loss

        if i >= self.args.second_stage + 1:
            semi_inst = self.model.contrast_logits(sim_embd0).to('cuda')
            self.accumulate_prototypes(semi_inst.detach(), semi_label.detach())
            semi_prototypes = self.prototypes_accum / (self.prototypes_counter.unsqueeze(1) + 1e-5)
            self.prototypes_accum = 0 * self.prototypes_accum
            self.prototypes_counter = 0 * self.prototypes_counter
            self.prototypes = self.prototypes * 0.95 + semi_prototypes * 0.05
            pc_loss = (self.pro_contrastive_loss(feat2, target_label) + self.pro_contrastive_loss(feat3, target_label)) / 2
            loss += pc_loss


        losses['loss'] = loss
        self.args.tensorboard.add_scalar('loss/loss', loss, global_step=i)
        return loss, losses

    def train_step_explicit(self, input_ids, attention_mask, selected, i, semi_batch):
        if i >= self.args.pre_step:
            self.optimize_labels(i, input_ids, attention_mask)

        loss, losses = self.loss_function(input_ids, attention_mask, selected, i, semi_batch)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return losses



    def optimize_labels(self, step, input_ids, attention_mask):

        emb1, emb2, emb3 = self.model.get_embeddings(input_ids, attention_mask, task_type=self.args.augtype)  # embedding
        P = F.softmax(self.model(emb1), dim=1)
        feature = self.model.contrast_logits(emb1)
        PS = P.detach().cpu()

        a = torch.ones((PS.shape[0],), dtype=torch.float64).to('cuda') / PS.shape[0]
        pseudo_label, c_b, max_probs = curriculum_structure_aware_PL(self.args.method, a, self.b, PS, feature.detach().cpu(), lambda1=self.args.lambda1, lambda2=self.args.lambda2, lambda3=self.args.lambda3,
                                                                 version='fast',
                                                                 reg_e=0.1,
                                                                 reg_sparsity=None)
        self.b = c_b
        self.L = pseudo_label.unsqueeze(0)



    def train(self):
        self.optimize_times = ((np.linspace(self.args.start, 1, self.args.M)**2)[::-1] * self.args.max_iter).tolist()
        _ = self.evaluate_embedding(-1)
        for i in np.arange(self.args.start+1, self.args.max_iter+1):
            self.model.train()
            try:
                batch, selected = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch, selected = next(train_loader_iter)

            try:
                semi_batch, semi_index = next(semi_train_loader_iter)
                self.semi_empty_tensor[semi_index] = semi_batch['label']
                self.semi_empty_index[semi_index] = semi_batch['index']
            except:
                semi_train_loader_iter = iter(self.semi_train_loader)
                semi_batch, semi_index = next(semi_train_loader_iter)
                self.semi_empty_tensor[semi_index] = semi_batch['label']
                self.semi_empty_index[semi_index] = semi_batch['index']


            input_ids, attention_mask = self.prepare_transformer_input(batch)
            losses = self.train_step_explicit(input_ids, attention_mask, batch['index'], i, semi_batch)


            if (self.args.print_freq>0) and ((i % self.args.print_freq == 0) or (i == self.args.max_iter) or (i == self.args.second_stage)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                flag = self.evaluate_embedding(i)
                if flag < 0:
                    break
        return None


    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))

        self.model.eval()
        for i, (batch, index) in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label']
                feat = self.get_batch_token(text)
                embeddings = self.model.get_embeddings(feat['input_ids'].cuda(), feat['attention_mask'].cuda(),
                                                       task_type="evaluate")
                P = F.softmax(self.model(embeddings), dim=1)
                pred = torch.argmax(P, dim=1)
                feature = self.model.contrast_logits(embeddings)
                PS = P.detach().cpu()

                if i == 0:
                    all_probs = P
                    all_labels = label
                    all_pred = pred.detach()
                    all_embeddings = embeddings.detach()
                else:
                    all_probs = torch.cat((all_probs, P), dim=0)
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_pred = torch.cat((all_pred, pred.detach()), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)

                if step >= self.args.second_stage:
                    a = torch.ones((PS.shape[0],), dtype=torch.float64).to('cuda') / PS.shape[0]
                    pseudo_label, _, confidence = curriculum_structure_aware_PL(self.args.method, a, self.b, PS, feature.detach().cpu(),lambda1=self.args.lambda1,
                                                                   lambda2=self.args.lambda2, lambda3=self.args.lambda3, version='fast', reg_e=0.1,
                                                                   reg_sparsity=None)
                    if i == 0:
                        all_confidence = confidence
                        all_features = feature.detach()
                        all_pseudo_labels = pseudo_label.detach()
                    else:
                        all_confidence = torch.cat((all_confidence, confidence), dim=0)
                        all_features = torch.cat((all_features, feature.detach()), dim=0)
                        all_pseudo_labels = torch.cat((all_pseudo_labels, pseudo_label.detach()), dim=0)
        for j in range(len(self.semi_labels)):
            all_pred[self.semi_indexs[j]] = self.semi_labels[j] - 1

        # clustering accuracy
        clusters_num = len(set(all_pred.cpu().numpy()))
        self.args.tensorboard.add_scalar('Test/preded_clusters', clusters_num, step)
        pred_labels = all_pred.cpu()
        confusion1 = Confusion(max(self.args.num_classes, self.args.classes))
        confusion1.add(pred_labels, all_labels)
        true_label, predic_label = confusion1.optimal_assignment(self.args.num_classes)
        acc1 = confusion1.acc()
        clusterscores1 = confusion1.clusterscores(all_labels, pred_labels)

        ressave = {"acc": acc1}
        ressave.update(clusterscores1)
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        arr_MLP = Counter(np.array(pred_labels))

        stop_flag = 0
        y_pred = pred_labels.numpy()
        if step == -1:
            self.y_pred_last = np.copy(y_pred)
        else:
            change_rate = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            self.args.tensorboard.add_scalar('Test/change_rate', change_rate, step)
            self.y_pred_last = np.copy(y_pred)
            print('[Step] {} Label change rate: {:.3f} tol: {:.3f}'.format(step, change_rate, self.args.tol))
            if (step > self.args.pre_step and change_rate < self.args.tol) or step >= 4000:
                print('Reached tolerance threshold, stop training.')
                stop_flag = -1
        if stop_flag + 1 >= 0:
                print('preded classes number:', clusters_num)
                print('[Step]', step)
                print('[Model] Clustering scores:', clusterscores1)
                print('[Model] ACC: {:.4f}'.format(acc1))
                print('MLP Model：', len(arr_MLP), arr_MLP)

        return stop_flag






