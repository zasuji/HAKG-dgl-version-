import random

import torch
import torch.optim as optim
import dgl
import numpy as np

from time import time
from prettytable import PrettyTable
import pandas as pd
from tqdm import tqdm
import pickle
import gc

from utils.parser import *
from utils.load_data import *
from utils.helper import *
from modules.kgat import *
from utils.evalu import *

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
n_new_node = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = list()
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            each_negs = list()
            neg_item = np.random.randint(low=0, high=n_items, size=args.num_neg_sample)
            if len(set(neg_item) & set(train_user_set[user])) == 0:
                each_negs += list(neg_item)
            else:
                neg_item = list(set(neg_item) - set(train_user_set[user]))
                each_negs += neg_item
                while len(each_negs) < args.num_neg_sample:
                    n1 = np.random.randint(low=0, high=n_items, size=1)[0]
                    if n1 not in train_user_set[user]:
                        each_negs += [n1]
            neg_items.append(each_negs)

        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
    return feed_dict


def get_feed_dict_kg(train_kg_dict, start, end, n_entities, exist_hand_idx):
    def negative_sampling_kg(train_kg_dict, h, relation, n_entities):
        pos_triples = train_kg_dict[h]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == 1:
                break

            tail = np.random.randint(low=0, high=n_entities, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def positive_sampling_kg(train_kg_dict, h):
        pos_triples = train_kg_dict[h]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == 1:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_relations:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    batch_head = exist_hand_idx[start:end]

    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = positive_sampling_kg(train_kg_dict, h)
        batch_relation += relation
        batch_pos_tail += pos_tail

        neg_tail = negative_sampling_kg(train_kg_dict, h, relation[0], n_entities)
        batch_neg_tail += neg_tail

    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


if __name__ == '__main__':
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """读取参数"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id))

    """读取数据与构建图"""
    train_cf, test_cf, user_dict, train_kg_dict, n_params, graph, graph_UIS, graph_kgat_all, graph_kg, graph_i2u = load_data_both(args)
    if device != 'cpu':
        graph = graph.to(device)
        graph_UIS = graph_UIS.to(device)
        graph_kgat_all = graph_kgat_all.to(device)
        graph_kg = graph_kg.to(device)
        graph_i2u = graph_i2u.to(device)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    # user_id与item_id这里是为了evaluate这里
    user_ids = list(user_dict['test_user_set'].keys())
    user_ids_batches = [user_ids[i: i + args.batch_size] for i in range(0, len(user_ids), args.batch_size)]
    for i in range(len(user_ids_batches)):
        user_ids_batches[i] = [t + n_entities for t in user_ids_batches[i]]
    # 因为是在测试时需要所以我给user的序号都加上了n_entities
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if device != 'cpu':
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(n_items, dtype=torch.long)
    if device != 'cpu':
        item_ids = item_ids.to(device)

    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    # 预训练 #
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(
            load_pretrain_data(args.dataset, args.pretrain_embedding_dir, n_users, n_items, args.entity_dim)[0])
        item_pre_embed = torch.tensor(
            load_pretrain_data(args.dataset, args.pretrain_embedding_dir, n_users, n_items, args.entity_dim)[1])

    else:
        user_pre_embed, item_pre_embed = None, None

    """定义模型"""
    n_params['epoch_num_cf'] = len(train_cf) // args.batch_size + 1
    n_params['epoch_num_kg'] = len(train_kg_dict) // args.batch_size + 1

    model = KGAT(args, n_params, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 各参数的初始化
    best_epoch = -1
    epoch = 0
    cur_best = 0
    stopping_step = 0
    should_stop = False

    print("start training...")
    for epoch in range(1, args.epoch + 1):
        time0 = time()
        model.train()

        ''''# 先计算att权重
        with torch.no_grad():
            att = model('calc_att', graph_kg)
        graph_kg.edata['att'] = att
        print('Update attention scores: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))
        '''
        # 先将cf的顺序打乱
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        # 开始训练cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = n_params['epoch_num_cf']

        for s in tqdm(range(0, len(train_cf), args.batch_size),
                      desc='epoch:{},batching cf data set'.format(epoch)):
            time2 = time()
            batch = get_feed_dict(train_cf_pairs, s, s + args.batch_size, user_dict['train_user_set'])
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = batch['users'], batch['pos_items'], batch['neg_items']
            if device != 'cpu':
                cf_batch_user = cf_batch_user.to(device)
                cf_batch_pos_item = cf_batch_pos_item.to(device)
                cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model('calc_cf_loss', graph_kg, graph_i2u, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)

            cf_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

        print('CF Training: Epoch {:04f} Total Iter {:04f} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))
        '''
        # 开始kge
        time1 = time()
        kg_total_loss = 0
        exist_hand_idx = list(train_kg_dict.keys())
        random.shuffle(exist_hand_idx)
        n_kg_batch = n_params['epoch_num_kg']

        for s in tqdm(range(0, len(train_kg_dict), args.batch_size),
                      desc='epoch:{},batching kg data set'.format(epoch)):
            time2 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = get_feed_dict_kg(train_kg_dict, s, s + args.batch_size, n_entities, exist_hand_idx)
            if device != 'cpu':
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model('calc_kg_loss', kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)

            kg_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

        print('KG Training: Epoch {:04f} Total Iter {:04f} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time1, kg_total_loss / n_kg_batch))
        '''
        # 开始测试
        if epoch % 5 == 0 or epoch == 1:
            """testing"""
            model.eval()
            test_s_t = time()
            with torch.no_grad():
                ret = test(model, user_dict, n_params, graph_kg, graph_i2u)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "tesing time", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row([epoch, test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']])
            print(train_res)
            f = open('./result/{}.txt'.format(args.dataset), 'a+')
            f.write(str(train_res) + '\n')
            f.close()

            # *********************************************************
            cur_best, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best, stopping_step, expected_order='acc', flag_step=20)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best:
                save_model(model, args.save_dir, epoch, best_epoch)
                print('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best))
