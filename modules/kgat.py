import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.softmax import edge_softmax
from utils.helper import edge_softmax_fix
from modules.hyperbolic import *


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, n_params):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.n_users = n_params['n_users']
        self.n_entities = n_params['n_entities']
        self.n_relations = n_params['n_relations']
        self.n_items = n_params['n_items']

        self.message_dropout = nn.Dropout(dropout)

        self.gate1 = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.gate2 = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mode, sg, g_i2u, entity_embed, user_embed, relation_emb, item_cf_embed, sg_inv):
        # 先在KG上传播
        sg = sg.local_var()
        sg.ndata['node'] = entity_embed

        def tan_sum(edges):
            # tan_sum = logmap(project(mobius_add(expmap(edges.dst['node'], expmap0(edges.src['node'])), expmap(relation_emb(edges.data['type']+2), expmap0(edges.src['node'])))), edges.src['node'])*edges.data['att']
            tan_sum = logmap(project(mobius_add(expmap(edges.dst['node'], expmap0(edges.src['node'])), expmap(relation_emb(edges.data['type']+2), expmap0(edges.src['node'])))), expmap0(edges.src['node']))
            return {'tan_sum': tan_sum}

        sg.apply_edges(tan_sum, sg.edges(form='all')[2])

        sg_inv.edata['tan_sum'] = sg.edata['tan_sum']

        sg_inv.update_all(dgl.function.copy_e('tan_sum', 'temp'), dgl.function.mean('temp', 'out'))

        out = sg_inv.ndata['out']
        out = self.message_dropout(out)

        # 门控结合
        gi = self.sigmoid(self.gate1(entity_embed[:self.n_items]) + self.gate2(item_cf_embed))
        item_emb_fusion = (gi * entity_embed[:self.n_items]) + ((1 - gi) * item_cf_embed)

        # 再在UI上传播
        g_i2u = g_i2u.local_var()

        g_i2u.ndata['node'] = torch.cat([item_emb_fusion, user_embed], dim=0)

        g_i2u.update_all(dgl.function.copy_u('node', 't'), dgl.function.mean('t', 'u'))

        u = g_i2u.ndata['u'][self.n_items:]
        i_cf = g_i2u.ndata['u'][:self.n_items]

        return out, u, i_cf


class KGAT(nn.Module):

    def __init__(self, args, n_params,
                 user_pre_embed=None, item_pre_embed=None):

        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_params['n_users']
        self.n_entities = n_params['n_entities']
        self.n_relations = n_params['n_relations']
        self.n_items = n_params['n_items']

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)  # 层的维度
        self.mess_dropout = eval(args.mess_dropout)  # layers dropout
        self.n_layers = len(eval(args.conv_dim_list))  # 层数
        self.dropout = nn.Dropout(p=0.1)

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.num_neg_sample = args.num_neg_sample
        self.margin_ccl = args.margin

        # Embedding
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.entity_dim)
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.entity_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))  # 均匀分布
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)

        self.items_embed_cf = nn.Embedding(self.n_items, self.entity_dim)
        self.items_embed_cf.weight = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(self.n_items, self.entity_dim)), gain=nn.init.calculate_gain('relu'))

        # 注意嵌入传播层的W [n_relations, entity_dim, relation_dim]
        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        # 聚合层
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type, n_params))

    def att_score(self, edges):
        # Equation (4)
        r_mul_t = torch.matmul(self.entity_user_embed(edges.src['id']), self.W_r)  # (n_edge, relation_dim)
        r_mul_h = torch.matmul(self.entity_user_embed(edges.dst['id']), self.W_r)  # (n_edge, relation_dim)
        r_embed = self.relation_embed(edges.data['type'])  # (n_edge, relation_dim)
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)  # (n_edge, 1)
        return {'att': att}

    def compute_attention(self, g):
        g = g.local_var()  # 返回局部变量的图形对象，避免在退出函数时更改图形特征
        for i in range(self.n_relations-2):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)  # 返回满足给定边类型的边ID
            self.W_r = self.W_R[i+2]  # [entity_dim, relation_dim]
            g.apply_edges(self.att_score, edge_idxs)  # 添加新的边缘特征，也就是文章中的pi值，Knowledge-aware Attention

        # Equation (5)
        g.edata['att'] = edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r = r + 2
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.entity_user_embed(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_user_embed(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_user_embed(neg_t)  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score + 1e-6)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def cf_embedding(self, mode, g_kg, g_i2u):
        g_i2u = g_i2u.local_var()
        g_kg = g_kg.local_var()
        # 构建逆图
        g_kg_inv = dgl.graph((g_kg.edges()[1], g_kg.edges()[0]))

        # KG图构建边子图
        idx = np.random.choice(g_kg.all_edges(form='all')[2].shape[0], size=int(g_kg.all_edges(form='all')[2].shape[0] * 0.2), replace=False)
        sg = dgl.edge_subgraph(g_kg, idx, relabel_nodes=False)
        sg_inv = dgl.edge_subgraph(g_kg_inv, idx, relabel_nodes=False)

        ego_embed = self.entity_user_embed(sg.ndata['id'])

        u_embed = self.entity_user_embed.weight[self.n_entities:]
        item_cf_embed = self.items_embed_cf.weight[:]

        entities_res = ego_embed
        user_res = u_embed
        item_cf_res = item_cf_embed

        # 高阶传播
        for i, layer in enumerate(self.aggregator_layers):
            ego_embed, u_embed, item_cf_embed = layer(mode, sg, g_i2u, ego_embed, u_embed, self.relation_embed, item_cf_embed, sg_inv)
            # message dropout
            ego_embed = self.dropout(ego_embed)
            u_embed = self.dropout(u_embed)
            item_cf_embed = self.dropout(item_cf_embed)

            ego_embed = F.normalize(ego_embed)
            u_embed = F.normalize(u_embed)
            item_cf_embed = F.normalize(item_cf_embed)

            entities_res = torch.add(entities_res, ego_embed)
            user_res = torch.add(user_res, u_embed)
            item_cf_res = torch.add(item_cf_res, item_cf_embed)

        return entities_res, user_res, item_cf_res

    def cf_score(self, mode, g_kg, g_i2u):
        """
        user_ids:   number of users to evaluate   (n_eval_users)
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        entities_embed, users_embed, item_cf_embed = self.cf_embedding(mode, g_kg, g_i2u)  # (n_users + n_entities, cf_concat_dim)
        entities_embed[:self.n_items] += item_cf_embed
        return users_embed, entities_embed[:self.n_items]

    def calc_cf_loss(self, mode, g_kg, g_i2u, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size, N_num_neg)
        """
        entities_embed, users_embed, item_cf_embed = self.cf_embedding(mode, g_kg, g_i2u)  # (n_users + n_entities, cf_concat_dim)

        item_neg_ids = item_neg_ids.view(-1)
        u_e = users_embed[user_ids]
        pos_e, neg_e = entities_embed[item_pos_ids], entities_embed[item_neg_ids]
        pos_e_cf, neg_e_cf = item_cf_embed[item_pos_ids], item_cf_embed[item_neg_ids]
        loss = self.create_contrastive_loss(u_e, pos_e, neg_e, pos_e_cf, neg_e_cf)

        return loss

    def rate(self, mode, u_g_embeddings, i_g_embeddings):
        return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2).detach().cpu()

    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_cf_loss':
            return self.calc_cf_loss(mode, *input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.cf_score(mode, *input)
        if mode == 'rating':
            return self.rate(mode, *input)

    def create_contrastive_loss(self, u_e, pos_e, neg_e, pos_e_cf, neg_e_cf):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e)
        pos_e = F.normalize(pos_e)
        neg_e = F.normalize(neg_e)
        pos_e_cf = F.normalize(pos_e_cf)
        neg_e_cf = F.normalize(neg_e_cf)

        ui_pos = torch.relu(2 - (torch.cosine_similarity(u_e, pos_e, dim=1) + torch.cosine_similarity(u_e, pos_e_cf, dim=1)))
        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1 > 0
        ui_neg_loss1 = torch.sum(ui_neg1, dim=-1)/(torch.sum(x, dim=-1) + 1e-5)

        ui_neg2 = torch.relu(torch.cosine_similarity(users_batch, neg_e_cf, dim=1) - self.margin_ccl)
        ui_neg2 = ui_neg2.view(batch_size, -1)
        x = ui_neg2 > 0
        ui_neg_loss2 = torch.sum(ui_neg2, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = ui_pos + ui_neg_loss1 + ui_neg_loss2

        return loss.mean()
