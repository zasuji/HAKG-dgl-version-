import dgl
import numpy as np
import torch
from collections import defaultdict

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def load_data_both(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    # 获得[u_id,i_id]对
    remap_cf(train_cf, test_cf)
    # 在user_set中构建形如{u_id:[i_id]}的字典

    print('combi train_cf and kg')
    triplets, train_kg_dict, r_num = read_triplets_gui(directory + 'kg_final.txt')

    print('build graph')
    graph, graph_UIS, relation_num_dict = build_graph(train_cf, triplets)
    graph_kgat_all, graph_kg, graph_i2u = build_graph_link(train_cf, triplets, args.link_nei)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        #  因为还需要包括U-I之间的交互（包括正反向）所以加2，然后还需要把link_nei加入
        'n_relations': int(n_relations+2*args.link_nei+2),
        'n_kg_train': len(triplets),
        'num_r': r_num,
        'num_r0': relation_num_dict
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, train_kg_dict, n_params, graph, graph_UIS, graph_kgat_all, graph_kg, graph_i2u


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_id = inters[0], inters[1:]
        pos_id = list(set(pos_id))
        for i_id in pos_id:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def remap_cf(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append((int(i_id)))


def read_triplets_gui(files):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(files, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    # kg中是否包括反向的关系
    if args.inverse_r:
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    # kg中有多少的实体，其中已经包括了item
    n_nodes = n_entities + n_users
    # 这里正常+1，但是因为我需要link的关系所以我在上面n_parm里面会额外加入u-i
    n_relations = max(triplets[:, 1]) + 1

    train_kg_dict = defaultdict(list)
    for row in triplets:
        h, r, t = row
        train_kg_dict[h].append((t, r))

    # 这里获得了以r为关系的头节点的数量
    cs = []
    for i in range(n_relations):
        idx = np.where(triplets[:, 1] == i)[0]
        cs.append(len(list(set(triplets[idx, 0].tolist()))))
    cs = np.array(cs)

    return triplets, train_kg_dict, cs


def build_graph(train_data, triplets):
    # 从kg开始建立图
    relation_dict = {}
    relation_num_dict = {}
    for i in range(n_relations):
        idx = np.where(triplets[:, 1] == i)[0]
        node_pair = triplets[:, [0, 2]][idx]
        name = ('item', i, 'item')
        relation_dict[name] = (node_pair[:, 0].tolist(), node_pair[:, 1].tolist())
        # 获得了以r为key的节点对（来自kg）
        relation_num_dict[name] = len(idx)
        # 获得了每一种关系在kg中的数量
    graph = dgl.heterograph(relation_dict)

    # 建立u-i图
    relation_dict_ui = {}
    name = ('item', i, 'user')
    relation_dict_ui[name] = (train_data[:, 1], train_data[:, 0])

    name_graph_UIS = {'user': n_users, 'item': n_items}
    graph_UIS = dgl.heterograph(relation_dict_ui, name_graph_UIS)

    return graph, graph_UIS, relation_num_dict


def build_graph_link(train_data, triplets, n_link):

    triplets_gai = triplets.copy()
    triplets_gai[:, 1] += 2*args.link_nei + 2
    num_of_train_data = train_data.shape[0]

    # 后面再改，跑起来再说
    a = np.zeros((num_of_train_data, 3))
    a[:, 0] = train_data[:, 0] + n_entities
    a[:, 1] = 0
    a[:, 2] = train_data[:, 1]
    b = np.zeros((num_of_train_data, 3))
    b[:, 0] = train_data[:, 1]
    b[:, 1] = 1
    b[:, 2] = train_data[:, 0] + n_entities

    for i in range(n_link):
        # 这里没写完
        print('here')
    # 创建all图
    kg_train_data = np.concatenate((triplets_gai, a, b), axis=0)
    g_all = dgl.graph((kg_train_data[:, 0], kg_train_data[:, 2]))
    g_all.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)  # 节点
    g_all.edata['type'] = torch.LongTensor(kg_train_data[:, 1])  # 边

    # 创建kg图
    g_kg = dgl.graph((triplets[:, 0], triplets[:, 2]))
    g_kg.ndata['id'] = torch.arange(n_entities, dtype=torch.long)  # 节点
    g_kg.edata['type'] = torch.LongTensor(triplets[:, 1])  # 边

    # 创建i2u图
    g_i2u = dgl.graph((np.concatenate((train_data[:, 1], train_data[:, 0]+n_items), axis=0), np.concatenate((train_data[:, 0]+n_items, train_data[:, 1]), axis=0)))
    g_i2u.ndata['id'] = torch.arange(n_items+n_users, dtype=torch.long)  # 节点

    return g_all, g_kg, g_i2u


def load_pretrain_data(pretrain_embedding_dir, dataset_name, n_user, n_items, entities_dim):
    pre_model = 'mf'
    pretrain_path = '%s/%s/%s.npz' % (dataset_name, pretrain_embedding_dir, pre_model)
    pretrain_data = np.load(pretrain_path)
    user_pre_embed = pretrain_data['user_embed']
    item_pre_embed = pretrain_data['item_embed']

    assert user_pre_embed.shape[0] == n_user
    assert item_pre_embed.shape[0] == n_items
    assert user_pre_embed.shape[1] == entities_dim
    assert item_pre_embed.shape[1] == entities_dim

    return user_pre_embed, item_pre_embed