"""
Generates data for the graph evolve them over time.
"""

# Copyright (C) 2019-2021
# Zhiyan guo
# Hunan University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import graph_edits as ge
import networkx as nx
from collections import Counter
import random
import os

__author__ = 'Zhiyan guo'
__copyright__ = f'Copyright 2019-2021, {__author__}'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Zhiyan guo'
__email__ = 'zhiyanguo@hnu.edu.cn'

node_label_name = "type"
node_id_name = "label"
edge_id_name = "id"

AIDS_NODE_LABEL = ['Si', 'O', 'Se', 'Ni', 'Cl', 'Co', 'Tb', 'Bi', 'Ho', 'S', 'P', 'B', 'Pd', 'Cu', 'Br', 'Ru',
                   'C', 'Sn', 'Pt', 'Ga', 'N', 'Te', 'Hg', 'F', 'Pb', 'Li', 'Sb', 'I', 'As']
AIDS_NODE_LABEL = sorted(AIDS_NODE_LABEL)

# 约束条件
NODE_NUM_LOWER_BOUND = 4
EDGE_NUM_LOWER_BOUND = 4

def extract_node_label_distribution(dataset):
    label_counter = Counter()
    neighbor_label_counter = Counter()
    for g in dataset:
        labels = list(map(lambda x: g.nodes[x][node_label_name], g.nodes))
        label_counter.update(labels)
        for node in g.nodes:
            label = g.nodes[node][node_label_name]
            nigs = g.neighbors(node)
            nigs_label = list(map(lambda x: g.nodes[x][node_label_name], nigs))
            tmp = neighbor_label_counter.get(label, Counter())
            tmp.update(nigs_label)
            if neighbor_label_counter.get(label, None) is None:
                neighbor_label_counter[label] = tmp
    return label_counter, neighbor_label_counter


def normalization_distribution(dis: Counter):
    items = list(dis.items())
    assert isinstance(items[0][1], (int, float))
    values = [x[1] for x in items]
    n = sum(values)
    items = [(x[0], x[1] / n) for x in items]

    normalized_dis = {k: v for k, v in items}

    return normalized_dis


def weighted_sample(weight: dict):
    """
    带权随机采样
    :param weight: 数据随对于的权重，dict类型
    :return: 采样到数据
    """
    items = weight.items()
    items = sorted(items, key=lambda x: x[1], reverse=True)

    s = 0
    r = np.random.random()
    for item in items:
        s += item[1]
        if s >= r:
            return item[0]

    return items[-1][0]


def sample_node_label(node_label_dis: dict, thresh=0.1):
    alpha = np.random.random()
    node_labels = list(dict.keys())
    if alpha < thresh:
        sampled = random.sample(node_labels, 1)[0]
    else:
        sampled = weighted_sample(node_label_dis)

    return sampled


def sample_neighbor_node_label(g: nx.Graph, node_index, node_label, node_neighbor_label_dis: dict, thresh=0.1, node_label_name=""):
    """
    根据给定结点的索引，从graph中采样一个结点。注意有以下问题：
    1）不能采样到自己；
    2）先根据条件概率采样得到结点的标签，再从graph中有相应标签的结点中均匀采样；
    3）有可能采样到的标签是给定节点的标签，且与给点结点的标签相同的结点只有给定节点；
    """
    alpha = np.random.random()
    nodes = g.nodes

    indexes = list(range(len(nodes)))

    if alpha < thresh:
        while True:
            sampled = random.sample(indexes, 1)[0]
            if sampled != node_index:
                return sampled
    else:
        new_label_dis = {}
        label_counter = Counter()
        for i, node in enumerate(nodes):
            new_label_dis[g.nodes[node][node_label_name]] = node_neighbor_label_dis[node_label].get(g.nodes[node][node_label_name], 0)

            tmp = label_counter.get(g.nodes[node][node_label_name], [])
            tmp.append(i)
            if label_counter.get(g.nodes[node][node_label_name], None) is None:
                label_counter[g.nodes[node][node_label_name]] = tmp

        if len(label_counter.get(node_label, [])) == 1 and node_index != -1:
            new_label_dis.pop(node_label)
        s = sum(new_label_dis.values())
        left = 1 - s
        num_labels = len(new_label_dis.items())
        left = left / num_labels
        new_label_dis = {k:v + left for k, v in new_label_dis.items()}
        sampled_label = weighted_sample(new_label_dis)
        sampled = random.sample(label_counter[sampled_label], 1)[0]

        return sampled


def sample_node(g: nx.Graph, node_label_dis: dict, thresh=.1, node_label_name=""):
    alpha = np.random.random()
    nodes = g.nodes
    indexes = list(range(len(nodes)))
    if alpha < thresh:
        sampled = random.sample(indexes, 1)[0]
    else:
        new_label_dis = {}
        label_counter = Counter()
        for i, node in enumerate(nodes):
            label = g.nodes[node][node_label_name]
            new_label_dis[g.nodes[node][node_label_name]] = node_label_dis[label]

            tmp = label_counter.get(g.nodes[node][node_label_name], [])
            tmp.append(i)
            if label_counter.get(g.nodes[node][node_label_name], None) is None:
                label_counter[g.nodes[node][node_label_name]] = tmp

        s = sum(new_label_dis.values())
        left = 1 - s
        num_labels = len(new_label_dis.items())
        left = left / num_labels
        new_label_dis = {k:v + left for k, v in new_label_dis.items()}
        sampled_label = weighted_sample(new_label_dis)
        sampled = random.sample(label_counter[sampled_label], 1)[0]

    return sampled


def gen_edit_num(loc=9, scale=5):
    while True:
        n = np.random.normal(loc, scale)
        n = int(n)
        if n > 0:
            return n


def nxgraph_to_adjlist(g: nx.Graph, directed=False):
    nodes = list(g.nodes)
    A = np.zeros(shape=(len(nodes), len(nodes)))
    edges = list(g.edges)
    for edge in edges:
        u = nodes.index(edge[0])
        v = nodes.index(edge[1])
        A[u,v] = 1
        if not directed:
            A[v,u] = 1

    return A


def nxgraph_to_feature_matrix(g: nx.Graph, node_label_name="type", node_labels=[]):
    assert len(node_labels) != 0
    nodes = list(g.nodes)
    X = np.zeros(shape=(len(nodes), len(node_labels)))
    for idx, node in enumerate(nodes):
        label = g.nodes[node][node_label_name]
        label_idx = node_labels.index(label)
        X[idx, label_idx] = 1

    return X


def is_graph_connected():
    pass


def can_del_node(index, edits):
    for edit in edits:
        if isinstance(edit, ge.NodeDeletion):
            if edit._index == index:
                return False
        elif isinstance(edit, ge.NodeInsertion):
            if edit._index == index:
                return False
        elif isinstance(edit, ge.EdgeInsertion):
            if edit._i == index or edit._j == index:
                return False
        elif isinstance(edit, ge.EdgeDeletion):
            continue
    return True


def can_add_node(index, edits):
    for edit in edits:
        if isinstance(edit, ge.NodeDeletion):
            if edit._index == index:
                return False
        elif isinstance(edit, ge.NodeInsertion):
            continue
        elif isinstance(edit, ge.EdgeInsertion):
            continue
        elif isinstance(edit, ge.EdgeDeletion):
            continue
    return True


def can_del_edge(i, j, edits):
    for edit in edits:
        if isinstance(edit, ge.NodeDeletion):
            if edit._index == i or edit._index == j:
                return False
        elif isinstance(edit, ge.NodeInsertion):
            continue
        elif isinstance(edit, (ge.EdgeInsertion, ge.EdgeDeletion)):
            if (edit._i == i and edit._j == j) or (edit._i == j and edit._j == i):
                return False
    return True


def can_add_edge(i, j, edits):
    for edit in edits:
        if isinstance(edit, ge.NodeDeletion):
            if edit._index == i or edit._index == j:
                return False
        elif isinstance(edit, ge.NodeInsertion):
            continue
        elif isinstance(edit, (ge.EdgeInsertion, ge.EdgeDeletion)):
            if (edit._i == i and edit._j == j) or (edit._i == j and edit._j == i):
                return False
    return True


def edit_graph(graph: nx.Graph, edit_num: int = None, node_label_dis=None, neighbor_label_dis=None, seed=0):
    '''
    每一次的编辑中不能有冲突的操作，这里的一次编辑可以包含多个编辑操作，一次指的是连续的不会产生冲突的编辑序列.
    1）新增结点时，不能删除与之相连的结点；
    2）不能增加了边又删除该边，可以不用考虑，因为在一次编辑中；
    3）不能删除了边又增加该边；
    4）不能删除了结点又在该结点的基础上增加边/结点，删除边/结点；
    5）
    :param graph:
    :param edit_num:
    :param seed:
    :return:
    '''

    node_labels = AIDS_NODE_LABEL

    tar = graph
    # print(tar)
    edits = []

    np.random.seed(seed)
    def gen_node_id(g: nx.Graph):
        nid = g.nodes.__len__()
        while True:
            while str(nid) in g:
                nid += 1
            yield nid
            nid += 1
    node_id_gen = gen_node_id(tar)

    i = 0

    while i < edit_num:
        r = np.random.rand()
        try:
            nodes = list(tar.nodes)
            if r < .18:  # 增加一个结点， 同时要为其增加一条边
                nid_1 = str(next(node_id_gen))
                label = nid_1 if not node_labels else weighted_sample(node_label_dis)  # random.sample(node_labels, 1)[0]
                new_node_attr = [0]*len(node_labels)
                new_node_attr[node_labels.index(label)] = 1
                # 为新增的结点增加一条边，选择另一结点
                index = sample_neighbor_node_label(tar, -1, label, neighbor_label_dis, node_label_name=node_label_name)  # random.sample(tar.nodes, 1)[0]
                if not can_add_node(index, edits):
                    continue
                edit = ge.NodeInsertion(index, attribute=new_node_attr, directed=False)
            elif r < .3:  # 删除一个结点
                if tar.nodes.__len__() <= NODE_NUM_LOWER_BOUND:  # 判断是否低于结点的最低数量
                    print(f"Node num will lower than NODE_NUM_LOWER_BOUND({NODE_NUM_LOWER_BOUND}) ! Node delete operation is aborted!")
                    continue
                index = sample_node(tar, node_label_dis, node_label_name=node_label_name)  # random.sample(tar.nodes, 1)[0]
                if not can_del_node(index, edits):
                    continue
                edit = ge.NodeDeletion(index)
            elif r < .68:  # 增加一条边
                nid_1, nid_2 = random.sample(tar.nodes, 2)
                if tar.has_edge(nid_1, nid_2):  # 判断是否有重复的边
                    # print(f"has edge : {nid_1} - {nid_2}")
                    continue
                order1 = nodes.index(nid_1)
                order2 = nodes.index(nid_2)
                if not can_add_edge(order1, order2, edits):
                    continue
                edit = ge.EdgeInsertion(order1, order2, directed=False)
            elif r < .90:  # 删除一条边
                if tar.edges.__len__() <= EDGE_NUM_LOWER_BOUND:  # 判断是否低于边的最低数量
                    print(f"Edge num will lower than EDGE_NUM_LOWER_BOUND({EDGE_NUM_LOWER_BOUND}) ! Edge delete operation is aborted!")
                nid_1, nid_2 = random.sample(tar.edges, 1)[0]
                if not tar.has_edge(nid_1, nid_2):
                    continue
                order1 = nodes.index(nid_1)
                order2 = nodes.index(nid_2)
                if not can_del_edge(order1, order2, edits):
                    continue
                edit = ge.EdgeDeletion(order1, order2, directed=False)
            else:  # 改变结点的标签
                continue
                # if node_labels is None:
                #     continue
                # nid = random.sample(tar.nodes, 1)[0]
                # label = random.sample(node_labels, 1)[0] if np.random.random() < .4 else weighted_sample(node_label_dis)
                # old_label = tar.nodes[nid][node_label_name]
                # order = nodes.index(nid)
                # new_node_attr = [0]*len(node_labels)
                # new_node_attr[node_labels.index(label)] = 1
                # edit = ge.NodeReplacement(order, new_node_attr)
                # tar.nodes[nid][node_label_name] = label
        except Exception as ex:
            print(ex)
            # raise RuntimeError(ex)
        edits.append(edit)
        i += 1

    # fig, axes = plt.subplots(1,2)
    # nx.draw(graph,ax=axes[0], with_labels=True)
    # nx.draw(epd.target, ax=axes[1], with_labels=True)
    # plt.pause(.82)
    # plt.close(fig)

    return edits


def apply(g: nx.Graph, edit: ge.Edit, nodes):
    offset = 20
    nodes = list(map(int, nodes))
    if isinstance(edit, ge.NodeInsertion):
        g.add_node(f'{offset+max(nodes)}')
        g.nodes[f'{offset+max(nodes)}'][node_label_name] = AIDS_NODE_LABEL[np.argmax(edit._attribute)]
    elif isinstance(edit, ge.NodeDeletion):
        g.remove_node(str(nodes[edit._index]))
    elif isinstance(edit, ge.EdgeInsertion):
        u = str(nodes[edit._i])
        v = str(nodes[edit._j])
        g.add_edge(u, v)
    elif isinstance(edit, ge.EdgeDeletion):
        u = str(nodes[edit._i])
        v = str(nodes[edit._j])
        g.remove_edge(u, v)
    else:
        raise ValueError("Unsupported edit")

    return g


dataset = "AIDS700nef"
dataset_path = "./dataset/" + dataset
folders = os.listdir(dataset_path)
folders = list(filter(lambda x: os.path.isdir( os.path.join(dataset_path, x) ), folders))
all_graphs = []
for e in folders:
    path = os.path.join(dataset_path, e)
    fns = os.listdir(path)
    fns = sorted(fns, key=lambda x: int(x[:-5]))
    # print(fns)
    for ee in fns:
        g = nx.read_gexf(os.path.join(path, ee))
        all_graphs.append(g)

node_label_dis, neighbor_label_dis = extract_node_label_distribution(all_graphs)
nor_node_label_dis = normalization_distribution(node_label_dis)
nor_neighbor_node_label_dis = dict()
for k, v in neighbor_label_dis.items():
    nor_neighbor_node_label_dis[k] = normalization_distribution(v)


def generate_time_series(T):
    """ Runs the graph edit cycle c for T time steps, starting from step t.

    Parameters
    ----------
    T: int
        The length of the output time series.

    Returns
    -------
    As: list
        A time series of adjacency matrices.
    Xs: list
        A time series of node attribute matrices.
    deltas: list
        A time series of targets, i.e. tuples of node scores and
        edge scores, where -1 indicates a deletion and +1 indicates
        an insertion.
    Epsilons: list
        a time series of edge edit matrices, where Epsilons[t][i, j] = +1 if
        nodes i and j are newly connected at time t, Epsilons[t][i, j] = -1 if
        the egde (i, j) is removed at time t, and Epsilons[t][i, j] = 0
        otherwise.

    """
    As = []
    Xs = []
    deltas = []
    Epsilons = []
    # todo 随机从数据集中抽取出一个graph，对其进行编辑
    # todo 编辑过程：
    """
    0. 初始step = 1; 
    1. 确定当前step编辑操作数量；
    2. 调用edit_graph对graph进行编辑，生成一个(A, X, delta, Epsilon)；
    3. 当step < T 时，返回1，否则进入4；
    4. 将所有step中得到的A, X, delta, Epsilon分别放进一个list中，得到As, Xs, deltas, Epsilons；
    5. 返回As, Xs, deltas, Epsilons， 程序结束；
    
    注意：As, Xs, deltas, Epsilons 应该一样长，deltas, Epsilons的最后一个应该是全零的
    """
    g = random.sample(all_graphs, 1)[0]
    A = nxgraph_to_adjlist(g, directed=False)
    X = nxgraph_to_feature_matrix(g, node_label_name, AIDS_NODE_LABEL)
    As.append(A)
    Xs.append(X)
    for t in range(T):

        edit_num = np.random.randint(1, 4)
        seed = 3297  # np.random.randint(0, 100000)
        edits = edit_graph(g, edit_num, nor_node_label_dis, nor_neighbor_node_label_dis, seed=seed)
        print(f"{t}: {len(g)} {edits}")
        delta = np.zeros(len(A))
        Epsilon = np.zeros_like(A)
        for edit in edits:
            _d, _E = edit.score(len(A))
            delta += _d
            Epsilon += _E
        nodes = list(g.nodes)
        for edit in edits:
            if isinstance(edit, (ge.EdgeDeletion, ge.EdgeInsertion)):
                print(edit)
            # A, X = edit.apply(A, X)
            g = apply(g, edit, nodes)
            # tmp = A - nxgraph_to_adjlist(g)
            # if np.sum(tmp) != 0:
            #     print("Error!!!")
        A = nxgraph_to_adjlist(g, directed=False)
        X = nxgraph_to_feature_matrix(g, node_label_name, AIDS_NODE_LABEL)
        As.append(A)
        Xs.append(X)
        deltas.append(delta)
        Epsilons.append(Epsilon)
    deltas.append(np.zeros(len(A)))
    Epsilons.append(np.zeros_like(X))
    return As, Xs, deltas, Epsilons


if __name__ == "__main__":
    As, Xs, deltas, Epsilons = generate_time_series(10)
    print(deltas)

