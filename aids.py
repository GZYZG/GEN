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


def edit_graph(graph: nx.Graph, edit_num: int = None, seed=0, dataset=None, node_label_dis=None, neighbor_label_dis=None):
    '''

    :param graph:
    :param edit_num:
    :param seed:
    :param dataset:
    :return:
    '''
    if not dataset:
        raise RuntimeError("Please specify one dataset: AIDS, LINUX, IMDB-MULTI")

    node_labels = AIDS_NODE_LABEL

    tar = graph.copy()
    # print(tar)
    epd = EditPathData(graph, tar, [])
    delta = None
    Epsilon = None


    np.random.seed(seed)
    if edit_num is None:
        edit_num = gen_edit_num()

    def gen_node_id(g: nx.Graph):
        nid = g.nodes.__len__()
        while True:
            while str(nid) in g:
                nid += 1
            yield nid
            nid += 1
    node_id_gen = gen_node_id(tar)

    max_node_id = tar.nodes.__len__()-1     # 注意！！！结点的id可能是不连续的！！！
    max_edge_id = tar.nodes.__len__()-1
    i = 0
    while i < edit_num:
        r = np.random.rand()
        try:
            nodes = list(tar.nodes)
            if r < .18:  # 增加一个结点， 同时要为其增加一条边
                max_node_id += 1
                nid_1 = str(next(node_id_gen))
                label = nid_1 if not node_labels else weighted_sample(node_label_dis)  # random.sample(node_labels, 1)[0]
                # 为新增的结点增加一条边，选择另一结点
                index = sample_neighbor_node_label(tar, -1, label, neighbor_label_dis, node_label_name=node_label_name)  # random.sample(tar.nodes, 1)[0]
                nid_2 = nodes[index]
                max_edge_id += 1
                order2 = nodes.index(nid_2)
                node1_attr = {node_label_name: label, node_id_name: nid_1}
                node2_attr = tar.nodes[nid_2]

                node1 = node_labels.index(label)
                node2 = order2

                e = EditElement(EditType.ADD_NODE, node1, node2, label, node1_attr, node2_attr)

                tar.add_node(nid_1)
                tar.nodes[nid_1][node_label_name] = label
                tar.nodes[nid_1][node_id_name] = nid_1
                tar.add_edge(nid_1, nid_2)
                tar.edges[nid_1, nid_2][edge_id_name] = str(max_edge_id)
                # print(f"add node : {nid}\t{tar.nodes}")
            elif r < .3:  # 删除一个结点
                if tar.nodes.__len__() <= NODE_NUM_LOWER_BOUND:  # 判断是否低于结点的最低数量
                    print(f"Node num will lower than NODE_NUM_LOWER_BOUND({NODE_NUM_LOWER_BOUND}) ! Node delete operation is aborted!")
                    continue
                index = sample_node(tar, node_label_dis, node_label_name=node_label_name)  # random.sample(tar.nodes, 1)[0]
                nid = nodes[index]
                # e = EditNode("del", nid)
                tar_copy = tar.copy()
                tar_copy.remove_node(nid)
                if not nx.is_connected(tar_copy):  # 如果导致不连通则抛弃这个操作
                    print(f"Graph will be disconnected if delete node {nid}! Node delete operation is aborted!")
                    continue
                order = nodes.index(nid)
                node_attr = tar.nodes[nid]
                node1 = order
                node2 = [nodes.index(nb) for nb in tar.neighbors(nid)]

                e = EditElement(EditType.DEL_NODE, node1, node2, node1_attr=node_attr)
                tar.remove_node(nid)
                # print(f"del node : {nid}\t{tar.nodes}")
            elif r < .68:  # 增加一条边
                nid_1, nid_2 = random.sample(tar.nodes, 2)
                if (not nid_1 in tar) or (not nid_2 in tar):
                    raise RuntimeError("增加边时，结点不在图的结点列表中！")
                if tar.has_edge(nid_1, nid_2):  # 判断是否有重复的边
                    # print(f"has edge : {nid_1} - {nid_2}")
                    continue
                max_edge_id += 1
                order1 = nodes.index(nid_1)
                order2 = nodes.index(nid_2)
                node1_attr = tar.nodes[nid_1]
                node2_attr = tar.nodes[nid_2]
                edge_attr = {edge_id_name: str(max_edge_id)}
                node1 = order1
                node2 = order2
                e = EditElement(EditType.ADD_EDGE, node1, node2,
                                node1_attr=node1_attr, node2_attr=node2_attr, edge_attr=edge_attr)

                tar.add_edge(nid_1, nid_2)
                tar.edges[nid_1, nid_2][edge_id_name] = str(max_edge_id)
            elif r < .90:  # 删除一条边
                if tar.edges.__len__() <= EDGE_NUM_LOWER_BOUND:  # 判断是否低于边的最低数量
                    print(f"Edge num will lower than EDGE_NUM_LOWER_BOUND({EDGE_NUM_LOWER_BOUND}) ! Edge delete operation is aborted!")
                nid_1, nid_2 = random.sample(tar.edges, 1)[0]
                tar_copy = tar.copy()
                tar_copy.remove_edge(nid_1, nid_2)
                if not nx.is_connected(tar_copy):  # 如果导致不连通，则抛弃操作
                    print(f"Graph will be disconnected if delete edge ({nid_1, nid_2}) ! Edge delete operation is aborted!")
                    continue
                order1 = nodes.index(nid_1)
                order2 = nodes.index(nid_2)
                node1_attr = tar.nodes[nid_1]
                node2_attr = tar.nodes[nid_2]

                node1 = order1
                node2 = order2
                e = EditElement(EditType.DEL_EDGE, node1, node2,
                                node1_attr=node1_attr, node2_attr=node2_attr, edge_attr=tar.edges[nid_1, nid_2])
                tar.remove_edge(nid_1, nid_2)
            else:  # 改变结点的标签
                if node_labels is None:
                    continue
                nid = random.sample(tar.nodes, 1)[0]
                label = random.sample(node_labels, 1)[0] if np.random.random() < .4 else weighted_sample(node_label_dis)
                old_label = tar.nodes[nid][node_label_name]
                order = nodes.index(nid)
                node_attr = tar.nodes[nid]
                node1 = order
                node2 = node_labels.index(label)
                e = EditElement(EditType.CHG_NODE_LABEL, node1, node2, node1_attr=node_attr)
                # e = EditNodeLabel(order, old_label, label, node_attr=node_attr)
                tar.nodes[nid][node_label_name] = label
        except Exception as ex:
            print(ex)
            # raise RuntimeError(ex)
        epd.add_edit_element(e)
        i += 1

    if epd.edit_path.__len__() == 0:
        print("something wrong ... ")

    # fig, axes = plt.subplots(1,2)
    # nx.draw(graph,ax=axes[0], with_labels=True)
    # nx.draw(epd.target, ax=axes[1], with_labels=True)
    # plt.pause(.82)
    # plt.close(fig)

    return epd


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
    """

    return As, Xs, deltas, Epsilons

