"""
Generates data for the 'degree rules' data set, where every node implements
the following rules:

1. delete the node if the degree is larger than 3.
2. add an edge to another node if both nodes share at least one neighbor.
3. add a node if the node degree is smaller than 3.

Rule 1 is always executed before rule 2 and rule 2 is always executed before
rule 3. When a rule applies to several nodes, older nodes take precedence.
Rules are applied in parallel for all connected components.

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
import graph_edits
import os
import glob
import random
from aids import nxgraph_to_adjlist, nxgraph_to_feature_matrix
import networkx as nx

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

dataset_path = "./dataset/"
dataset = "AIDS700nef"
aids_path = os.path.join(dataset_path, dataset)
train_data_path = os.path.join(aids_path, "train")
train_graph_files = glob.glob(f"{train_data_path}/*.gexf")


def next_step(A, n_max = None):
    """ Generates the next step for a graph with adjacency matrix A,
    according to the rules.

    Parameters
    ----------
    A: class numpy.array
        A graph, given as an adjacency matrix.
    n_max: int (default = len(A))
        A reference number of dimensions for the node encoding, which will be
        a one-hot coding padded with zeros if necessary.

    Returns
    -------
    edits: list
        A list of graph edits that should be applied next.
    delta: class numpy.array
        The node len(A) x 1 edit vector with entries +1 for node insertions and
        -1 for node deletions. This vector considers all nodes for which a rule
        applies, disregarding the preference rules.
    Epsilon: class numpy array
        The egde len(A) x len(A) edit matrix with entries +1 for egde
        insertions and -1 for edge deletions. This matrix considers all edges
        for which a rule applies, disregarding the preference rules.

    """
    if n_max is None:
        n_max = len(A)
    elif n_max < len(A):
        raise ValueError('n_max must be at least len(A) = %d to ensure that the graph is codeable.' % len(A))
    # first, distribute the graph into connected components
    # and process smaller connected components first.
    Cs = []
    remaining = set(range(len(A)))
    while(remaining):
        i = min(remaining)
        C = []
        stk = [i]
        while(stk):
            i = stk.pop()
            if(i not in remaining):
                continue
            remaining.remove(i)
            C.append(i)
            for j in np.where(A[i, :])[0]:
                if(j in remaining):
                    stk.append(j)
        Cs.append(np.array(C, dtype=int))
    # process all connected components
    edits = []
    dels  = []
    new_node_idx = len(A)
    for C in Cs:
        C.sort()
        # sort the nodes inside the component according to degree
        degrees = np.sum(A[C, :], axis=1)
        deg_ordered = np.argsort(degrees)
        # evaluate rule 1, i.e. delete nodes with degree higher than 3,
        # where nodes with higher degree take precedence
        if(degrees[deg_ordered[-1]] > 3):
            dels.append(graph_edits.NodeDeletion(C[deg_ordered[-1]]))
            continue
        # evaluate rule 2, i.e. connect nodes with shared neighbors,
        # where tuples with lower degrees take precedence
        rule2mat = np.logical_and(np.dot(A[C, :], A[:, C]) > 0.5, np.logical_not(A[C, :][:, C]))
        np.fill_diagonal(rule2mat, False)
        if np.any(rule2mat):
            degmat   = np.expand_dims(degrees, 0) + np.expand_dims(degrees, 1)
            degmat[np.logical_not(rule2mat)] = np.max(degmat) + 1
            i, j = np.unravel_index(np.argmin(degmat), degmat.shape)
            i, j = C[i], C[j]
            edits.append(graph_edits.EdgeInsertion(i, j, False))
            continue
        # evaluate rule 3, i.e. add new nodes to nodes with a low degree,
        # preferring nodes with lowest degrees
        if(degrees[deg_ordered[0]] < 3):
            new_node_code = np.zeros(n_max)
            new_node_code[new_node_idx] = 1
            new_node_idx += 1
            edits.append(graph_edits.NodeInsertion(C[deg_ordered[0]], new_node_code, False))
    # append deletions at the end of the edits list
    edits += dels
    # compute delta and Epsilon
    delta = np.zeros(len(A))
    Epsilon = np.zeros(A.shape)
    degrees = np.sum(A, axis=1)
    delta[degrees > 3] = -1. # rule 1
    Epsilon[np.logical_and(np.dot(A, A) > 0.5, np.logical_not(A))] = 1. # rule 2
    np.fill_diagonal(Epsilon, 0.) # correct for self-connections
    delta[degrees < 3] = 1. # rule 3
    # return results
    return edits, delta, Epsilon

def generate_time_series(A, n_max = None, t_max = 100):
    """ Generates a time series of graphs, in terms of their adjacency matrix,
    based on the degree rules from the given seed graph. The time series stops
    once the graph converges to a stable state.

    Parameters
    ----------
    A: class numpy.array
        An initial adjacency matrix for an undirected graph.
    n_max: int (default = 4 * len(A))
        a reference number of dimensions for the node encoding, which will be a
        one-hot coding padded with zeros if necessary.
    t_max: int (default = 100)
        A maximum number of steps.

    Returns
    -------
    As: list
        A time series of graphs until the graph converges to a stable state
        or t_max steps have been done.
    Xs: list
        a time series of node attribute matrices. These are just one-hot coding
        vectors of the node id.
    deltas: list
        a time series of node edit vectors where deltas[t][i] = +1 if node i
        spawns a new node at time step t, deltas[t][i] = -1 if node i is
        deleted at step t, and deltas[t][i] = 0 otherwise.
    Epsilons: list
        a time series of edge edit matrices, where Epsilons[t][i, j] = +1 if
        nodes i and j are newly connected at time t, Epsilons[t][i, j] = -1 if
        the egde (i, j) is removed at time t, and Epsilons[t][i, j] = 0
        otherwise.

    """
    if n_max is None:
        n_max = 4 * len(A)
    elif n_max < 4 * len(A):
        raise ValueError('n_max must be at least 4 * len(A) = %d to ensure that even after growth beyond its initial size the graph still is codeable.' % (4 * len(A)))

    # number the nodes in each connected component
    Cs = []
    remaining = set(range(len(A)))
    while(remaining):
        i = min(remaining)
        C = []
        stk = [i]
        while(stk):
            i = stk.pop()
            if(i not in remaining):
                continue
            remaining.remove(i)
            C.append(i)
            for j in np.where(A[i, :])[0]:
                if(j in remaining):
                    stk.append(j)
        Cs.append(np.array(C, dtype=int))
    As = [A]
    X  = np.eye(len(A), n_max)
    Xs = [X]
    deltas = []
    Epsilons = []
    t = 1
    while(t_max is None or t < t_max):
        edits, delta, Epsilon = next_step(A, n_max)
        m = len(A)
        if(len(edits) == 0):
            break
        # apply edits
        for edit in edits:
            A, X = edit.apply(A, X)
        As.append(A)
        Xs.append(X)
        deltas.append(delta)
        Epsilons.append(Epsilon)
        t += 1
    deltas.append(delta)
    Epsilons.append(Epsilon)
    return As, Xs, deltas, Epsilons

def generate_time_series_from_random_graph(n_max = None, t_max = 100):
    """ Generates a random, undirected, initial graph and lets it evolve
    until the graph converges or until the maximum number of steps is reached.

    Parameters
    ----------
    n_max: int (default = 4 * N)
        a reference number of dimensions for the node encoding, which will
        be a one-hot coding padded with zeros if necessary.
    t_max: int (default = 100)
        A maximum number of steps.

    Returns
    -------
    As: list
        A time series of graphs until the graph converges to a stable state
        or t_max steps have been done.
    Xs: list
        a time series of node attribute matrices. These are just one-hot coding
        vectors of the node id.
    deltas: list
        a time series of node edit vectors where deltas[t][i] = +1 if node i
        spawns a new node at time step t, deltas[t][i] = -1 if node i is
        deleted at step t, and deltas[t][i] = 0 otherwise.
    Epsilons: list
        a time series of edge edit matrices, where Epsilons[t][i, j] = +1 if
        nodes i and j are newly connected at time t, Epsilons[t][i, j] = -1 if
        the egde (i, j) is removed at time t, and Epsilons[t][i, j] = 0
        otherwise.

    """
    # randomly sample a graph from train dataset
    f = random.sample(train_graph_files, 1)[0]
    g = nx.read_gexf(f)
    A = nxgraph_to_adjlist(g, directed=False)

    # return evolution
    return generate_time_series(A, n_max = n_max, t_max = t_max)


if __name__ == "__main__":
    As, Xs, deltas, Epsilons = generate_time_series_from_random_graph(64, 100)
    # print(As)
    print(Xs)
    print(deltas)
    print(Epsilons)
