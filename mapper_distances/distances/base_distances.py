"""
Contains base distances between nodes, namely:

Hausdorff distance (the maximum closest point distance)
Intersection distance (Minimum point to point distance - probably has a formal name)
Jaccard - for nodes that have perfect overlap, we can examine their Jaccard coeficient
"""

import numpy as np
from numba import jit


def network_merge_distance(net1, net2, metric_space):
    """ Calculates pairwise Hausdorff distances between nodes
    in the two networks.

    Parameters
    ----------
    net1 : lightweight_mapper.Network
        Mapper graph
    net2 : lightweight_mapper.Network
        Mapper graph
    metric_space : np.array
        Pairwise distance matrix

    Returns
    -------
    np.array - nxm
        Pairwise distance matrix between networks
    """
    net_clust1 = net1.export_clustering_as_cover()
    net_clust2 = net2.export_clustering_as_cover()

    n = len(net_clust1.partitions_)
    m = len(net_clust2.partitions_)

    merge_dists = np.zeros((n, m))

    for i, p1 in enumerate(net_clust1.partitions_):
        for j, p2 in enumerate(net_clust2.partitions_):
            merge_dists[i, j] = node_mutual_merge_distance_opt(p1.members_, p2.members_, metric_space)

    return merge_dists

@jit(nopython=True)
def node_merge_distances_opt(node1, node2, metric_space):
    """ Returns the size of the neighborhood expansion that would
    be required for node1 to subsume node2.
    
    Parameters
    ----------
    node1 : set[int]
        Set of integers corresponding to points in the space contained by the node
    node2 : set[int]
        Set of integers corresponding to points in the space contained by the node
    metric_space : np.array
        n x n array containing pairwise distances in the metric space.
    Returns
    -------
    dist : float
    """
    node_diff = node2.difference(node1)
    if len(node_diff) == 0:
        return 0.0
    max_val = 0.0
    for n2 in node_diff:
        min_val = np.inf
        for n1 in node1:
            min_val = min(min_val, metric_space[n1, n2])
        max_val = max(max_val, min_val)
    return max_val


@jit(nopython=True)
def node_mutual_merge_distance_opt(node1, node2, metric_space):
    """ Returns the size of larger of the neighborhood expansions that
    would be required node1 to subsume node2 or vice versa.

    This is the Hausdorff distance for the nodes.
    
    Parameters
    ----------
    node1 : set[int]
        Set of integers corresponding to points in the space contained by the node
    node2 : set[int]
        Set of integers corresponding to points in the space contained by the node
    metric_space : np.array
        n x n array containing pairwise distances in the metric space.
    Returns
    -------
    dist : float
    """
    return max(node_merge_distances_opt(node1, node2, metric_space),
               node_merge_distances_opt(node2, node1, metric_space))


def node_intersection_distance(node1, node2, metric_space):
    """ Returns the minimum distance between points in a pair of
    subsets.

    From a Mapper perspective, this can be seen as the minimum neighborhood
    expansion that would result in an edge forming between the two points.

    Parameters
    ----------
    node1 : set[int]
        Set of integers corresponding to points in the space contained by the node
    node2 : set[int]
        Set of integers corresponding to points in the space contained by the node
    metric_space : np.array
        n x n array containing pairwise distances in the metric space.
    Returns
    -------
    dist : float
    """
    if node1.intersection(node2):
        return 0

    node1 = list(node1)
    node2 = list(node2)
    sub_dist = metric_space[node1][:, node2]

    return sub_dist.min()

def separation(node1, metric_space):
    """ Returns the separation of the subset of the metric
    space defined by node1 (the minimum non-zero separating space)
    """
    if len(node1) <= 1:
        return 0
    node1 = list(node1)
    sub_dist = metric_space[node1][:, node1]
    sub_dist[sub_dist == 0] = np.inf
    sdmin = sub_dist.min()
    if sdmin == np.inf:
        return 0
    return sdmin

def diameter(node1, metric_space):
    """ Returns the diameter of the subset of the metric space
    defined by node1 (the maximum distance)
    """
    if len(node1) <= 1:
        return 0
    node1 = list(node1)
    sub_dist = metric_space[node1][:, node1]
    np.fill_diagonal(sub_dist, -np.inf)
    return sub_dist.max()

def nodewise_jaccard(net1, net2):
    inter = net1.node_row_matrix.toarray().dot(net2.node_row_matrix.toarray().T)
    d1 = np.diag(net1.adjacency_matrix.toarray())
    d2 = np.diag(net2.adjacency_matrix.toarray())
    union = np.add.outer(d1, d2) - inter
    return inter/union

