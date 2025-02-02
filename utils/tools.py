import numpy as np
import torch
import torch.nn as nn


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, hierarchy):
    A = []
    for i in range(len(hierarchy)):
        A.append(normalize_digraph(edge2mat(hierarchy[i], num_node)))

    A = np.stack(A)

    return A

def get_spatial_graph_original(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def get_graph(num_node, edges):

    I = edge2mat(edges[0], num_node)
    Forward = normalize_digraph(edge2mat(edges[1], num_node))
    Reverse = normalize_digraph(edge2mat(edges[2], num_node))
    A = np.stack((I, Forward, Reverse))
    return A # 3, 25, 25

def get_hierarchical_graph(num_node, edges):
    A = []
    for edge in edges:
        A.append(get_graph(num_node, edge))
    A = np.stack(A)
    return A

def get_groups(dataset='human3.6m', CoM=21):
    groups  =[]
    
    if dataset == 'NTU':
        if CoM == 2:
            groups.append([2])
            groups.append([1, 21])
            groups.append([13, 17, 3, 5, 9])
            groups.append([14, 18, 4, 6, 10])
            groups.append([15, 19, 7, 11])
            groups.append([16, 20, 8, 12])
            groups.append([22, 23, 24, 25])

        ## Center of mass : 21
        elif CoM == 21:
            groups.append([21])
            groups.append([2, 3, 5, 9])
            groups.append([4, 6, 10, 1])
            groups.append([7, 11, 13, 17])
            groups.append([8, 12, 14, 18])
            groups.append([22, 23, 24, 25, 15, 19])
            groups.append([16, 20])

        ## Center of Mass : 1
        elif CoM == 1:
            groups.append([1])
            groups.append([2, 13, 17])
            groups.append([14, 18, 21])
            groups.append([3, 5, 9, 15, 19])
            groups.append([4, 6, 10, 16, 20])
            groups.append([7, 11])
            groups.append([8, 12, 22, 23, 24, 25])

        else:
            raise ValueError()
    if dataset == 'human3.6m':
        if CoM == 0:
            groups.append([0])
            groups.append([1, 5, 10, 12, 17])
            groups.append([2, 6, 11, 13, 18])
            groups.append([3, 7, 14, 19])
            groups.append([4, 8, 15, 16, 20, 21])

        if CoM == 9:
            groups.append([9])
            groups.append([1, 5, 10, 12, 17])
            groups.append([2, 6, 11, 13, 18])
            groups.append([3, 7, 14, 19])
            groups.append([4, 8, 15, 16, 20, 21])
        
    return groups

def get_edgeset(dataset='NTU', CoM=21):
    groups = get_groups(dataset=dataset, CoM=CoM)
    
    # for i, group in enumerate(groups):
    #     group = [i - 1 for i in group]
    #     groups[i] = group

    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []

    # for i in range(len(groups) - 1):
    #     self_link = groups[i] + groups[i + 1]
    #     self_link = [(i, i) for i in self_link]
    #     identity.append(self_link)
    #     forward_g = []
    #     for j in groups[i]:
    #         for k in groups[i + 1]:
    #             forward_g.append((j, k))
    #     forward_hierarchy.append(forward_g)
    #
    #     reverse_g = []
    #     for j in groups[-1 - i]:
    #         for k in groups[-2 - i]:
    #             reverse_g.append((j, k))
    #     reverse_hierarchy.append(reverse_g)
    #
    edges = []
    # for i in range(len(groups) - 1):
    #     edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])
    for i in range(len(groups)-1):
        g = []
        for j in groups[i]:
            for k in groups[i+1]:
                g.append((j, k))
        edges.append(g)

    return edges

class Graph():
    def __init__(self, node_num, dataset='human3.6m'):
        self.node_num = node_num
        self.dataset = dataset
        self.get_edge()

    def get_edge(self):
        self_link = [(i, i) for i in range(22)]
        bone_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
                     (9, 17), (17, 18), (18, 19), (19, 20), (20, 21), (9, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
        if self.dataset == 'human3.6m':
            larm_rleg = [(13, 2), (14, 2)]
            larm_lleg = [(13, 7), (14, 7)]
            rarm_rleg = [(18, 2), (19, 2)]
            rarm_lleg = [(18, 7), (19, 7)]
            trunk_lleg = [(0, 7), (9, 7)]
            trunk_rleg = [(0, 2), (9, 2)]
            trunk_larm = [(0, 13), (9, 13)]
            trunk_rarm = [(0, 18), (9, 18)]
            lrarm = [(13, 18), (14, 18)]
            lrleg = [(7, 2), (8, 3)]
        if self.dataset == 'cmu':
            larm_rleg = [(17, 7), (16, 7)]
            larm_lleg = [(17, 2), (16, 2)]
            rarm_rleg = [(21, 7), (22, 7)]
            rarm_lleg = [(21, 2), (22, 2)]
            trunk_lleg = [(0, 2), (11, 2)]
            trunk_rleg = [(0, 7), (11, 7)]
            trunk_larm = [(0, 17), (11, 17)]
            trunk_rarm = [(0, 22), (11, 22)]
            lrarm = [(17, 21), (16, 21), (17, 22), (16, 22)]
            lrleg = [(3, 8), (2, 7)]
        if self.dataset == '3dpw':
            larm_rleg = [(17, 4), (19, 4)]
            larm_lleg = [(17, 3), (19, 3)]
            rarm_rleg = [(18, 4), (20, 4)]
            rarm_lleg = [(18, 3), (20, 3)]
            trunk_lleg = [(2, 3), (8, 3)]
            trunk_rleg = [(2, 4), (8, 4)]
            trunk_larm = [(2, 17), (8, 17)]
            trunk_rarm = [(2, 18), (8, 18)]
            lrarm = [(17, 18), (19, 18), (17, 20), (19, 20)]
            lrleg = [(3, 4), (6, 7)]
        down_lamb = [(0, 1), (0, 5), (0, 2), (0, 6), (0, 9), (1, 9), (5, 9)]

        edges = get_edgeset('human3.6m', 0)

        bone = bone_link #self_link +
        self.edge = [larm_rleg, larm_lleg, rarm_rleg, rarm_lleg, trunk_lleg, trunk_rleg, trunk_larm,
                     trunk_rarm, lrarm, lrleg]
        self.adj = []

        # self.pre_sem_edge = [(2, 7), (3, 8), (16, 21), (17, 22)]
        A_ske = torch.zeros((self.node_num, self.node_num))
        # for m in edges:
        for m in self.edge:
            count=0
            A_ske = torch.zeros((self.node_num, self.node_num))
            for i, j in m:
                for m in range(3):
                    for n in range(3):
                        count+=1
                        A_ske[3*j+m, 3*i+n] = 1
                        A_ske[3*i+n, 3*j+m] = 1
                        # if count==6:
            self.adj.append(A_ske.cuda())
        self.A_ske = A_ske
        # A_pre_sem = torch.zeros((self.node_num, self.node_num))
        # for p, q in self.pre_sem_edge:
        #     A_pre_sem[p, q] = 1
        #     A_pre_sem[q, p] = 1
        # self.A_pre_sem = A_pre_sem

        return self.adj
def Enhance(weight, tensor):
    """
    Transform the input tensor by converting 0s to 1s and multiplying 1s by the given weight.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    weight (float): The weight to multiply with elements that are 1.

    Returns:
    torch.Tensor: The transformed tensor.
    """
    transformed_tensor = tensor.clone()  # Create a copy of the tensor to avoid modifying the original tensor
    # Multiply 1s by the weight
    transformed_tensor[transformed_tensor == 1] *= weight
    # Convert 0s to 1s
    transformed_tensor[transformed_tensor == 0] = 1
    return transformed_tensor


if __name__ == '__main__':
    pre = Graph(66).get_edge()
    print()
    # larm_rleg = [(12, 1), (12, 2), (12, 3), (13, 1), (13, 2), (13, 3), (14, 1), (14, 2), (14, 3)]
    # larm_lleg = [(12, 6), (12, 7), (12, 8), (13, 6), (13, 7), (13, 8), (14, 6), (14, 7), (14, 8)]
    # rarm_rleg = [(17, 1), (17, 2), (17, 3), (18, 1), (18, 2), (18, 3), (19, 1), (19, 2), (19, 3)]
    # rarm_lleg = [(17, 6), (17, 7), (17, 8), (18, 6), (18, 7), (18, 8), (19, 6), (19, 7), (19, 8)]
    # trunk_lleg = [(0, 6), (0, 7), (0, 8), (9, 6), (9, 7), (9, 8), (10, 6), (10, 7), (10, 8), (11, 6), (11, 7), (11, 8)]
    # trunk_rleg = [(0, 1), (0, 2), (0, 3), (9, 1), (9, 2), (9, 3), (10, 1), (10, 2), (10, 3), (11, 1), (11, 2), (11, 3)]
    # trunk_larm = [(0, 12), (0, 13), (0, 14), (9, 12), (9, 13), (9, 14), (10, 12), (10, 13), (10, 14), (11, 12),
    #               (11, 13), (11, 14)]
    # trunk_rarm = [(0, 17), (0, 18), (0, 19), (9, 17), (9, 18), (9, 19), (10, 17), (10, 18), (10, 19), (11, 17),
    #               (11, 18), (11, 19)]
    # lrarm = [(12, 17), (12, 18), (12, 19), (13, 17), (13, 18), (13, 19), (14, 17), (14, 18), (14, 19)]
    # lrleg = [(6, 1), (7, 2), (8, 3), (6, 1), (7, 2), (8, 3), (6, 1), (7, 2), (8, 3)]
    # down_lamb = [(0, 1), (0, 5), (0, 2), (0, 6), (0, 9), (1, 9), (5, 9)]

    # larm_rleg = [(13, 2), (13, 3), (14, 2), (14, 3)]
    # larm_lleg = [(13, 7), (13, 8), (14, 7), (14, 8)]
    # rarm_rleg = [(18, 2), (18, 3), (19, 2), (19, 3)]
    # rarm_lleg = [(18, 7), (18, 8), (19, 7), (19, 8)]
    # trunk_lleg = [(0, 7), (0, 8), (9, 7), (9, 8)]
    # trunk_rleg = [(0, 2), (0, 3), (9, 2), (9, 3)]
    # trunk_larm = [(0, 13), (0, 14), (9, 13), (9, 14)]
    # trunk_rarm = [(0, 18), (0, 19), (9, 18), (9, 19)]
    # lrarm = [(13, 18), (13, 19), (14, 18), (14, 19)]
    # lrleg = [(7, 2), (8, 3)]
    # down_lamb = [(0, 1), (0, 5), (0, 2), (0, 6), (0, 9), (1, 9), (5, 9)]