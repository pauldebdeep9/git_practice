import numpy as np
import random as rn


class CForest(object):
    def __init__(self, data, ntrees, lim):
        self.ntrees = ntrees
        self.data = data
        self.lim = lim
        self.trees = []
        for i in range(ntrees):
            self.trees.append(CTree(data, lim))


class CTree(object):
    def __init__(self, data, lim):
        self.data = data
        self.lim = lim
        self.ancestors = [None] * (2**(1+lim))
        self.node_depth = []  # depth in tree of each internal node, in order of nodeid
        init_bounds = np.array([[data[:, q].min(), data[:, q].max()] for q in range(data.shape[1])])
        # print(init_bounds)
        self.root = self.make_tree(None, np.arange(len(data), dtype=np.int32), 0, init_bounds)

    def make_tree(self, parent, idx, depth, bounds):
        nodeid = len(self.node_depth)
        self.node_depth.append(depth)
        if depth == self.lim or len(idx) <= 1:
            cur = parent
            cur_ancestors = []
            while cur is not None:
                cur_ancestors.append(cur.nodeid)
                cur = cur.parent
            self.ancestors[nodeid] = cur_ancestors
            return Node(parent, depth, nodeid, None, None)
        else:
            nfeatures = self.data.shape[1]
            splitvar = rn.randint(0, nfeatures - 1)
            cur = self.data[idx, splitvar]

            samp = rn.sample(list(cur), 2)
            split = .5 * (rn.uniform(samp[0], samp[1]) + rn.uniform(samp[0], samp[1]))
            w = np.where(cur < split, True, False)

            cur_node = Node(parent, depth, nodeid, splitvar, split)

            bounds_left = bounds.copy()
            bounds_right = bounds.copy()
            bounds_left[splitvar, 1] = split
            bounds_right[splitvar, 0] = split

            cur_node.left = self.make_tree(cur_node, idx[w], depth + 1, bounds_left)
            cur_node.right = self.make_tree(cur_node, idx[~w], depth + 1, bounds_right)
            return cur_node

    def eval_point(self, point):
        # print('eval:', point)
        cur = self.root
        while cur.left is not None:
            cur = cur.left if point[cur.splitvar] < cur.split else cur.right
        return cur.nodeid


class Node(object):
    def __init__(self, parent, depth, nodeid, splitvar, split):
        self.parent = parent
        self.depth = depth
        self.nodeid = nodeid
        self.splitvar = splitvar
        self.split = split
        self.left = None
        self.right = None
