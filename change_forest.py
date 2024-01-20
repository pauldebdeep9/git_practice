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
        self.ancestors = [None] * np.shape(data)[0]
        self.inode_depth = []  # depth in tree of each internal node, in order of nodeid
        self.root = self.make_tree(None, np.arange(len(data), dtype=np.int32), 0)

    def make_tree(self, parent, idx, depth):
        if len(idx) == 1 or depth == self.lim:
            cur = parent
            cur_ancestors = []
            while cur is not None:
                cur_ancestors.append(cur.nodeid)
                cur = cur.parent
            for i in idx:
                self.ancestors[i] = cur_ancestors
            return Node(parent, idx, depth, None)
        elif len(idx) > 1:
            nodeid = len(self.inode_depth)
            self.inode_depth.append(depth)
            nfeatures = np.shape(self.data)[1]
            q = rn.randint(0, nfeatures - 1)
            cur = self.data[idx, q]

            a = cur.min()
            b = cur.max()
            p = (rn.random()+rn.random())/2 * (b-a) + a

            w = np.where(cur < p, True, False)
            cur_node = Node(parent, idx, depth, nodeid)
            self.make_tree(cur_node, idx[w], depth + 1)
            self.make_tree(cur_node, idx[~w], depth + 1)
            return cur_node


class Node(object):
    def __init__(self, parent, idx, depth, nodeid):
        self.parent = parent
        self.depth = depth
        self.nodeid = nodeid
