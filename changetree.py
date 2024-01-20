from __future__ import division
import numpy as np
import change_forest
import online_forest
from collections import Counter
import util


def change_score(X, ntrees=50, w=15, lim=15):
    n = len(X)
    print('starting change detection')
    forest = change_forest.CForest(data=X, ntrees=ntrees, lim=lim)
    print('finishing learning forest')
    max_depth = np.full((ntrees, n), np.nan)
    for tree_idx in range(ntrees):
        tree = forest.trees[tree_idx]
        left_sum = Counter()
        right_sum = Counter()
        for t in range(w):
            left_sum += Counter(tree.ancestors[t])
            right_sum += Counter(tree.ancestors[t + w])
        overlaps = set([i for i in left_sum if i in right_sum])
        max_depth[tree_idx, w] = max([tree.inode_depth[i] for i in overlaps])

        left_sum = [left_sum[i] for i in range(len(tree.inode_depth))]
        right_sum = [right_sum[i] for i in range(len(tree.inode_depth))]

        for t in range(w, n - w):
            for i in tree.ancestors[t]:
                left_sum[i] += 1
                if left_sum[i] == 1 and right_sum[i] > 0:
                    overlaps.add(i)
            for i in tree.ancestors[t - w]:
                left_sum[i] -= 1
                if left_sum[i] == 0 and right_sum[i] > 0:
                    overlaps.remove(i)
            for i in tree.ancestors[t + w]:
                right_sum[i] += 1
                if right_sum[i] == 1 and left_sum[i] > 0:
                    overlaps.add(i)
            for i in tree.ancestors[t]:
                right_sum[i] -= 1
                if right_sum[i] == 0 and left_sum[i] > 0:
                    overlaps.remove(i)
            max_depth[tree_idx, t + 1] = max([tree.inode_depth[i] for i in overlaps])

    print('finished scoring')
    score = 1 + lim - np.mean(max_depth, axis=0)
    peaks_sorted = util.sortedpeaks(score)
    return score, peaks_sorted


def change_score_online(X, ntrees=50, w=15, lim=15):
    n = len(X)
    print('starting change detection')
    forest = online_forest.CForest(data=X[0:min(250, len(X)), :], ntrees=ntrees, lim=lim)
    print('finishing learning forest')
    max_depth = np.full((ntrees, n), np.nan)
    for tree_idx in range(ntrees):
        tree = forest.trees[tree_idx]
        assigned_leaf = [tree.eval_point(X[t, :]) for t in range(len(X))]
        test_ancestors = [tree.ancestors[x] for x in assigned_leaf]

        left_sum = Counter()
        right_sum = Counter()

        for t in range(w):
            left_sum += Counter(test_ancestors[t])
            right_sum += Counter(test_ancestors[t + w])
        overlaps = set([i for i in left_sum if i in right_sum])
        max_depth[tree_idx, w] = max([tree.node_depth[i] for i in overlaps])

        left_sum = [left_sum[i] for i in range(len(tree.node_depth))]
        right_sum = [right_sum[i] for i in range(len(tree.node_depth))]

        for t in range(w, n - w):
            for i in test_ancestors[t]:
                left_sum[i] += 1
                if left_sum[i] == 1 and right_sum[i] > 0:
                    overlaps.add(i)
            for i in test_ancestors[t - w]:
                left_sum[i] -= 1
                if left_sum[i] == 0 and right_sum[i] > 0:
                    overlaps.remove(i)
            for i in test_ancestors[t + w]:
                right_sum[i] += 1
                if right_sum[i] == 1 and left_sum[i] > 0:
                    overlaps.add(i)
            for i in test_ancestors[t]:
                right_sum[i] -= 1
                if right_sum[i] == 0 and left_sum[i] > 0:
                    overlaps.remove(i)
            max_depth[tree_idx, t + 1] = max([tree.node_depth[i] for i in overlaps])

    print('finished scoring')
    score = 1 + lim - np.mean(max_depth, axis=0)
    peaks_sorted = util.sortedpeaks(score)
    return score, peaks_sorted

