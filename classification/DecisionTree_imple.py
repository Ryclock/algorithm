import numpy as np
from collections import Counter
import pandas as pd
from pathlib import Path
import os


class Node:
    def __init__(self, feature=None, entropy=None, classification=None, sub_nodes=None):
        self.feature = feature
        self.entropy = entropy
        self.classification = classification
        self.sub_nodes = sub_nodes

    def predict(self, x):
        if self.sub_nodes is None:
            return self.classification
        else:
            x_new = np.delete(x, self.feature)
            return self.sub_nodes[x[self.feature]].predict(x_new)


def call_entropy(D):
    count = np.array(list(Counter(D[:, -1]).values()))
    p = count / sum(count)
    return -sum(p * np.log2(p))


def call_info_gain(D, A):
    entropy_D = call_entropy(D)
    uniq_A = np.unique(A)
    entropy_DA = 0
    for a in uniq_A:
        D_a = D[A == a]
        entropy_Da = call_entropy(D_a)
        entropy_DA += entropy_Da * (len(D_a) / len(D))
    return entropy_D - entropy_DA


def id3(D, A):
    feature_idx = np.argmax([call_info_gain(D, A[:, i])
                            for i in range(len(A[0]))])
    uniq_vals = np.unique(A[:, feature_idx])
    node = Node(feature_idx)
    sub_nodes = {}
    for val in uniq_vals:
        D_val = D[A[:, feature_idx] == val]
        if len(D_val) == 0:
            sub_nodes[val] = Node(None, 0, Counter(
                D[:, -1]).most_common(1)[0][0])
        elif len(np.unique(D_val[:, -1])) == 1:
            sub_nodes[val] = Node(None, 0, D_val[0, -1])
        else:
            sub_nodes[val] = id3(D_val, np.delete(
                A[A[:, feature_idx] == val], feature_idx, axis=1))
    node.sub_nodes = sub_nodes
    return node


def call_gr(D, A):
    entropy_D = call_entropy(D)
    uniq_A = np.unique(A)
    entropy_DA = 0
    iv_A = 0
    for a in uniq_A:
        D_a = D[A == a]
        entropy_Da = call_entropy(D_a)
        entropy_DA += entropy_Da * (len(D_a) / len(D))
        iv_A -= (len(D_a) / len(D)) * np.log2(len(D_a) / len(D))
    if iv_A == 0:
        return float('inf')
    else:
        return (entropy_D - entropy_DA) / iv_A


def c45(D, A, delta=0):
    feature_idx = np.argmax([call_gr(D, A[:, i]) for i in range(len(A[0]))])
    info_gain_ratio = call_gr(D, A[:, feature_idx])
    uniq_vals = np.unique(A[:, feature_idx])
    node = Node(feature_idx)
    sub_nodes = {}
    for val in uniq_vals:
        D_val = D[A[:, feature_idx] == val]
        if len(D_val) == 0:
            sub_nodes[val] = Node(None, 0, Counter(
                D[:, -1]).most_common(1)[0][0])
        elif len(np.unique(D_val[:, -1])) == 1:
            sub_nodes[val] = Node(None, 0, D_val[0, -1])
        elif info_gain_ratio < delta:
            sub_nodes[val] = Node(None, 0, Counter(
                D_val[:, -1]).most_common(1)[0][0])
        else:
            sub_nodes[val] = c45(D_val, np.delete(
                A[A[:, feature_idx] == val], feature_idx, axis=1))
    node.sub_nodes = sub_nodes
    return node


if __name__ == "__main__":
    parent_path = Path(__file__).parents[1]
    paths = ['data_sets', 'watermelon']
    df = pd.read_csv(filepath_or_buffer=os.path.join(
        parent_path, *paths, 'data.csv'), index_col=0)
    # node = id3(np.array(df), np.array(df.drop(columns=['好瓜'])))
    node = c45(np.array(df), np.array(df.drop(columns=['好瓜'])))

    # ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    res = node.predict(['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'])
    print(res)
    res = node.predict(['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑'])
    print(res)
