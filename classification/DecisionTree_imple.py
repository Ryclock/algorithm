import pandas as pd
import math
from pathlib import Path
import os


def entropy(data):
    label_counts = {}
    for row in data:
        key = row[-1]
        label_counts[key] = label_counts.get(key, 0)+1
    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / len(data)
        entropy -= prob * math.log(prob, 2)
    return entropy


def info_gain(data, feature_idx):
    feature_counts = {}
    for row in data:
        key = row[feature_idx]
        feature_counts[key] = feature_counts.get(key, 0)+1
    info_gain = entropy(data)
    for key in feature_counts:
        prob = float(feature_counts[key]) / len(data)
        subset = [row for row in data if row[feature_idx] == key]
        info_gain -= prob * entropy(subset)
    return info_gain


def choose_best_feature(data, features):
    '''
        features: only be used to print information
    '''
    best_feature_idx = -1
    best_info_gain = 0.0
    for feature_idx in range(len(data[0]) - 1):
        info_gain_current = info_gain(data, feature_idx)
        print('IG({}): {}'.format(
            features[feature_idx], info_gain_current), file=output_file)
        if info_gain_current <= best_info_gain:
            continue
        best_info_gain = info_gain_current
        best_feature_idx = feature_idx
    return best_feature_idx


def majority_label(label_list):
    return max(label_list, key=label_list.count)


def get_sub_data(data, feature_idx, value):
    return [row[:feature_idx]+row[feature_idx+1:]
            for row in data if row[feature_idx] == value]


def create_tree(data, features, floor):
    '''
        floor: only be used to print information
    '''
    print('Floor: {}'.format(floor), file=output_file)
    label_list = [row[-1] for row in data]
    if label_list.count(label_list[0]) == len(label_list):
        print('unique label: {}'.format(label_list[0]), file=output_file)
        return label_list[0]
    if len(data[0]) == 1:
        return majority_label(label_list)
    best_feature_idx = choose_best_feature(data, features)
    best_feature = features[best_feature_idx]
    print('Chosen(best): {}'.format(best_feature), file=output_file)
    tree = {best_feature: {}}
    sub_features = features[:best_feature_idx]+features[best_feature_idx+1:]
    for value in set([row[best_feature_idx] for row in data]):
        print('\nNow analysize value: {}'.format(value), file=output_file)
        sub_data = get_sub_data(data, best_feature_idx, value)
        tree[best_feature][value] = create_tree(
            sub_data, sub_features, floor+1)
    return tree


def predict(tree, features, sample):
    root_label = list(tree.keys())[0]
    root_node = tree[root_label]
    feature_idx = features.index(root_label)
    for key in root_node:
        if sample[feature_idx] != key:
            continue
        if not isinstance(root_node[key], dict):
            return root_node[key]
        return predict(root_node[key], features, sample)


if __name__ == "__main__":
    parent_path = Path(__file__).parents[1]
    paths = ['data_sets', 'watermelon']
    df = pd.read_csv(filepath_or_buffer=os.path.join(
        parent_path, *paths, 'data.csv'), index_col=0)
    features = df.columns.values.tolist()
    output_file = open(file="output.txt", mode='w+', encoding='utf-8')
    print('DFS型输出\n', file=output_file)
    tree = create_tree(df.values.tolist(), features, 1)
    output_file.close()
    # res = predict(tree, ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜'],
    #               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'])
    # print(res)