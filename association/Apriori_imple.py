def get_items_first(dataset):
    items = []
    for e in dataset:
        for i in e:
            if [i] not in items:
                items.append([i])
    items.sort()
    return items


def get_items(dataset):
    items = []
    for i in range(len(dataset)):
        for j in range(i+1, len(dataset)):
            dataset[i].sort()
            dataset[j].sort()
            if dataset[i][:-2] != dataset[j][:-2]:
                continue
            if len(dataset[i]) == 1:
                new = set()
            else:
                new = set(dataset[i]) ^ set(dataset[j])
            for e in dataset:
                item = list(set(dataset[i]) | set(dataset[j]))
                if new.issubset(set(e)) and item not in items:
                    items.append(item)
    items.sort()
    return items


def get_frequent_items(dataset, items, min_support):
    total_times = len(dataset)
    support_dict = {}
    for i in items:
        for e in dataset:
            if set(i).issubset(set(e)):
                support_dict[tuple(i)] = support_dict.get(
                    tuple(i), 0)+1.0/total_times

    frequent_items = []
    pop_keys = []
    for k, v in support_dict.items():
        if v < min_support:
            pop_keys.append(k)
            continue
        frequent_items.append(list(k))
    for k in pop_keys:
        support_dict.pop(k)
    return frequent_items, support_dict


def Apriori(dataset, min_support):
    frequent_items = []
    support_dict = {}

    current_items = get_items_first(dataset)
    current_frequent_items, current_support_dict = get_frequent_items(
        dataset, current_items, min_support)
    frequent_items.append(current_frequent_items)
    support_dict.update(current_support_dict)

    while len(current_frequent_items) > 1:
        current_items = get_items(current_frequent_items)
        current_frequent_items, current_support_dict = get_frequent_items(
            dataset, current_items, min_support)
        frequent_items.append(current_frequent_items)
        support_dict.update(current_support_dict)

    return frequent_items, support_dict


if __name__ == "__main__":
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    frequent_items, support_dict = Apriori(dataset, min_support=0.4)

    print("具有关联的商品是{}".format(frequent_items))
    print('------------------')
    print("对应的支持度为{}".format(support_dict))
