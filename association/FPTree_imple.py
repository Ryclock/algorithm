import collections
import itertools


class FPNode:
    def __init__(self, count=0, str=None):
        self.count = count
        self.str = str
        self.children = collections.defaultdict(FPNode)
        self.parent = None
        self.next = None


class FPTree:
    def __init__(self, min_support):
        self.root = FPNode(-1, 'root')
        self.header_table_count = collections.defaultdict(int)
        self.header_table_link = {}
        self.min_support = min_support
        self.size = 0
        self.frequent_items = collections.defaultdict(float)

    def preprocess(self, data):
        data = self.uniq_data(data)
        self.size = len(data)
        for line in data:
            for item in line:
                self.header_table_count[item] += 1
        data = self.frequent_data(data)
        pop_list = []
        for x in self.header_table_count.items():
            if x[1] >= self.min_support*self.size:
                continue
            pop_list.append(x[0])
        for i in pop_list:
            self.header_table_count.pop(i)
        data = self.sort_data(data)
        return data

    def uniq_data(self, data):
        for i in range(len(data)):
            data[i] = list(set(data[i]))
        return data

    def frequent_data(self, data):
        for i in range(len(data)):
            data[i] = [item for item in data[i]
                       if self.header_table_count[item] >= self.min_support*self.size]
        return data

    def sort_data(self, data):
        count_list = sorted(self.header_table_count.keys(
        ), key=lambda k: self.header_table_count[k], reverse=True)
        for i in range(len(data)):
            data[i] = sorted(
                data[i], key=lambda x: count_list.index(x))
            print(data[i])
        return data

    def bulid_tree(self, data):
        data = self.preprocess(data)
        for line in data:
            root = self.root
            for item in line:
                root.children[item].count += 1
                if not root.children[item].str:
                    root.children[item].str = item
                if not root.children[item].parent:
                    root.children[item].parent = root
                root = root.children[item]
                if item not in self.header_table_link.keys():
                    self.header_table_link[item] = root
                    continue
                if root == self.header_table_link[item]:
                    continue
                root.next = self.header_table_link[item]
                self.header_table_link[item] = root
        return self.root

    def is_single_path(self, root: FPNode):
        if not root or not root.children:
            return True
        if len(root.children) > 1:
            return False
        for _, node in root.children.items():
            if not self.is_single_path(node):
                return False
        return True

    # def FP_growth(self, root: FPNode, frequent_items):
    def FP_growth(self):
        # size = sum([count for _, count in self.header_table_count.items()])
        # if self.is_single_path(root):
        #     tmp = []
        #     while root.children:
        #         for _, node in root.children.items():
        #             tmp.append(node.str, node.count)
        #             root = node
        #     ans = [list(itertools.combinations(tmp, i)
        #                 for i in range(1, len(tmp)+1))]
        #     for items in ans:
        #         count = min([item[1] for item in items])
        #         if count < self.min_support*size:
        #             continue
        #         new_frequent_items = frozenset(
        #             [item[0]for item in items]+frequent_items)
        #         self.frequent_items[new_frequent_items] = count * 1.0/size
        #     return

        header_table_count = sorted(
            self.header_table_count.items(), key=lambda x: x[1], reverse=True)
        for item, count in header_table_count:
            if count < self.min_support*self.size:
                continue
            self.frequent_items[frozenset([item])] = count * 1.0/self.size
            node = self.header_table_link[item]
            condition_bases_with_count = []
            count_item = collections.defaultdict(int)
            while node:
                node_cur = node
                condition_base = [[], node.count]
                while node_cur.parent:
                    condition_base[0].append(node_cur)
                    count_item[node_cur.str] += condition_base[1]
                    node_cur = node_cur.parent
                condition_base[0] = condition_base[0][::-1]
                condition_bases_with_count.append(condition_base)
                node = node.next
            for condition_base, count in condition_bases_with_count:
                new_condition_base = []
                for node in condition_base[:-1]:
                    if count_item[node.str] < self.min_support*self.size:
                        continue
                    new_condition_base.append(node.str)
                if len(new_condition_base) == 0:
                    continue
                for i in range(1, len(new_condition_base)+1):
                    for pre_condition_base in itertools.combinations(new_condition_base, i):
                        self.frequent_items[frozenset(
                            list(pre_condition_base)+[item])] += count*1.0/self.size


if __name__ == "__main__":
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

    fp = FPTree(0.6)
    fp.bulid_tree(dataset)
    fp.FP_growth()
    print(sorted(fp.frequent_items.items(),
          key=lambda x: x[1], reverse=True))
