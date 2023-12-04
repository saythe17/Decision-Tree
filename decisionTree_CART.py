import pandas as pd
import numpy as np
import sys


# tree node
class Node:
    def __init__(self, feature=None, branch1=None, value1=None, branch2=None, value2=None, label=None):
        self.feature = feature
        self.value1 = value1
        self.value2 = value2
        self.branch1 = branch1
        self.branch2 = branch2
        self.label = label

    def _feature(self):
        return self.feature

    def _value1(self):
        return self.value1

    def _value2(self):
        return self.value2

    def _branch1(self):
        return self.branch1

    def _branch2(self):
        return self.branch2

    def _label(self):
        return self.label


class decisionTree:
    def __init__(self, train_data, max_depth=0):
        self.value1 = None
        self.value2 = None
        self.feature_name = train_data.columns.tolist()
        self.root = None
        self.max_depth = max_depth
        self.train_data = train_data
        self.train()

    def train(self):
        # divide train_data into X and y
        X = self.train_data.iloc[:, :-1].values
        y = self.train_data.iloc[:, -1].values
        labels = sorted(np.unique(y))
        self.value1 = labels[0]
        self.value2 = labels[1]
        # create root
        self.root = self.build(0, X, y)

    def build(self, cur_depth, X, y):
        # count sample numbers of each label
        count1 = np.count_nonzero(y == self.value1)
        count2 = np.count_nonzero(y == self.value2)
        print('[', count1, self.value1, '/', count2, self.value2, ']')
        # if over the depth
        if cur_depth == self.max_depth:
            if count1 == count2:
                # choose label last in the lexicographical order
                label = max(self.value1, self.value2)
            elif count1 > count2:
                label = self.value1
            else:
                label = self.value2
            # return leaf node with a major label
            return Node(None, None, None, None, None, label)
        # if all sample belongs to one kind
        if count1 == 0:
            return Node(None, None, None, None, None, self.value2)
        if count2 == 0:
            return Node(None, None, None, None, None, self.value1)
        # choose feature to spilt
        num = self.choose_feature(X, y)
        # no feature can be spilt
        if num == -1:
            if count1 == count2:
                label = max(self.value1, self.value2)
            elif count1 > count2:
                label = self.value1
            else:
                label = self.value2
            return Node(None, None, None, None, None, label)
        # spilt
        else:
            new_feature = self.feature_name[num]
            value_range = sorted(np.unique(X[:, num]))
            if len(value_range) == 1:
                if count1 == count2:
                    label = max(self.value1, self.value2)
                elif count1 > count2:
                    label = self.value1
                else:
                    label = self.value2
                return Node(num, None, None, None, None, label)
            mask1 = X[:, num] == value_range[0]
            mask2 = X[:, num] == value_range[1]
            X1 = X[mask1]
            X2 = X[mask2]
            y1 = y[mask1]
            y2 = y[mask2]
            print("| " * (cur_depth + 1), end='')
            print(new_feature, '=', value_range[0], ': ', end='')
            # iteration
            branch1 = self.build(cur_depth + 1, X1, y1)
            print("| " * (cur_depth + 1), end='')
            print(new_feature, '=', value_range[1], ': ', end='')
            # iteration
            branch2 = self.build(cur_depth + 1, X2, y2)
            return Node(num, branch1, value_range[0], branch2, value_range[1], None)

    def _root(self):
        return self.root

    def choose_feature(self, X, y):
        gini_index = []
        for feature in X.T:
            gini_index.append(self.calc_gini_index(feature, y))
        return np.argmin(gini_index)

    def calc_gini_index(self, feature, label):
        values = np.unique(feature)
        gini_index = 0.0
        for value in values:
            value_indices = np.where(feature == value)
            value_labels = label[value_indices]
            # gini(value) = 1 - (p)^2
            gini_for_value = 1.0
            unique_labels = np.unique(value_labels)
            for i in unique_labels:
                prob = np.count_nonzero(value_labels == i) / len(value_labels)
                gini_for_value -= prob ** 2
            # gini(y|feature) += p_value * gini(value)
            gini_index += len(value_labels) / len(feature) * gini_for_value
        return gini_index


# input: a learned decision tree, data, and output file name
# function: output the predicted labels into output file
def test(tree, test_data, file_name):
    err_num = 0
    test_data = test_data.iloc[:, :].values
    sum_num = len(test_data)
    fp = open(file_name, "w")
    # find predicted label from root node
    for example in test_data:
        node = tree._root()
        while node._label() is None:
            if example[node._feature()] == node._value1():
                node = node._branch1()
            else:
                node = node._branch2()
        print(node._label(), file=fp)
        if node._label() != example[-1]:
            err_num = err_num + 1
    fp.close()
    return err_num / sum_num


def main():
    # read
    train_data = pd.read_csv(str(sys.argv[1]), sep='\t', header=0)
    test_data = pd.read_csv(str(sys.argv[2]), sep='\t', header=0)
    max_depth = int(sys.argv[3])

    # get the output filename
    train_out = str(sys.argv[4])
    test_out = str(sys.argv[5])
    matrix_out = str(sys.argv[6])

    # train
    tree = decisionTree(train_data, max_depth)

    # test
    error_train = test(tree, train_data, train_out)
    error_test = test(tree, test_data, test_out)

    # output
    fp = open(matrix_out, "w")
    print("error(train): ", error_train, file=fp)
    print("error(test): ", error_test, file=fp)
    fp.close()


if __name__ == "__main__":
    main()
