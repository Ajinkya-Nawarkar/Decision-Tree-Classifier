from __future__ import print_function
from random import seed
from random import randrange
from csv import reader
from query import Query 
from leaf import Leaf_Node
from decision import Decision_Node


def unique_vals(rows, col):
    """
    This function helps find the unique values for a column in a dataset.
    """
    row_list= [row[col] for row in rows]
    return set(row_list)


def class_counts(rows):
    """
    This function helps count the number of each type of example in a dataset.
    """
    counts = {} 
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def partition(rows, query):
    """
    This function helps to partition a dataset.
    Split a dataset based on whether or not value of 
    the row matches the query
    """
    left, right = [], []
    for row in rows:
        if query.quantify(row):
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(rows):
    """
    This function helps calculate the Gini Impurity Index for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    gini = 1
    for lbl in counts:
        p = counts[lbl] / float(len(rows))
        gini -= p**2
    return gini


def info_gain(left, right, current_uncertainty):
    """
    This function helps calculate the Information Gain.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini_index(left) - (1 - p) * gini_index(right)


def get_best_split(rows):
    """
    This function helps for calculating the information gain 
    for every feature / value pair and determining which 
    feature provides best split
    """
    
    # best information gain
    b_gain = 0  
    # best feature / value that produced it
    b_query = None  

    current_uncertainty = gini_index(rows)
    n_features = len(rows[0]) - 1  

    for col in range(n_features):  
        values = unique_vals(rows, col)  

        for val in values: 
            # generate a query 
            query = Query(col, val, header)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, query)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # dataset.
            if gain >= b_gain:
                b_gain, b_query = gain, query

    return b_gain, b_query


def build_tree(rows):
    """
    This function helps builds the decision tree using recursion 
    At each step the best decision metric is determined using information gain
    and split.     
    """

    # Return the best split with given features so far
    gain, query = get_best_split(rows)

    if gain == 0:
        # Address this node as terminal node (leaf) or the prediction metric
        return Leaf_Node(rows)

    true_rows, false_rows = partition(rows, query)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    # This records the best feature / value to ask at this point
    return Decision_Node(query, true_branch, false_branch)


def print_tree(node, spacing=""):
    """
    Print the tree
    """

    # Base case: we've reached a leaf
    if isinstance(node, Leaf_Node):
        print (spacing + "Predict", node.predictions)
        return

    # Print the query at this node
    print (spacing + str(node.query))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def predict(row, node):
    """
    Here we classify a given row based on where and which leaf our decision tree points to.
    """

    # Base case: we've reached a leaf node, return the predictions
    if isinstance(node, Leaf_Node):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch
    # based on nthe given query at that node
    if node.query.quantify(row):
        return predict(row, node.true_branch)
    else:
        return predict(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def decision_tree(training_data, testing_data):
    decision_tree = build_tree(training_data)
    predictions = list()
    for row in testing_data:
        prediction = print_leaf(predict(row, decision_tree))
        predictions.append(prediction)
    return(predictions)

def cross_validation_split(dataset, n_folds):
    """
    This function helps split a dataset into k folds
    Standard function cited from www.machinelearningmastery.com
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    """
    This function helps calculate the accuracy percentage
    Standard function cited from www.machinelearningmastery.com
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """
    This function helps to evaluate an algorithm using a cross validation split
    Standard function cited from www.machinelearningmastery.com
    """
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


if __name__ == '__main__':

    # Dataset
    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    header = ["color", "diameter", "label"]

    # evaluate algorithm
    n_folds = 5
    scores = evaluate_algorithm(dataset, decision_tree, n_folds)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    # Evaluate
    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(predict(row, decision_tree))))

