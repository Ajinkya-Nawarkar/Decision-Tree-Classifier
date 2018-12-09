from __future__ import print_function
import argparse
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


def get_best_split(rows, header):
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


def build_tree(rows, header):
    """
    This function helps builds the decision tree using recursion 
    At each step the best decision metric is determined using information gain
    and split.     
    """

    # Return the best split with given features so far
    gain, query = get_best_split(rows, header)

    if gain == 0:
        # Address this node as terminal node (leaf) or the prediction metric
        return Leaf_Node(rows)

    true_rows, false_rows = partition(rows, query)

    true_branch = build_tree(true_rows, header)
    false_branch = build_tree(false_rows, header)

    # This records the best feature / value to ask at this point
    return Decision_Node(query, true_branch, false_branch)


def predict(row, node):
    """
    This function helps classify a given row based on where and which leaf our decision tree points to.
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


def interpret_predictions(prediction_list):
    """
    This function helps to interpret the list of dictionaries
    where each dictionary contains predictions for an example  
    
    Return a single prediction from each dictionary as a list 
    for the final accuracy calculation. 

    For example: if     [{"prediction_1": 2, "prediction_2": 1}, {"prediction_3": 2}]
    Interpreted as:     [{"prediction_1": 66%, "prediction_2": 33%}, {"prediction_3": 100%}]
    predicted as:       ['prediction_1', 'prediction3']
    """
    predictions = []
    
    for each_dict in prediction_list:
        def interpret_dict(prediction_dict):
            """
            This function interprets each dictionary
            returns a single prediction
            """
            if (len(prediction_dict) == 1):
                key, value = prediction_dict.items()[0]
                return key
            else:
                values = list(prediction_dict.values())
                keys = list(prediction_dict.keys())
                return keys[values.index(max(values))]

        predictions.append(interpret_dict(each_dict))

    return predictions


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


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def decision_tree(training_data, testing_data, results, header):
    """
    This functions runs trained decision tree on test set
    and returns indirect predictions
    """
    decision_tree = build_tree(training_data, header)
    
    if (results.print_DT):
        print_tree(decision_tree)
    
    predictions = list()
    for row in testing_data:
        prediction = predict(row, decision_tree)
        # print (print_leaf(prediction))
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


def evaluate_algorithm(dataset, algorithm, results, *args):
    """
    This function helps to evaluate an algorithm using a cross validation split
    Standard function cited from www.machinelearningmastery.com
    """
    folds = cross_validation_split(dataset, results.n_folds)
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
        predicted = algorithm(train_set, test_set, results, *args)
        actual = [row[-1] for row in fold]

        # print ("Actual: %s. Predicted: %s" %
        #        (actual, [print_leaf(each) for each in predicted] ))

        print ("\ntrain_set size: %d | test_set size: %d\n" %(len(train_set), len(test_set)))

        accuracy = accuracy_metric(actual, interpret_predictions(predicted))
        scores.append(accuracy)
    return scores


def get_dataset(filename):
    """
    This function helps retrive different datasets and sanitzes them
    Standard functions cited from www.machinelearningmastery.com
    """

    def load_csv(filename):
        file = open(filename, "rb")
        lines = reader(file)
        dataset = list(lines)
        return dataset

    def sanitize(dataset):
        """ Converts str numeral into floats """
        for col in range(len(dataset[0])):
            for row in dataset:
                if (row[col].lstrip('-').replace('.','',1).isdigit()):
                    row[col] = float(row[col])
        return dataset


    dataset = load_csv(filename)
    dataset = sanitize(dataset)
    return dataset


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', action='store', dest='set_type',
                    help='Type \'seismic\' for seismic-bumps dataset or\
                    \'banknote\' for banknote-authentication dataset or\
                    \'bankruptsy\' for bankrupty qualitative parameters or\
                    \'balance\' for balance-scale dataset')

    parser.add_argument('--dt', action='store_true', default=False,
                        dest='print_DT',
                        help='Print the decision tree')

    parser.add_argument('--kf', action='store', dest='n_folds',
                        help='No. of K folds for cross validation', type=int, default=3)

    results = parser.parse_args()

    filename1 = 'data/seismic-bumps.csv'
    filename2 = 'data/banknote-authentication.csv'
    filename3 = 'data/bankrupty-parameters.csv'
    filename4 = 'data/balance-scale.csv'
    header = ['0']*20

    if (results.set_type == 'seismic'):
        filename = filename1
        header = ['seismic {a,b,c,d}',
                    'seismoacoustic {a,b,c,d}',
                    'shift {W, N}',
                    'genergy',
                    'gpuls',
                    'gdenergy',
                    'gdpuls',
                    'ghazard {a,b,c,d}',
                    'nbumps',
                    'nbumps2',
                    'nbumps3',
                    'nbumps4',
                    'nbumps5',
                    'nbumps6',
                    'nbumps7',
                    'nbumps89',
                    'energy',
                    'maxenergy',
                    'label']

    elif (results.set_type == 'banknote'):
        filename = filename2
        header = ['variance',
                    'skewness',
                    'curtosis',
                    'entropy',
                    'label']

    elif (results.set_type == 'bankruptcy'):
        filename = filename3
        header = ['Industrial Risk: {P,A,N}',
                    'Management Risk: {P,A,N}', 
                    'Financial Flexibility: {P,A,N}',
                    'Credibility: {P,A,N}',
                    'Competitiveness: {P,A,N}',
                    'Operating Risk',
                    'label']

    elif (results.set_type == 'balance'):
        filename = filename4
        header = ['Left-Weight: 0-5',
                    'Left-Distance: 0-5', 
                    'Right-Weight: 0-5', 
                    'Right-Distance: 0-5']

    def run_classfier(filename, results):
        dataset = get_dataset(filename)
        scores = evaluate_algorithm(dataset, decision_tree, results, header)
        print('\nScores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    run_classfier(filename, results)

