class Leaf_Node:
    """
    A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
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
        
        self.predictions = class_counts(rows)



