class Query:
    """
    This class helps in identifying how to split the datasets 
    for further reduction in impurity
    """

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def is_numeric(self, value):
        return isinstance(value, int) or isinstance(value, float)

    def quantify(self, example):
        val = example[self.column]
        if self.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if self.is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))