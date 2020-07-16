class NoHistory(Exception):
    """
    Raised when a ModelParam instance's history attribute is none.
    """
    pass


class MissingArguments(Exception):
    """
    Raised when trying to make predictions/evaluations without passing in either features or labels
    """
    pass
