class NoHistory(Exception):
    """
    Raised when a ModelParam instance's history attribute is none.
    """
    pass


class InvalidFormat(Exception):
    """
    Raised when an invalid/unsupported file format is passed into the make_and_store_predictions function
    """
    pass
