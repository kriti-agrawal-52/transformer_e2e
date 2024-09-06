class ValidationLossComputationError(Exception):
    """
    Raised when not enough validation batches succeed to compute a reliable validation loss.
    """

    pass
