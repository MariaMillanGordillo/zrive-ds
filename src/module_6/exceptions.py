class UserNotFoundException(Exception):
    """Exception thrown when a user is not found in the database."""

    pass


class PredictionException(Exception):
    """Excepción thrown when there is an error during the prediction process."""

    pass
