class UserNotFoundException(Exception):
    """Excepción lanzada cuando el usuario no se encuentra en el feature store."""
    pass


class PredictionException(Exception):
    """Excepción lanzada cuando ocurre un error durante la predicción del modelo."""
    pass
