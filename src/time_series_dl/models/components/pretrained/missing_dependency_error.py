class MissingDependencyError(ImportError):
    """
    Raised when published weights are requested but the optional ``pretrained`` dependencies are not installed.
    """
