class SaccoError(Exception):
    """Base class for every error raised by the SACCO collection system."""
 
 
class ImportFileError(SaccoError):
    """An import file is missing, unreadable, empty or badly formed."""

class DisplayError(SaccoError):
    """A list or grid can't be built, e.g. a filter refers to a column that doesn't exist."""
 

class PolicyError(SaccoError):
    """configs/policy.json is missing, or one of its rules is incomplete or contradicts another."""


class ModelError(SaccoError):
    """A risk model can't be trained, used or loaded, e.g. it hasn't been trained yet."""
