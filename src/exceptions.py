class SaccoError(Exception):
    """Base class for every error raised by the SACCO collection system."""
 
 
class ImportFileError(SaccoError):
    """An import file is missing, unreadable, empty or badly formed."""

class DisplayError(SaccoError):
    """A list or grid can't be built, e.g. a filter refers to a column that doesn't exist."""
 