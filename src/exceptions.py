class SaccoError(Exception):
    """Base class for every error raised by the SACCO collection system."""
 
 
class ImportFileError(SaccoError):
    """An import file is missing, unreadable, empty or badly formed."""
 