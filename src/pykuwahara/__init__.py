from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .kuwahara import kuwahara

try:
    __version__ = _distribution_version("pykuwahara")
except PackageNotFoundError:  # imported from sources without installation
    __version__ = "0+unknown"

__all__ = ["kuwahara", "__version__"]
