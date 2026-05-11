from importlib.metadata import version as _distribution_version

from .kuwahara import kuwahara

__version__ = _distribution_version("pykuwahara")

__all__ = ["kuwahara", "__version__"]
