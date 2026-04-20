"""Multimodal Anti Money Laundering.

Multimodal AML detection using GNN, DistilBERT, and Bi-LSTM
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multimodal_anti_money_laundering")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "Anusooya Thimmarayi Neha, Jaya Prakash Yadav Gorla, Preshita Soni, Rajani Meka"
__email__ = "nanusooy@depaul.edu, jgorla@depaul.edu, psoni7@depaul.edu, rmeka1@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]
