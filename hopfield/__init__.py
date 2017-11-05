"""
Train and use Hopfield networks.

See https://en.wikipedia.org/wiki/Hopfield_network.
"""

from .update import hebbian_update, covariance_update
from .network import Network
