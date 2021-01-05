"""This module implements methods for defining functions to scalarize vector valued functions.
These are knows as `Scalarizer`s.
It also provides achievement
scalarizing functions to be used with the scalarizers.

"""

__all__ = [
    "AugmentedGuessASF",
    "MaxOfTwoASF",
    "PointMethodASF",
    "ReferencePointASF",
    "SimpleASF",
    "StomASF",
    "DiscreteScalarizer",
    "Scalarizer",
]

from desdeo_tools.scalarization.ASF import (
    AugmentedGuessASF,
    MaxOfTwoASF,
    PointMethodASF,
    ReferencePointASF,
    SimpleASF,
    StomASF,
)
from desdeo_tools.scalarization.Scalarizer import DiscreteScalarizer, Scalarizer
