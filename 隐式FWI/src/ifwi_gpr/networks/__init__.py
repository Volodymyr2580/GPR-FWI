"""Neural parameterization modules for IFWI."""

from .fr_inr import (
    FRINR,
    FourierReparamLinear,
    IFWIFrInrNetwork,
    SineFourierReparamLayer,
)
from .siren import IFWINetwork, SineLayer, Siren

__all__ = [
    "FRINR",
    "FourierReparamLinear",
    "IFWIFrInrNetwork",
    "IFWINetwork",
    "SineFourierReparamLayer",
    "SineLayer",
    "Siren",
]
