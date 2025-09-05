import os

import pydantic
import omegaconf
import numpy as np

def dump_cfg(path: str, obj: omegaconf.DictConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(obj))

def dump_pydantic(path: str, obj: pydantic.BaseModel):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(obj.model_dump_json(indent=2))

def str2float(s):
    """Cast string to float if it is not None. Otherwise return None.

    Args:
        s (str): String to convert or None.

    Returns:
        str or NoneType: The converted string or None.
    """
    return float(s) if s is not None else None

def array_eq(arr1, arr2, tolerance=1e-8):
    if arr1 is None and arr2 is None:
        return True
    return (
        isinstance(arr1, np.ndarray)
        and isinstance(arr2, np.ndarray)
        and arr1.shape == arr2.shape
        and np.allclose(arr1, arr2, atol=tolerance)
    )

def scalar_eq(scalar1, scalar2, tolerance=1e-8):
    return np.isclose(scalar1, scalar2, atol=tolerance)