import os
from functools import partial

import pydantic
import omegaconf
import numpy as np

#################################
#     utils for exportation     #
#################################


def dump_cfg(path: str, obj: omegaconf.DictConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(obj))

def dump_pydantic(path: str, obj: pydantic.BaseModel):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(obj.model_dump_json(indent=2))

#####################################
#     utils for basic operation     #
#####################################

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

####################################
#    utils for filename handling   #
####################################

def filename_handler_ignore_directive(fname):
    """A filename handler that removes anything before (and including) '://'.

    Args:
        fname (str): A file name.

    Returns:
        str: The file name without the prefix.
    """
    if "://" in fname or ":\\\\" in fname:
        return ":".join(fname.split(":")[1:])[2:]
    return fname

def filename_handler_relative(fname, dir):
    """A filename handler that joins a file name with a directory.

    Args:
        fname (str): A file name.
        dir (str): A directory.

    Returns:
        str: The directory joined with the file name.
    """
    return os.path.join(dir, filename_handler_ignore_directive(fname))

def filename_handler_relative_to_urdf_file(fname, urdf_dir, level=0):
    """A filename handler that makes a file name relative to the URDF file.

    Args:
        fname (str): A file name.
        urdf_dir (str): The directory where URDF file is located.
        level (int, optional): The number of levels to go up from the URDF file directory. Defaults to 0.

    Returns:
        str: The file name.
    """

    dir = urdf_dir
    for i in range(level):
        dir = os.path.dirname(dir)

    return filename_handler_relative(fname, dir)

def create_filename_handlers_relative_to_urdf_file(urdf_dir):
    return [
        partial(
            filename_handler_relative_to_urdf_file,
            urdf_dir=urdf_dir,
            level=i,
        )
        for i in range(len(os.path.normpath(urdf_dir).split(os.path.sep)))
    ]

def filename_handler_meta(fname, filename_handlers):
    """A filename handler that calls other filename handlers until the resulting file name points to an existing file.

    Args:
        fname (str): A file name.
        filename_handlers (list(fn)): A list of function pointers to filename handlers.

    Returns:
        str: The resolved file name that points to an existing file or the input if none of the files exists.
    """
    for fn in filename_handlers:
        candidate_fname = fn(fname=fname)
        if os.path.isfile(candidate_fname):
            return candidate_fname
    return fname

def filename_handler_magic(fname, dir):
    """A magic filename handler.

    Args:
        fname (str): A file name.
        dir (str): A directory.

    Returns:
        str: The file name that exists or the input if nothing is found.
    """
    return filename_handler_meta(
        fname=fname,
        filename_handlers=[
            partial(filename_handler_relative, dir=dir),
            filename_handler_ignore_directive,
        ]
        + create_filename_handlers_relative_to_urdf_file(urdf_dir=dir),
    )