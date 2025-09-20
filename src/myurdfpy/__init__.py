from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .urdf import (
    Actuator,
    Box,
    Calibration,
    Collision,
    Color,
    Cylinder,
    Dynamics,
    Geometry,
    Inertial,
    Joint,
    Link,
    Limit,
    Material,
    Mesh,
    Mimic,
    Robot,
    SafetyController,
    Sphere,
    Texture,
    Transmission,
    TransmissionJoint,
    URDF,
    Visual,
    URDFError,
    URDFIncompleteError,
    URDFBrokenRefError,
    URDFSaveValidationError,
    URDFMalformedError,
    URDFUnsupportedError,
)

# from .urdf_rtb import URDF as URDF_RTB

__all__ = [Actuator, Box, Calibration, Collision, Color, Cylinder, Dynamics, Geometry, Inertial, Joint, Link, Limit, 
           Material, Mesh, Mimic, Robot, SafetyController, Sphere, Texture, Transmission, TransmissionJoint, URDF, 
           Visual, URDFError, URDFIncompleteError, URDFBrokenRefError, URDFSaveValidationError, URDFMalformedError, 
           URDFUnsupportedError]
