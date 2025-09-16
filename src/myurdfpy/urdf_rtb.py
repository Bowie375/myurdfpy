import os
import logging
from functools import partial
from dataclasses import is_dataclass
from typing import Union, Literal, Tuple

import six
import numpy as np
import trimesh
import trimesh.transformations as tra
from lxml import etree
import spatialgeometry as sg
import roboticstoolbox as rtb
import roboticstoolbox.tools.urdf as rtb_urdf
import roboticstoolbox.tools.xacro as rtb_xacro

from myurdfpy.utils import filename_handler_magic, timeit_decorator

_logger = logging.getLogger(__name__)

# threshold for comparison
EQUALITY_TOLERANCE = 1e-8

class URDFError(Exception):
    """General URDF exception."""

    def __init__(self, msg):
        super(URDFError, self).__init__()
        self.msg = msg

    def __str__(self):
        return type(self).__name__ + ": " + self.msg

    def __repr__(self):
        return type(self).__name__ + '("' + self.msg + '")'


class URDFIncompleteError(URDFError):
    """Raised when needed data for an object that isn't there."""

    pass


class URDFAttributeValueError(URDFError):
    """Raised when attribute value is not contained in the set of allowed values."""

    pass


class URDFBrokenRefError(URDFError):
    """Raised when a referenced object is not found in the scope."""

    pass


class URDFMalformedError(URDFError):
    """Raised when data is found to be corrupted in some way."""

    pass


class URDFUnsupportedError(URDFError):
    """Raised when some unexpectedly unsupported feature is found."""

    pass


class URDFSaveValidationError(URDFError):
    """Raised when XML validation fails when saving."""

    pass


class URDF:
    def __init__(
        self,
        urdf: rtb_urdf.URDF = None,
        robot: rtb.ERobot = None,
        build_scene_graph: bool = True,
        build_collision_scene_graph: bool = True,
        mesh_dir: str = "",
        filename_handler: callable = None,
        load_meshes: bool = True,
        load_collision_meshes: bool = True,
        force_mesh: bool = False,
        force_collision_mesh: bool = True,
        skip_materials: bool = False,
    ):
        """A URDF model.

        Args:
            urdf (rtb.tools.URDF): the URDF object after parsing the URDF file.
            robot (rtb.ERobot): The robot model. Defaults to None.
            build_scene_graph (bool, optional): Whether to build a scene graph to enable transformation queries and forward kinematics. Defaults to True.
            build_collision_scene_graph (bool, optional): Whether to build a scene graph for <collision> elements. Defaults to True.
            mesh_dir (str, optional): Directory where to look for meshes. Defaults to None.
            filename_handler (callable, optional): A function that takes a file name and returns a valid file path. This can be used to modify the file names of <mesh> elements. If None, a default handler that makes the file names relative to mesh_dir is used. Defaults to None.
            load_meshes (bool, optional): Whether to load the meshes referenced in the <mesh> elements. Defaults to True.
            load_collision_meshes (bool, optional): Whether to load the collision meshes referenced in the <mesh> elements. Defaults to False.
            force_mesh (bool, optional): Each loaded geometry will be concatenated into a single one (instead of being turned into a graph; in case the underlying file contains multiple geometries). This might loose texture information but the resulting scene graph will be smaller. Defaults to True.
            force_collision_mesh (bool, optional): Same as force_mesh, but for collision scene. Defaults to True.
            skip_materials (bool, optional): Materials will not be loaded. Defaults to False.
        """
        if filename_handler is None:
            self._filename_handler = partial(filename_handler_magic, dir=mesh_dir)
        else:
            self._filename_handler = filename_handler

        self.urdf: rtb_urdf.URDF = urdf
        self.robot: rtb.ERobot = robot
        self._create_maps()
        self._update_actuated_joints()

        if build_scene_graph or build_collision_scene_graph:
            self._base_link = self.robot.base_link
        else:
            self._base_link = None

        self._errors = []

        if build_scene_graph:
            self._scene = self._create_scene(
                use_collision_geometry=False,
                load_geometry=load_meshes,
                force_mesh=force_mesh,
                force_single_geometry_per_link=skip_materials or force_mesh,
                skip_materials=skip_materials,
            )
        else:
            self._scene = None

        if build_collision_scene_graph:
            self._scene_collision = self._create_scene(
                use_collision_geometry=True,
                load_geometry=load_collision_meshes,
                force_mesh=force_collision_mesh,
                force_single_geometry_per_link=force_collision_mesh,
                skip_materials=True,
            )
        else:
            self._scene_collision = None

    def __eq__(self, other):
        if not isinstance(other, URDF):
            raise NotImplementedError
        return self.robot == other.robot

    def clear_errors(self):
        """Clear the validation error log."""
        self._errors = []

    def _create_maps(self):
        self._joint_map: dict[str, rtb.Link] = {}
        self._link_map: dict[str, rtb.Link] = {}
        for link in self.robot.links:
            self._link_map[link.name] = link
            if link.isjoint:
                self._joint_map[link.name] = link

    def _update_actuated_joints(self):
        """
        Actuated joints will be sorted according to their jindex, so when
        updating configuration using an np.ndarray, the order of values in 
        the array will match the order of joint in list self.actuated_joints.
        """

        self._actuated_joints: list[rtb.Link] = []
        self._actuated_dof_indices: list[list[int]] = []

        for j in [link for link in self.robot.links if link.isjoint]:
            self._actuated_joints.append(j)
        self._actuated_joints.sort(key = lambda j:j.jindex)
        for j in self._actuated_joints:
            self._actuated_dof_indices.append([j.jindex])


    ###############################################################
    # The following method is the main entry point for this class #
    ###############################################################

    @staticmethod
    def load(file_path, **kwargs):
        """Load URDF file and Construct a Robot.

        Args:
            **file_path (str): A valid path to the URDF file.
            **gripper (str | int, optional): index or name of the gripper link(s). If ``gripper`` is specified, links from that link outward are removed from the rigid-body tree and folded into a ``Gripper`` object.
            **build_scene_graph (bool, optional): Whether to build a scene graph to enable transformation queries and forward kinematics. Defaults to True.
            **build_collision_scene_graph (bool, optional): Whether to build a scene graph for <collision> elements. Defaults to False.
            **mesh_dir (str, optional): Directory where to look for meshes. Defaults to "".
            **load_meshes (bool, optional): Whether to load the meshes referenced in the <mesh> elements. Defaults to True.
            **filename_handler (callable, optional): A function that takes a file name and returns a valid file path. This can be used to modify the file names of <mesh> elements. If None, a default handler that makes the file names relative to mesh_dir is used. Defaults to None.
            **load_collision_meshes (bool, optional): Whether to load the collision meshes referenced in the <mesh> elements. Defaults to False.
            **force_mesh (bool, optional): Each loaded geometry will be concatenated into a single one (instead of being turned into a graph; in case the underlying file contains multiple geometries). This might loose texture information but the resulting scene graph will be smaller. Defaults to False.
            **force_collision_mesh (bool, optional): Same as force_mesh, but for collision scene. Defaults to True.
            **skip_materials (bool, optional): Materials will not be loaded. Defaults to False.
        Raises:
            ValueError: If filename does not exist or gripper link specified with invalid index or name.

        Returns:
            myurdfpy.URDF: URDF model.
        """

        if not os.path.isfile(file_path):
            raise ValueError("{} is not a file".format(file_path))
        file_path = os.path.abspath(file_path)

        if kwargs.get("mesh_dir", None) is None:
            kwargs["mesh_dir"] = os.path.dirname(file_path)

        try:
            ### First: Parse URDF File
            _, ext = os.path.splitext(file_path)

            if ext == ".xacro":
                # it's a xacro file, preprocess it
                urdf_string = rtb_xacro.main(file_path)
            else:  # pragma nocover
                urdf_string = open(file_path).read()
            if not isinstance(urdf_string, str):  # pragma nocover
                raise ValueError("urdf_rtb.py >> load(): Parsing failed, did not get valid URDF string back")
            urdf: rtb_urdf.URDF = rtb_urdf.URDF.loadstr(urdf_string, file_path, base_path="")


            ### Second: Create Robot Model
            links, name, urdf_string, urdf_filepath = urdf.elinks, urdf.name, urdf_string, file_path

            gripper = kwargs.get("gripper", None)
            gripperLink: Union[rtb.Link, None] = None
            if gripper is not None:
                if isinstance(gripper, int):
                    if gripper > len(links):
                        raise ValueError(f"urdf_rtb.py >> load(): gripper index {gripper} out of range")
                    gripperLink = links[gripper]
                elif isinstance(gripper, str):
                    for link in links:
                        if link.name == gripper:
                            gripperLink = link
                            break
                    else:  # pragma nocover
                        raise ValueError(f"urdf_rtb.py >> load(): no link named {gripper}")
                else:  # pragma nocover
                    raise TypeError("urdf_rtb.py >> load(): bad argument passed as gripper")

            robot = rtb.ERobot(
                links, 
                name=name, 
                gripper_links=gripperLink, 
                urdf_string=urdf_string, 
                urdf_filepath=urdf_filepath
            )

        except Exception as e:
            _logger.error(e)
            raise URDFMalformedError("urdf_rtb.py >> load(): Failed to parse XML file")

        return URDF(urdf=urdf, robot=robot, **kwargs)


    ###############################################
    # The following parts are APIs for properties #
    ###############################################

    @property
    def scene(self) -> trimesh.Scene:
        """A scene object representing the URDF model.

        Returns:
            trimesh.Scene: A trimesh scene object.
        """
        return self._scene

    @property
    def collision_scene(self) -> trimesh.Scene:
        """A scene object representing the <collision> elements of the URDF model.

        Returns:
            trimesh.Scene: A trimesh scene object.
        """
        return self._scene_collision

    @property
    def link_map(self) -> dict[str, rtb.Link]:
        """A dictionary mapping link names to link objects.

        Returns:
            dict: Mapping from link name (str) to Link.
        """
        return self._link_map

    @property
    def joint_map(self) -> dict[str, rtb.Link]:
        """A dictionary mapping joint names to joint objects.

        Returns:
            dict: Mapping from joint name (str) to Joint.
        """
        return self._joint_map

    @property
    def joint_names(self):
        """List of joint names.

        Returns:
            list[str]: List of joint names of the URDF model.
        """
        return [j.name for j in self.robot.links if j.isjoint]

    @property
    def actuated_joints(self):
        """List of actuated joints. This excludes mimic and fixed joints.

        Returns:
            list[Joint]: List of actuated joints of the URDF model.
        """
        return self._actuated_joints

    @property
    def actuated_dof_indices(self):
        """List of DOF indices per actuated joint. Can be used to reference configuration.

        Returns:
            list[list[int]]: List of DOF indices per actuated joint.
        """
        return self._actuated_dof_indices

    #@property
    #def actuated_joint_indices(self):
    #    """List of indices of all joints that are actuated, i.e., not of type mimic or fixed.

    #    Returns:
    #        list[int]: List of indices of actuated joints.
    #    """
    #    return self._actuated_joint_indices

    @property
    def actuated_joint_names(self):
        """List of names of actuated joints. This excludes mimic and fixed joints.

        Returns:
            list[str]: List of names of actuated joints of the URDF model.
        """
        return [j.name for j in self._actuated_joints]

    @property
    def num_actuated_joints(self):
        """Number of actuated joints.

        Returns:
            int: Number of actuated joints.
        """
        return len(self._actuated_joints)

    @property
    def num_dofs(self):
        """Number of degrees of freedom of actuated joints. Depending on the type of the joint, the number of DOFs might vary.

        Returns:
            int: Degrees of freedom.
        """

        return len(self._actuated_joints)

    @property
    def zero_cfg(self):
        """Return the zero configuration.

        Returns:
            np.ndarray: The zero configuration.
        """
        return np.zeros(self.num_dofs)

    @property
    def center_cfg(self):
        """Return center configuration of URDF model by using the average of each joint's limits if present, otherwise zero.

        Returns:
            (n), float: Default configuration of URDF model.
        """
        config = []
        config_names = []
        for j in self._actuated_joints:
            if j.qlim is not None:
                cfg = [j.qlim[0] + 0.5 * (j.qlim[1] - j.qlim[0])]
            else:
                cfg = [0.0]

            config.append(cfg)
            config_names.append(j.name)

        if len(config) == 0:
            return np.array([], dtype=np.float64)
        return np.concatenate(config)

    @property
    def cfg(self):
        """Current configuration.

        Returns:
            np.ndarray: Current configuration of URDF model.
        """
        return self.robot.q

    @cfg.setter
    def cfg(self, value):
        """Set the current configuration.

        Args:
            value (np.ndarray): The new configuration.
        """
        self.robot.q = value    

    @property
    def base_link(self):
        """Name of URDF base/root link.

        Returns:
            str: Name of base link of URDF model.
        """
        return self._base_link

    @property
    def errors(self) -> list:
        """A list with validation errors.

        Returns:
            list: A list of validation errors.
        """
        return self._errors


    ###########################################################
    # The following methods are used to inspect into the URDF #
    ###########################################################

    def contains(self, key, value, element=None) -> bool:
        """Checks recursively whether the URDF tree contains the provided key-value pair.

        Args:
            key (str): A key.
            value (str): A value.
            element (etree.Element, optional): The XML element from which to start the recursive search. None means URDF root. Defaults to None.

        Returns:
            bool: Whether the key-value pair was found.
        """
        if element is None:
            element = self.robot

        result = False
        for fld in element.__dataclass_fields__:
            field_value = getattr(element, fld)
            if is_dataclass(field_value):
                result = result or self.contains(
                    key=key, value=value, element=field_value
                )
            elif (
                isinstance(field_value, list)
                and len(field_value) > 0
                and is_dataclass(field_value[0])
            ):
                for field_value_element in field_value:
                    result = result or self.contains(
                        key=key, value=value, element=field_value_element
                    )
            else:
                if key == fld and value == field_value:
                    result = True
        return result

    def get_transform(self, frame_to, frame_from=None, collision_geometry=False):
        """Get the transform from one frame to another.

        Args:
            frame_to (str): Node name.
            frame_from (str, optional): Node name. If None it will be set to self.base_frame. Defaults to None.
            collision_geometry (bool, optional): Whether to use the collision geometry scene graph (instead of the visual geometry). Defaults to False.

        Raises:
            ValueError: Raised if scene graph wasn't constructed during intialization.

        Returns:
            (4, 4) float: Homogeneous transformation matrix
        """
        if collision_geometry:
            if self._scene_collision is None:
                raise ValueError(
                    "No collision scene available. Use build_collision_scene_graph=True during loading."
                )
            else:
                return self._scene_collision.graph.get(
                    frame_to=frame_to, frame_from=frame_from
                )[0]
        else:
            if self._scene is None:
                raise ValueError(
                    "No scene available. Use build_scene_graph=True during loading."
                )
            else:
                return self._scene.graph.get(frame_to=frame_to, frame_from=frame_from)[
                    0
                ]

    ##########################################################
    # The following methods are used to create trimesh scene #
    ##########################################################

    def _geometry2trimeshscene(self, geometry, load_file: bool, force_mesh: bool):
        """Import a single geometry object into a trimesh scene.

        Args:
            geometry (sg.Geometry): A geometry object.
            load_file (bool): Whether to load the geometry file.
            force_mesh (bool): Whether to force the geometry to be a single mesh.

        Returns:
            trimesh.Scene: A trimesh scene object.
        """

        new_s = None
        if isinstance(geometry, sg.Cuboid):
            new_s = trimesh.primitives.Box(extents=geometry.scale).scene()
        elif isinstance(geometry, sg.Sphere):
            new_s = trimesh.primitives.Sphere(radius=geometry.radius).scene()
        elif isinstance(geometry, sg.Cylinder):
            new_s = trimesh.primitives.Cylinder(
                radius=geometry.radius, height=geometry.length
            ).scene()
        elif isinstance(geometry, sg.Mesh) and load_file:
            new_filename = self._filename_handler(geometry.filename)

            if os.path.isfile(new_filename):
                _logger.debug(f"Loading {geometry.filename} as {new_filename}")

                if force_mesh:
                    new_g = trimesh.load(new_filename, force="mesh")

                    # add original filename
                    if "file_path" not in new_g.metadata:
                        new_g.metadata["file_path"] = os.path.abspath(new_filename)
                        new_g.metadata["file_name"] = os.path.basename(new_filename)

                    new_s = trimesh.Scene()
                    new_s.add_geometry(new_g)
                else:
                    new_s = trimesh.load(new_filename, force="scene")

                    if "file_path" in new_s.metadata:
                        for i, (_, geom) in enumerate(new_s.geometry.items()):
                            if "file_path" not in geom.metadata:
                                geom.metadata["file_path"] = new_s.metadata["file_path"]
                                geom.metadata["file_name"] = new_s.metadata["file_name"]
                                geom.metadata["file_element"] = i

                # scale mesh appropriately
                if geometry.scale is not None:
                    if isinstance(geometry.scale, float):
                        new_s = new_s.scaled(geometry.scale)
                    elif isinstance(geometry.scale, np.ndarray):
                        new_s = new_s.scaled(geometry.scale)
                    else:
                        _logger.warning(
                            f"Warning: Can't interpret scale '{geometry.scale}'"
                        )
            else:
                _logger.warning(f"Can't find {new_filename}")
        return new_s

    def _add_geometries_to_scene(
        self,
        s: trimesh.Scene,
        geometries: sg.SceneGroup,
        link_name: str,
        load_geometry: bool,
        force_mesh: bool,
        force_single_geometry: bool,
        skip_materials: bool,
    ):
        """
        Add all geometries on a Link to a trimesh scene.
        """

        if force_single_geometry:
            tmp_scene = trimesh.Scene(base_frame=link_name)

        def apply_visual_color(
            geom: trimesh.Trimesh,
            visual,
            material_map: dict,
        ) -> None:
            """Apply the color of the visual material to the mesh.

            Args:
                geom: Trimesh to color.
                visual: Visual description from XML.
                material_map: Dictionary mapping material names to their definitions.
            """
            if visual.material is None:
                return

            if visual.material.color is not None:
                color = visual.material.color
            elif visual.material.name is not None and visual.material.name in material_map:
                color = material_map[visual.material.name].color
            else:
                return

            if color is None:
                return
            if isinstance(geom.visual, trimesh.visual.ColorVisuals):
                geom.visual.face_colors[:] = [int(255 * channel) for channel in color.rgba]

        first_geom_name = None

        for g in geometries:
            new_s = self._geometry2trimeshscene(
                geometry=g,
                load_file=load_geometry,
                force_mesh=force_mesh,
            )
            if new_s is not None:
                origin = g._wT if g._wT is not None else np.eye(4)

                if force_single_geometry:
                    for name in new_s.graph.nodes_geometry:
                        T, geom_name = new_s.graph.get(name)
                        geom = new_s.geometry[geom_name]

                        # if isinstance(v, Visual):
                        #     if skip_materials:
                        #         geom.visual = trimesh.visual.ColorVisuals(geom) # remove color information
                        #     else:
                        #         apply_visual_color(geom, v, self._material_map)

                        tmp_scene.add_geometry(
                            geometry=geom,
                            geom_name=geom_name,
                            parent_node_name=link_name,
                            transform=origin @ T,
                        )
                else:
                    for name in new_s.graph.nodes_geometry:
                        T, geom_name = new_s.graph.get(name)
                        geom = new_s.geometry[geom_name]

                        # if isinstance(v, Visual):
                        #     if skip_materials:
                        #         geom.visual = trimesh.visual.ColorVisuals(geom) # remove color information
                        #     else:
                        #         apply_visual_color(geom, v, self._material_map)

                        s.add_geometry(
                            geometry=geom,
                            geom_name=geom_name,
                            parent_node_name=link_name,
                            transform=origin @ T,
                        )

        if force_single_geometry and len(tmp_scene.geometry) > 0:
            s.add_geometry(
                geometry=tmp_scene.dump(concatenate=True),
                geom_name=first_geom_name,
                parent_node_name=link_name,
                transform=np.eye(4),
            )

    def _create_scene(
        self,
        use_collision_geometry: bool = False,
        load_geometry: bool = True,
        force_mesh: bool = False,
        force_single_geometry_per_link: bool = False,
        skip_materials: bool = False,
    ):
        s = trimesh.Scene(base_frame=self._base_link)

        T = self.robot.fkine_all(self.robot.q)

        for link in self.robot.links:
            s.graph.update(frame_from=s.graph.base_frame, frame_to=link.name, matrix=T[link.number])

            meshes = link.collision if use_collision_geometry else link.geometry
            self._add_geometries_to_scene(
                s,
                geometries=meshes,
                link_name=link.name,
                load_geometry=load_geometry,
                force_mesh=force_mesh,
                force_single_geometry=force_single_geometry_per_link,
                skip_materials=skip_materials,
            )

        return s


    ###############################################################################
    # The following methods are used for computing forward and inverse kinematics #
    ###############################################################################

    def update_cfg(self, configuration) -> list[np.ndarray]:
        """Update joint configuration of URDF; does forward kinematics.

        Args:
            configuration (dict, list[float], tuple[float] or np.ndarray): A mapping from joints or joint names to configuration values, or a list containing a value for each actuated joint.

        Raises:
            ValueError: Raised if dimensionality of configuration does not match number of actuated joints of URDF model.
            TypeError: Raised if configuration is neither a dict, list, tuple or np.ndarray.

        Returns:
            list[np.ndarray]: A list of poses for base frame and each link in the URDF model.
            Total length of the list is (1 + n). The first entry in the list if base frame pose,
            the link.number entry in list is the pose for that link.
        """
        joint_cfg = self.robot.q.copy()

        if isinstance(configuration, dict):
            for joint in configuration:
                if isinstance(joint, six.string_types):
                    joint_cfg[self._joint_map[joint].jindex] = configuration[joint]
                else:
                    raise TypeError("Invalid type for joint name")
        elif isinstance(configuration, (list, tuple, np.ndarray)):
            if len(configuration) != self.num_actuated_joints:
                raise ValueError(
                    f"Dimensionality of configuration ({len(configuration)}) doesn't match number of actuated joints ({self.num_actuated_joints})."
                )
        else:
            raise TypeError("Invalid type for configuration")

        # append all mimic joints in the update
        T = self.robot.fkine_all(q=joint_cfg)

        # update internal configuration vector - only consider actuated joints
        self.cfg = joint_cfg

        if self._scene is not None:
            for name, link in self.link_map.items():
                self._scene.graph.update(name, matrix=T[link.number])

        if self._scene_collision is not None:
            for name, link in self.link_map.items():
                self._scene_collision.graph.update(name, matrix=T[link.number])

        return T

    def IK(
        self, 
        pose: np.ndarray,
        end: Union[str, rtb.Link, rtb.Gripper],
        start: Union[str, rtb.Link, rtb.Gripper, None] = None,
        q0: Union[np.ndarray, None] = None,
        delta_thresh: float = None,
        max_iter: int = 30,
        max_search: int = 100,
        tol: float = 1e-6,
        mask: Union[np.ndarray, None] = None,
        joint_limits: bool = True,
        name: Literal["GN", "LM", "NR"] = "LM",
        pinv: bool = True,
        pinv_damping: float = 0.01,
        k: float = 1.0,
        method: Literal["chan", "wampler", "sugihara"] = "chan",
    ) -> Tuple[np.ndarray, int, int, int, float]:
        """Compute the inverse kinematics for a given frame_to and target position.

        Args:
        -----
            pose (np.ndarray) 
                The desired pose for the designated end link.
            end (union[str, rtb.Link, rtb.Gripper])
                the link considered as the end-effector
            **start (union[str, rtb.Link, rtb.Gripper, None])
                the link considered as the base frame, defaults to the robots's base frame
            **q0 (np.ndarray)
                The initial joint coordinate vector. Provide the cfg for all joints or only
                provide joint values from start to end, an error will be raised otherwise.
            **delta_thresh (float)
                Threshold for joint position change when q0 is provided.
                This is used to constrain the solution to be close to the initial qpos.
            **max_iter (int)
                How many iterations are allowed within a search before a new search
            **max_search (int)
                How many searches are allowed before being deemed unsuccessfull
            **tol (float)
                Maximum allowed residual error E
            **mask (np.ndarray)
                A 6D vector which assigns weights to Cartesian degrees-of-freedom
                error priority
            **joint_limits (bool)
                Reject solutions with joint limit violations
            **name (literal["GN", "LM", "NR"]):
                Name of the numerical IK method to use. Defaults to "LM"
            **pinv (bool)
                Used in "GN" and "NR". Use the psuedo-inverse instad of the normal matrix inverse.
                This arg must be set to True for redundant robots (>6 DoF).
            **pinv_damping (float)
                Used in "GN" and "NR". Damping factor for the psuedo-inverse
            **k (float)
                Used only in "LM". Sets the gain value for the damping matrix Wn in the next iteration.
            **method: literal["chan", "wampler", "sugihara"]
                Used only in "LM". One of "chan", "sugihara" or "wampler". Defines which method is used
                to calculate the damping matrix Wn in the ``step`` method. Default to "chan"
        
        Returns
        -------
            sol
                tuple (q, success, iterations, searches, residual)

            The return value ``sol`` is a tuple with elements:

            ============    ==========  ===============================================
            Element         Type        Description
            ============    ==========  ===============================================
            ``q``           ndarray(n)  joint coordinates in units of radians or metres
            ``success``     int         whether a solution was found
            ``iterations``  int         total number of iterations
            ``searches``    int         total number of searches
            ``residual``    float       final value of cost function
            ============    ==========  ===============================================

            If ``success == 0`` the ``q`` values will be valid numbers, but the
            solution will be in error.  The amount of error is indicated by
            the ``residual``.
        """

        # First: examine the validity of q0
        start_link = self.base_link if start is None else start
        end_link = self.link_map[end] if isinstance(end, str) else end
        middle_links = [end_link]
        while end_link.parent.name != start_link.name:
            middle_links.append(end_link.parent)
            end_link = end_link.parent
            if end_link is None:
                raise ValueError("End link is not a child of start link")
        middle_links = [j for j in middle_links if j.isjoint][::-1]

        if q0 is not None:
            if len(q0) == self.num_dofs:
                tmp_q = []
                for joint in middle_links:
                    tmp_q.append(q0[joint.jindex]) 
                q0 = np.array(tmp_q)
            elif len(q0) == len(middle_links):
                q0 = np.array(q0)
            else:
                raise ValueError("q0 has wrong dimensionalitys")

        # Second: setup IK function and kwargss
        if name == "GN":
            func = self.robot.ik_GN
            kwargs = dict(pinv=pinv, pinv_damping=pinv_damping)
        elif name == "LM":
            func = self.robot.ik_LM
            kwargs = dict(k=k, method=method)
        elif name == "NR":
            func = self.robot.ik_NR
            kwargs = dict(pinv=pinv, pinv_damping=pinv_damping)
        else:
            raise ValueError(f"Invalid IK name {name}")

        # Third: run IK loop
        s = max_search
        sol = None

        while s > 0:
            q, success, iterations, searches, residual = func(
                Tep=pose, 
                end=end,
                start=start, 
                q0=q0, 
                ilimit=max_iter,
                slimit=s, 
                tol=tol, 
                mask=mask, 
                joint_limits=joint_limits,
                **kwargs
            )
            
            s -= searches
            sol = (q, success, iterations, searches, residual) if sol is None or residual < sol[4] else sol

            if success:
                if q0 is not None and delta_thresh is not None and np.linalg.norm(q - q0) > delta_thresh:
                    continue

                return q, success, iterations, searches, residual

        return (sol[0], 0, sol[2], sol[3], sol[4]) # IK failed, return best solution

    ###########################################
    # The following part is for XML exporting #
    ###########################################

    def write_xml(self):
        """Write URDF model to an XML element hierarchy.

        Returns:
            etree.ElementTree: XML data.
        """
        xml_element = self._write_robot(self.urdf)
        return etree.ElementTree(xml_element)

    def write_xml_string(self, **kwargs):
        """Write URDF model to a string.

        Returns:
            str: String of the xml representation of the URDF model.
        """
        xml_element = self.write_xml()
        return etree.tostring(xml_element, xml_declaration=True, *kwargs)

    def write_xml_file(self, fname):
        """Write URDF model to an xml file.

        Args:
            fname (str): Filename of the file to be written. Usually ends in `.urdf`.
        """
        xml_element = self.write_xml()
        xml_element.write(fname, xml_declaration=True, pretty_print=True, encoding="utf-8")

    def _write_robot(self, urdf: rtb_urdf.URDF):
        xml_element = etree.Element("robot", attrib={"name": urdf.name})
        for link in urdf.links:
            self._write_link(xml_element, link)
        for joint in urdf.joints:
            self._write_joint(xml_element, joint)
        for material in urdf._materials:
            self._write_material(xml_element, material)

        return xml_element

    def _write_link(self, xml_parent, link: rtb_urdf.Link):
        xml_element = etree.SubElement(
            xml_parent,
            "link",
            attrib={
                "name": link.name,
            },
        )

        self._write_inertial(xml_element, link.inertial)
        for visual in link.visuals:
            self._write_visual(xml_element, visual)
        for collision in link.collisions:
            self._write_collision(xml_element, collision)

    def _write_inertial(self, xml_parent, inertial:rtb_urdf.Inertial):
        if inertial is None:
            return

        xml_element = etree.SubElement(xml_parent, "inertial")

        self._write_origin(xml_element, inertial.origin)
        self._write_mass(xml_element, inertial.mass)
        self._write_inertia(xml_element, inertial.inertia)

    def _write_origin(self, xml_parent, origin):
        if origin is None:
            return

        etree.SubElement(
            xml_parent,
            "origin",
            attrib={
                "xyz": " ".join(map(str, tra.translation_from_matrix(origin))),
                "rpy": " ".join(map(str, tra.euler_from_matrix(origin))),
            },
        )

    def _write_inertia(self, xml_parent, inertia):
        if inertia is None:
            return None

        etree.SubElement(
            xml_parent,
            "inertia",
            attrib={
                "ixx": str(inertia[0, 0]),
                "ixy": str(inertia[0, 1]),
                "ixz": str(inertia[0, 2]),
                "iyy": str(inertia[1, 1]),
                "iyz": str(inertia[1, 2]),
                "izz": str(inertia[2, 2]),
            },
        )

    def _write_mass(self, xml_parent, mass):
        if mass is None:
            return

        etree.SubElement(
            xml_parent,
            "mass",
            attrib={
                "value": str(mass),
            },
        )

    def _write_collision(self, xml_parent, collision: rtb_urdf.Collision):
        attrib = {"name": collision.name} if collision.name is not None else {}
        xml_element = etree.SubElement(
            xml_parent,
            "collision",
            attrib=attrib,
        )

        self._write_origin(xml_element, collision.origin)
        self._write_geometry(xml_element, collision.geometry)

    def _write_visual(self, xml_parent, visual: rtb_urdf.Visual):
        attrib = {"name": visual.name} if visual.name is not None else {}
        xml_element = etree.SubElement(
            xml_parent,
            "visual",
            attrib=attrib,
        )

        self._write_origin(xml_element, visual.origin)
        self._write_geometry(xml_element, visual.geometry)
        self._write_material(xml_element, visual.material)

    def _write_geometry(self, xml_parent, geometry: rtb_urdf.Geometry):
        if geometry is None:
            return

        xml_element = etree.SubElement(xml_parent, "geometry")
        if isinstance(geometry.ob, sg.Box):
            self._write_box(xml_element, geometry.box)
        elif isinstance(geometry.ob, sg.Cylinder):
            self._write_cylinder(xml_element, geometry.cylinder)
        elif isinstance(geometry.ob, sg.Sphere):
            self._write_sphere(xml_element, geometry.sphere)
        elif isinstance(geometry.ob, sg.Mesh):
            self._write_mesh(xml_element, geometry.mesh)

    def _write_box(self, xml_parent, box: rtb_urdf.Box):
        etree.SubElement(
            xml_parent, "box", attrib={"size": " ".join(map(str, box.size))}
        )

    def _write_cylinder(self, xml_parent, cylinder: rtb_urdf.Cylinder):
        etree.SubElement(
            xml_parent,
            "cylinder",
            attrib={"radius": str(cylinder.radius), "length": str(cylinder.length)},
        )

    def _write_sphere(self, xml_parent, sphere: rtb_urdf.Sphere):
        etree.SubElement(xml_parent, "sphere", attrib={"radius": str(sphere.radius)})

    def _write_mesh(self, xml_parent, mesh: rtb_urdf.Mesh):
        # TODO: turn into different filename handler
        xml_element = etree.SubElement(
            xml_parent,
            "mesh",
            attrib={"filename": mesh.filename},
        )

        self._write_scale(xml_element, mesh.scale)

    def _write_scale(self, xml_parent, scale):
        if scale is not None:
            if isinstance(scale, float) or isinstance(scale, int):
                xml_parent.set("scale", " ".join([str(scale)] * 3))
            else:
                xml_parent.set("scale", " ".join(map(str, scale)))

    def _write_material(self, xml_parent, material):
        if material is None:
            return

        attrib = {"name": material.name} if material.name is not None else {}
        xml_element = etree.SubElement(
            xml_parent,
            "material",
            attrib=attrib,
        )

        self._write_color(xml_element, material.color)
        self._write_texture(xml_element, material.texture)

    def _write_color(self, xml_parent, color: np.ndarray):
        if color is None:
            return

        etree.SubElement(
            xml_parent, "color", attrib={"rgba": " ".join(map(str, color))}
        )

    def _write_texture(self, xml_parent, texture):
        if texture is None:
            return

        # TODO: use texture filename handler
        etree.SubElement(xml_parent, "texture", attrib={"filename": texture.filename})

    def _write_joint(self, xml_parent: etree.Element, joint: rtb_urdf.Joint):
        xml_element = etree.SubElement(
            xml_parent,
            "joint",
            attrib={
                "name": joint.name,
                "type": joint.joint_type,
            },
        )

        etree.SubElement(xml_element, "parent", attrib={"link": joint.parent})
        etree.SubElement(xml_element, "child", attrib={"link": joint.child})
        self._write_origin(xml_element, joint.origin)
        self._write_axis(xml_element, joint.axis)
        self._write_limit(xml_element, joint.limit)
        self._write_dynamics(xml_element, joint.dynamics)
        self._write_mimic(xml_element, joint.mimic)
        self._write_calibration(xml_element, joint.calibration)
        self._write_safety_controller(xml_element, joint.safety_controller)

    def _write_axis(self, xml_parent, axis):
        if axis is None:
            return

        etree.SubElement(xml_parent, "axis", attrib={"xyz": " ".join(map(str, axis))})

    def _write_limit(self, xml_parent, limit: rtb_urdf.JointLimit):
        if limit is None:
            return

        attrib = {}
        if limit.effort is not None:
            attrib["effort"] = str(limit.effort)
        if limit.velocity is not None:
            attrib["velocity"] = str(limit.velocity)
        if limit.lower is not None:
            attrib["lower"] = str(limit.lower)
        if limit.upper is not None:
            attrib["upper"] = str(limit.upper)

        etree.SubElement(
            xml_parent,
            "limit",
            attrib=attrib,
        )

    def _write_dynamics(self, xml_parent, dynamics: rtb_urdf.JointDynamics):
        if dynamics is None:
            return

        attrib = {}
        if dynamics.damping is not None:
            attrib["damping"] = str(dynamics.damping)
        if dynamics.friction is not None:
            attrib["friction"] = str(dynamics.friction)

        etree.SubElement(
            xml_parent,
            "dynamics",
            attrib=attrib,
        )

    def _write_mimic(self, xml_parent, mimic: rtb_urdf.JointMimic):
        if mimic is None:
            return

        etree.SubElement(
            xml_parent,
            "mimic",
            attrib={
                "joint": mimic.joint,
                "multiplier": str(mimic.multiplier),
                "offset": str(mimic.offset),
            },
        )

    def _write_calibration(self, xml_parent, calibration: rtb_urdf.JointCalibration):
        if calibration is None:
            return

        etree.SubElement(
            xml_parent,
            "calibration",
            attrib={
                "rising": str(calibration.rising),
                "falling": str(calibration.falling),
            },
        )

    def _write_safety_controller(self, xml_parent, safety_controller: rtb_urdf.SafetyController):
        if safety_controller is None:
            return

        etree.SubElement(
            xml_parent,
            "safety_controller",
            attrib={
                "soft_lower_limit": str(safety_controller.soft_lower_limit),
                "soft_upper_limit": str(safety_controller.soft_upper_limit),
                "k_position": str(safety_controller.k_position),
                "k_velocity": str(safety_controller.k_velocity),
            },
        )

    def _write_transmission_joint(self, xml_parent, transmission_joint: rtb_urdf.TransmissionJoint):
        xml_element = etree.SubElement(
            xml_parent,
            "joint",
            attrib={
                "name": str(transmission_joint.name),
            },
        )
        for h in transmission_joint.hardwareInterfaces:
            tmp = etree.SubElement(
                xml_element,
                "hardwareInterface",
            )
            tmp.text = h

    def _write_actuator(self, xml_parent, actuator: rtb_urdf.Actuator):
        xml_element = etree.SubElement(
            xml_parent,
            "actuator",
            attrib={
                "name": str(actuator.name),
            },
        )
        if actuator.mechanicalReduction is not None:
            tmp = etree.SubElement("mechanicalReduction")
            tmp.text = str(actuator.mechanicalReduction)

        for h in actuator.hardwareInterfaces:
            tmp = etree.SubElement(
                xml_element,
                "hardwareInterface",
            )
            tmp.text = h

    def _write_transmission(self, xml_parent, transmission: rtb_urdf.Transmission):
        xml_element = etree.SubElement(
            xml_parent,
            "transmission",
            attrib={
                "name": str(transmission.name),
            },
        )

        for j in transmission.joints:
            self._write_transmission_joint(xml_element, j)

        for a in transmission.actuators:
            self._write_actuator(xml_element, a)


    ################################################
    # The following part is for URDF visualization #
    ################################################
    
    def show(self, collision_geometry=False, callback=None):
        """Open a simpler viewer displaying the URDF model.

        Args:
            collision_geometry (bool, optional): Whether to display the <collision> or <visual> elements. Defaults to False.
        """
        if collision_geometry:
            if self._scene_collision is None:
                raise ValueError(
                    "No collision scene available. Use build_collision_scene_graph=True and load_collision_meshes=True during loading."
                )
            else:
                self._scene_collision.show(callback=callback)
        else:
            if self._scene is None:
                raise ValueError(
                    "No scene available. Use build_scene_graph=True and load_meshes=True during loading."
                )
            elif len(self._scene.bounds_corners) < 1:
                raise ValueError(
                    "Scene is empty, maybe meshes failed to load? Use build_scene_graph=True and load_meshes=True during loading."
                )
            else:
                self._scene.show(callback=callback)


