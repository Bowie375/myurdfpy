import sys
import time
import logging

import viser
import numpy as np
import trimesh.transformations as tra

from myurdfpy import URDF

_logger = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging.

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def generate_joint_limit_trajectory(urdf_model: URDF, loop_time: float):
    """Generate a trajectory for all actuated joints that interpolates between joint limits.
    For continuous joint interpolate between [0, 2 * pi].

    Args:
        urdf_model (yourdfpy.URDF): _description_
        loop_time (float): Time in seconds to loop through the trajectory.

    Returns:
        dict: A dictionary over all actuated joints with list of configuration values.
    """
    trajectory_via_points = {}
    for joint_name in urdf_model.actuated_joint_names:
        if urdf_model.joint_map[joint_name].type.lower() == "continuous":
            via_point_0 = 0.0
            via_point_2 = 2.0 * np.pi
            via_point_1 = (via_point_2 - via_point_0) / 2.0
        else:
            limit_lower = (
                urdf_model.joint_map[joint_name].limit.lower
                if urdf_model.joint_map[joint_name].limit.lower is not None
                else -np.pi
            )
            limit_upper = (
                urdf_model.joint_map[joint_name].limit.upper
                if urdf_model.joint_map[joint_name].limit.upper is not None
                else +np.pi
            )
            via_point_0 = limit_lower
            via_point_1 = limit_upper
            via_point_2 = limit_lower

        trajectory_via_points[joint_name] = np.array(
            [
                via_point_0,
                via_point_1,
                via_point_2,
            ]
        )
    times = np.linspace(0.0, 1.0, int(loop_time * 100.0))
    bins = np.arange(3) / 2.0

    # Compute alphas for each time
    inds = np.digitize(times, bins, right=True)
    inds[inds == 0] = 1
    alphas = (bins[inds] - times) / (bins[inds] - bins[inds - 1])

    # Create the new interpolated trajectory
    trajectory = {}
    for k in trajectory_via_points:
        trajectory[k] = (
            alphas * trajectory_via_points[k][inds - 1]
            + (1.0 - alphas) * trajectory_via_points[k][inds]
        )

    return trajectory


def viewer_callback(scene, urdf_model, trajectory, loop_time):
    frame = int(100.0 * (time.time() % loop_time))
    cfg = {k: trajectory[k][frame] for k in trajectory}

    urdf_model.update_cfg(configuration=cfg)

         
class Visualizer:
    def __init__(
        self, 
        urdf_file: str, 
        skip_materials: bool = False,
        use_collision_mesh: bool = False
    ):
        self._urdf_model = URDF.load(urdf_file, skip_materials=skip_materials)
        self._server = viser.ViserServer()
        self._use_collision_mesh = use_collision_mesh

        self._mesh_handles = {}
        self._frame_handles = {}

        self._mesh_colors = {}
        self._highlighted_handle = None

    def _add_mesh(self):
        scene = self._urdf_model.collision_scene if self._use_collision_mesh else self._urdf_model.scene

        with self._server.gui.add_folder("Meshs"):
            # Add all meshes from the scene
            for i, geometry_name in enumerate(scene.geometry):
                with self._server.gui.add_folder(f"{i}"):
                    mesh = scene.geometry[geometry_name]
                    transform = scene.graph.get(geometry_name)[0]

                    handle = self._server.scene.add_mesh_trimesh(
                        name=f"mesh_{i}",
                        mesh=mesh,
                        wxyz=tra.quaternion_from_matrix(transform),
                        position=transform[:3, 3],
                    )
                    self._mesh_handles[geometry_name] = handle

                    handle = self._server.scene.add_frame(
                        name=f"frame_{i}",
                        axes_length=0.1,
                        wxyz=tra.quaternion_from_matrix(transform),
                        position=transform[:3, 3],
                        visible=False
                    )
                    self._frame_handles[geometry_name] = handle

    def _add_joint_control(self):
        with self._server.gui.add_folder("Controls"):
            for joint_name in self._urdf_model.actuated_joint_names:
                joint = self._urdf_model.joint_map[joint_name]

                slider = self._server.gui.add_slider(
                    joint_name,
                    min=joint.limit.lower,
                    max=joint.limit.upper,
                    step=0.01,
                    initial_value=0.,
                )

                def make_callback(joint_name):
                    def callback(e: viser.GuiEvent):
                        self._urdf_model.update_cfg({joint_name: e.target.value})
                        scene = (
                            self._urdf_model.collision_scene
                            if self._use_collision_mesh
                            else self._urdf_model.scene
                        )
                        for geometry_name, handle in self._mesh_handles.items():
                            transform = scene.graph.get(geometry_name)[0]
                            handle.wxyz = tra.quaternion_from_matrix(transform)
                            handle.position = transform[:3, 3]
                            frame_handle = self._frame_handles[geometry_name]
                            frame_handle.wxyz = tra.quaternion_from_matrix(transform)
                            frame_handle.position = transform[:3, 3]

                    return callback

                slider.on_update(make_callback(joint_name))

    def _add_link_click(self):
        for geometry_name, handle in enumerate(self._mesh_handles.items()):
            
            @handle.on_click
            def callback(event: viser.SceneNodePointerEvent):
                handle.mesh.visual.face_colors[:] = [1.0, 0.0, 0.0, 1.0]
                self._frame_handles[geometry_name].visible = True
            return callback

    def run(self):
        self._add_mesh()
        self._add_joint_control()
        #self._add_link_click()


if __name__ == "__main__":
    pass
