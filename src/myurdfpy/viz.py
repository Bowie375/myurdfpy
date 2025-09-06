import logging
from functools import partial

import viser
import trimesh.transformations as tra

from myurdfpy import URDF

_logger = logging.getLogger(__name__)
         
class Visualizer:
    def __init__(
        self, 
        urdf_file: str, 
        mesh_dir: str = "",
        filename_handler: callable = None,
        skip_materials: bool = False,
        use_collision_mesh: bool = False
    ):
        self._urdf_model = URDF.load(
            urdf_file, mesh_dir=mesh_dir, filename_handler=filename_handler, skip_materials=skip_materials)
        self._server = viser.ViserServer()
        self._use_collision_mesh = use_collision_mesh

        self._mesh_handles: dict[str, viser.MeshHandle] = {}
        self._frame_handles: dict[str, viser.FrameHandle] = {}
        self._slider_handles: dict[str, viser.GuiSliderHandle] = {}

    def _add_mesh(self):
        scene = self._urdf_model.collision_scene if self._use_collision_mesh else self._urdf_model.scene

        for i, geometry_name in enumerate(scene.geometry):
            mesh = scene.geometry[geometry_name]
            transform = scene.graph.get(geometry_name)[0]

            ## add mesh
            handle = self._server.scene.add_mesh_trimesh(
                name=f"mesh_{i}",
                mesh=mesh,
                wxyz=tra.quaternion_from_matrix(transform),
                position=transform[:3, 3],
            )
            self._mesh_handles[geometry_name] = handle

            ## add frame            
            handle = self._server.scene.add_frame(
                name=f"frame_{i}",
                axes_length=0.3,
                axes_radius=0.01,
                wxyz=tra.quaternion_from_matrix(transform),
                position=transform[:3, 3],
                visible=False
            )
            self._frame_handles[geometry_name] = handle

    def _add_joint_control(self):

        def callback(e: viser.GuiEvent, joint_name: str = None, cfg: dict|list[float] = None):
            if joint_name is not None:
                self._urdf_model.update_cfg({joint_name: e.target.value})
            else:
                self._urdf_model.update_cfg(configuration=cfg)
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

            if cfg is not None:
                if isinstance(cfg, dict):
                    for joint_name, value in cfg.items():
                        self._slider_handles[joint_name].value = value
                else:
                    for joint_name, value in zip(self._urdf_model.actuated_joint_names, cfg):
                        self._slider_handles[joint_name].value = value

        with self._server.gui.add_folder("Controls"):

            for joint_name in self._urdf_model.actuated_joint_names:
                joint = self._urdf_model.joint_map[joint_name]

                slider = self._server.gui.add_slider(
                    joint_name,
                    min=joint.limit.lower,
                    max=joint.limit.upper,
                    step=0.01,
                    initial_value=max(joint.limit.lower, min(joint.limit.upper, 0.))
                )
                self._slider_handles[joint_name] = slider

                slider.on_update(partial(callback, joint_name=joint_name))

            self._server.gui.add_button("Reset").on_click(partial(callback, cfg=self._urdf_model.zero_cfg))

    def _add_link_click(self):
        for geometry_name, handle in self._mesh_handles.items():
            
            def make_callback(geometry_name):
                def callback(event: viser.SceneNodePointerEvent):
                    # handle.mesh.visual.face_colors[:] = [1.0, 0.0, 0.0, 1.0]
                    v = self._frame_handles[geometry_name].visible
                    self._frame_handles[geometry_name].visible = not v
                return callback

            handle.on_click(make_callback(geometry_name))

    def run(self):
        self._add_mesh()
        self._add_joint_control()
        self._add_link_click()


if __name__ == "__main__":
    pass
