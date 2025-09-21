import logging
from functools import partial

import viser
import numpy as np
import trimesh.transformations as tra

from myurdfpy import URDF
# from myurdfpy.urdf_rtb import URDF as URDF_RTB

_logger = logging.getLogger(__name__)
         
class Visualizer:
    def __init__(
        self, 
        urdf_file: str, 
        mesh_dir: str = "",
        filename_handler: callable = None,
        use_collision_mesh: bool = False
    ):
        self._urdf_model = URDF.load(
            urdf_file, mesh_dir=mesh_dir, filename_handler=filename_handler)
        self._server = viser.ViserServer()
        self._use_collision_mesh = use_collision_mesh

        self._mesh_handles: dict[str, viser.MeshHandle] = {}
        self._frame_handles: dict[str, viser.FrameHandle] = {}
        self._slider_handles: dict[str, viser.GuiSliderHandle] = {}
        self._folder_handles: dict[str, viser.GuiFolderHandle] = {}

    def run(self):
        """Main API of this Class"""
        self._create_folders()
        self._add_mesh()
        self._add_link_click()
        self._add_joint_control()
        self._add_ik_viz()


    #################################################
    # The following functions are used to setup GUI #
    #################################################

    def _create_folders(self):
        """Create GUI folders"""
        self._folder_handles["Link Name"] = self._server.gui.add_folder("Link Name")
        self._folder_handles["IK"] = self._server.gui.add_folder("IK")
        self._folder_handles["Controls"] = self._server.gui.add_folder("Controls")

    def _add_mesh(self):
        """Add meshes to viser scene"""

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
        """Add joint control sliders and reset button"""

        # sliders control joint values
        with self._folder_handles["Controls"]:
            self._server.gui.add_button("Reset Joint Values").on_click(
                partial(self._set_joint_value, cfg=self._urdf_model.zero_cfg))

            for joint_name in self._urdf_model.actuated_joint_names:
                joint = self._urdf_model.joint_map[joint_name]
                joint_limit_lower = joint.limit.lower if isinstance(self._urdf_model, URDF) else joint.qlim[0]
                joint_limit_upper = joint.limit.upper if isinstance(self._urdf_model, URDF) else joint.qlim[1]

                slider = self._server.gui.add_slider(
                    joint_name,
                    min=joint_limit_lower,
                    max=joint_limit_upper,
                    step=0.01,
                    initial_value=max(joint_limit_lower, min(joint_limit_upper, 0.))
                )
                self._slider_handles[joint_name] = slider

                slider.on_update(partial(self._set_joint_value, joint_name=joint_name))

    def _add_link_click(self):
        """Add click event to each mesh to toggle visibility of frame and text"""

        with self._folder_handles["Link Name"]:
            self._text_handle = self._server.gui.add_markdown("None")

        for geometry_name, handle in self._mesh_handles.items():
            
            def make_callback(geometry_name):
                def callback(event: viser.SceneNodePointerEvent):
                    # handle.mesh.visual.face_colors[:] = [1.0, 0.0, 0.0, 1.0]
                    v = self._frame_handles[geometry_name].visible
                    self._frame_handles[geometry_name].visible = not v
                    self._text_handle.content = geometry_name
                return callback

            handle.on_click(make_callback(geometry_name))

        def reset_frames(event: viser.GuiEvent):
            for h in self._frame_handles.values():
                h.visible = False

        with self._folder_handles["Controls"]:
            self._server.gui.add_button("Reset Frames").on_click(reset_frames)

    def _add_ik_viz(self):
        """Add IK visualization"""

        def run_ik(event: viser.GuiEvent):
            start_link = self._ik_start_link.value
            end_link = self._ik_end_link.value
            target_xyz = self._ik_target_xyz.value
            target_rpy = self._ik_target_rpy.value
            max_iter = self._ik_max_iter.value
            max_search = self._ik_max_search.value
            tol = self._ik_tol.value
            joint_limits = self._ik_joint_limits.value
            name = self._ik_name.value
            pinv = self._ik_pinv.value
            pinv_damping = self._ik_pinv_damping.value
            k = self._ik_k.value
            method = self._ik_method.value

            try:
                sol = self._urdf_model.IK(
                    pose = tra.compose_matrix(angles=target_rpy, translate=target_xyz),
                    end=end_link, 
                    start=start_link,
                    max_iter=max_iter,
                    max_search=max_search,
                    tol=tol,
                    joint_limits=joint_limits,
                    name=name,
                    pinv=pinv,
                    pinv_damping=pinv_damping,
                    k=k,
                    method=method
                )

                if sol is not None:
                    self._set_joint_value(None, cfg=sol[0])
                    client = self._server._connected_clients[0]
                    client.add_notification(
                        "IK Solution", 
                        "IK {} after {} searches, residual: {:.3f}".format(
                            "failed" if sol[1] == 0 else "succeeded", sol[3], sol[4]), 
                        auto_close_seconds=30.0)

            except Exception as e:
                client = self._server._connected_clients[0]
                client.add_notification("IK Error", str(e), auto_close_seconds=30.0)              

        options = self._urdf_model.link_map.keys()
        with self._folder_handles["IK"]:
            ## GUIs for setting start and end links
            self._ik_start_link: viser.GuiDropdownHandle = self._server.gui.add_dropdown(
                "Start Link", 
                options = options, 
                initial_value = self._urdf_model.base_link)

            self._ik_end_link: viser.GuiDropdownHandle = self._server.gui.add_dropdown(
                "End Link", 
                options = options, 
                initial_value = self._urdf_model.base_link)

            ## GUIs for showing target relative pose
            with self._server.gui.add_folder("Pose (relative):"):
                self._ik_target_xyz = self._server.gui.add_vector3(
                    "xyz", initial_value=np.zeros(3, dtype=np.float32), step=1e-6)
                self._ik_target_rpy = self._server.gui.add_vector3(
                    "rpy", initial_value=np.zeros(3, dtype=np.float32), step=1e-6)

            ## GUIs for configuring IK parameters
            with self._server.gui.add_folder("More Options:", expand_by_default=False):
                self._ik_max_iter = self._server.gui.add_number("Max Iter", min=1, max=100, initial_value=30)
                self._ik_max_search = self._server.gui.add_number("Max Search", min=1, max=100, initial_value=100)
                self._ik_tol = self._server.gui.add_number("Tol", min=1e-6, max=1e-3, initial_value=1e-6)
                self._ik_joint_limits = self._server.gui.add_checkbox("Joint Limits", initial_value=True)
                self._ik_name = self._server.gui.add_dropdown("Name", options=["GN", "LM", "NR"], initial_value="NR")
                self._ik_pinv = self._server.gui.add_checkbox("Pinv(GN, NR)", initial_value=True)
                self._ik_pinv_damping = self._server.gui.add_number(
                    "Pinv Damping(GN, NR)", min=0, max=1, initial_value=0.01)
                self._ik_k = self._server.gui.add_number("K(LM)", min=0, max=1, initial_value=1.0)
                self._ik_method = self._server.gui.add_dropdown(
                    "Method(LM)", options=["chan", "wampler", "sugihara"], initial_value="chan")
            
            self._server.gui.add_button("Run IK").on_click(run_ik)
            self._ik_start_link.on_update(self._get_transform)
            self._ik_end_link.on_update(self._get_transform)
            self._get_transform(None) # refresh at beginning


    #######################################################################
    # The following functions are useful callbacks used by GUI components #
    #######################################################################

    def _set_joint_value(self, e: viser.GuiEvent, joint_name: str = None, cfg: dict|list[float] = None):
        """Do forward kinematics, update sliders and frames"""
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

        self._get_transform(None) # refresh

    def _get_transform(self, e: viser.GuiEvent):
        """Get transformation between Links"""
        start_link = self._ik_start_link.value
        end_link = self._ik_end_link.value
        T = self._urdf_model.get_transform(
            frame_from = start_link, 
            frame_to = end_link,
            collision_geometry=self._use_collision_mesh
        )
        xyz = tra.translation_from_matrix(T)
        rpy = tra.euler_from_matrix(T)
        self._ik_target_xyz.value = xyz
        self._ik_target_rpy.value = rpy

if __name__ == "__main__":
    pass
