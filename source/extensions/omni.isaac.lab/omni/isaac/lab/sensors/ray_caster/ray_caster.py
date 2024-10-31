# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, UsdPhysics, Usd, Gf
from pxr import UsdGeom, UsdPhysics, Usd, Gf

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.terrains.trimesh.utils import make_plane
from omni.isaac.lab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from omni.isaac.lab.utils.warp import convert_to_warp_mesh, raycast_mesh

from ..sensor_base import SensorBase
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg


class RayCaster(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        # Initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterData()
        # create empty dictionary to store meshes and views
        self._views: dict[str, physx.RigidBodyView] = {}
        self._origin_points: dict[str, np.array] = {}
        self._origin_indices: dict[str, np.array] = {}
        # the warp meshes used for raycasting.
        self.meshes: dict[str, wp.Mesh] = {}

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)


        # TODO debug
        from omni.isaac.lab.utils.timer import Timer
        with Timer("update meshes: "):
            # reinit meshes to update positons
            self._update_warp_meshes(env_ids)

        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # resample the drift
        self.drift[env_ids].uniform_(*self.cfg.drift_range)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrimView(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrimView.")
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")

        # load the meshes by parsing the stage
        self._initialize_warp_meshes()
        # initialize the ray start and directions
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):

        # read prims to ray-cast
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # check if mesh already casted into warp mesh
            if mesh_prim_path in self.meshes:
                continue

            # TODO make this nicer, remove duplicate code
            # check if mesh is view
            if '*' in mesh_prim_path:
                prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    mesh_view = self._physics_sim_view.create_rigid_body_view(mesh_prim_path.replace(".*", "*"))
                    self._views[mesh_prim_path] = mesh_view

                if mesh_view is None:
                    raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {mesh_prim_path}")

                transformations_w = mesh_view.get_transforms()

                for env, prim_path in enumerate(mesh_view.prim_paths):
                    # check if mesh already casted into warp mesh
                    if mesh_prim_path in self.meshes:
                        continue

                    mesh_prim = sim_utils.get_first_matching_child_prim(
                    prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                    )
                    # check if valid
                    if mesh_prim is None or not mesh_prim.IsValid():
                        raise RuntimeError(f"Invalid mesh prim path: {prim_path}")
                    # cast into UsdGeomMesh
                    mesh_prim = UsdGeom.Mesh(mesh_prim)
                    # read the vertices and faces
                    points = np.asarray(mesh_prim.GetPointsAttr().Get())
                    # Transform mesh into world frame

                    pos = Gf.Vec3d(transformations_w[env][0].item(), transformations_w[env][1].item(), transformations_w[env][2].item())
                    rotation = Gf.Rotation(Gf.Quatf(transformations_w[env][3].item(), transformations_w[env][4].item(), transformations_w[env][5].item(), transformations_w[env][6].item()))
                    transform: Gf.Matrix4d = Gf.Matrix4d(rotation, pos)
                    transformed_points_list = []
                    for point in points:
                        transformed_point = transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                        transformed_points_list.append((transformed_point[0], transformed_point[1], transformed_point[2]))

                    # Convert the list to a NumPy array
                    transformed_points = np.asarray(transformed_points_list)

                    indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                    wp_mesh = convert_to_warp_mesh(transformed_points, indices, device=self.device)

                    self._origin_points[prim_path] = points
                    self._origin_indices[prim_path] = indices
                    self.meshes[prim_path] = wp_mesh
                continue


            # check if the prim is a plane - handle PhysX plane as a special case
            # if a plane exists then we need to create an infinite mesh that is a plane
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )
            # if we did not find a plane then we need to read the mesh
            if mesh_prim is None:
                # obtain the mesh prim
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                )
                # check if valid
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
                # cast into UsdGeomMesh
                mesh_prim = UsdGeom.Mesh(mesh_prim)
                # read the vertices and faces
                points = np.asarray(mesh_prim.GetPointsAttr().Get())
                # Transform mesh into world frame
                time = Usd.TimeCode.Default()
                transform : Gf.Matrix4d = mesh_prim.ComputeLocalToWorldTransform(time)
                transformed_points_list = []
                for point in points:
                    transformed_point = transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                    transformed_points_list.append((transformed_point[0], transformed_point[1], transformed_point[2]))

                # Convert the list to a NumPy array
                transformed_points = np.asarray(transformed_points_list)

                indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(transformed_points, indices, device=self.device)

                self._origin_points[mesh_prim_path] = points
                self._origin_indices[mesh_prim_path] = indices
                # print info
                omni.log.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                # print info
                omni.log.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
            # add the warp mesh to the list
            self.meshes[mesh_prim_path] = wp_mesh

        # throw an error if no meshes are found
        if all([mesh_prim_path not in self.meshes for mesh_prim_path in self.cfg.mesh_prim_paths]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )
        
    def _update_warp_meshes(self, env_ids: Sequence[int] | None = None):
        # TODO update all non view meshes 
        if env_ids is not None:
            for env_id in env_ids:
                 for view in self._views:

                    transformations_w = self._views[view].get_transforms()
                    prim_path = self._views[view].prim_paths[env_id]
                                         
                    # read the vertices and faces
                    points = self._origin_points[prim_path]
                    indices = self._origin_indices[prim_path]
                    # Transform mesh into world frame

                    pos = Gf.Vec3d(transformations_w[env_id][0].item(), transformations_w[env_id][1].item(), transformations_w[env_id][2].item())
                    rotation = Gf.Rotation(Gf.Quatf(transformations_w[env_id][3].item(), transformations_w[env_id][4].item(), transformations_w[env_id][5].item(), transformations_w[env_id][6].item()))
                    transform: Gf.Matrix4d = Gf.Matrix4d(rotation, pos)
                    transformed_points_list = []
                    for point in points:
                        transformed_point = transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                        transformed_points_list.append((transformed_point[0], transformed_point[1], transformed_point[2]))

                    # Convert the list to a NumPy array
                    transformed_points = np.asarray(transformed_points_list)
                    wp_mesh = convert_to_warp_mesh(transformed_points, indices, device=self.device)
                    self.meshes[prim_path] = wp_mesh

    def _initialize_rays_impl(self):
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)
        # apply offset transformation to the rays
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos
        # repeat the rays for each sensor
        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self._data.ray_distances = torch.zeros(self._view.count, self.num_rays, device=self._device)
        self._data.ray_distances = torch.zeros(self._view.count, self.num_rays, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrimView):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        # ray cast and store the hits
        ray_hits = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        ray_distances = torch.full((self._view.count, self.num_rays), float('inf'), device=self._device)
        
        # for mesh in self.meshes:
        for mesh in self.meshes:

            new_ray_hits, new_ray_distances, _, _ = raycast_mesh(
                ray_starts_w,
                ray_directions_w,
                max_dist=self.cfg.max_distance,
                mesh=self.meshes[mesh],
                return_distance=True,
            )
            
            # Update ray_distances and ray_hits where the new distances are smaller
            closer_mask = new_ray_distances < ray_distances
            ray_distances = torch.where(closer_mask, new_ray_distances, ray_distances)
            expanded_mask = closer_mask.unsqueeze(-1).expand_as(ray_hits)
            ray_hits = torch.where(expanded_mask, new_ray_hits, ray_hits)

        self._data.ray_hits_w[env_ids] = ray_hits
        self._data.ray_distances[env_ids] = ray_distances

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # show ray hit positions
        self.ray_visualizer.visualize(self._data.ray_hits_w.view(-1, 3))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._view = None
