# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns, OffsetCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Pre-defined configs
##

# TODO add Roboamster config        
# model_dir_path = "/home/admin-jfinke/Projects"
model_dir_path = os.path.abspath("../isaac_models")
robomaster_usd_path = model_dir_path + "/robomaster_model/usd_files/robomaster_usd_texturesV2/Robomaster_visuals_v2_glide.usd"
# robomaster_usd_path = model_dir_path + "/robomaster_model/usd_files/robomaster_usd_texturesV2/Robomaster_visuals_v2_simplified_highres_viz.usd"
arena_usd_path = model_dir_path + "/Isaac_RL_Stage_Blender/rl_stage.usd"


ROBOMASTER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/AgilexRobotics/limo/limo.usd", # TODO get model from nucleus server
        usd_path=robomaster_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1)
    ),
    actuators={
        "base_link": ImplicitActuatorCfg(
            joint_names_expr=["base.*"],
            stiffness=0.0,
            # damping=0.001,
            damping=1000.0,
        ),
    },
)

ARENA_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Arena",
        spawn=sim_utils.UsdFileCfg(
            usd_path=arena_usd_path,
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

KLT_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objects/KLT_0",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            scale=(10.0, 10.0, 10.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

CUBE_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube_0",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.3, 0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
        # init_state=RigidObjectCfg.InitialStateCfg(),
    )

def sample_circle(max_radius, min_radius, size = torch.Size([1,3]), z = 0.0, device = None):
    # sample uniformly from a circle with a maximum radius of max_radius with a height of z from a circle aligned with the z-axis 
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly


    r = torch.sqrt(torch.rand(size[0]) * (max_radius**2 - min_radius**2) + min_radius**2)

    # r = max_radius * torch.sqrt(torch.rand(size[0]))
    theta = 2 * math.pi * torch.rand(size[0])

    points = torch.zeros(size, device=device)
    points[:, 0] = r * torch.cos(theta)
    points[:, 1] = r * torch.sin(theta)
    points[:, 2] = z

    return points

def sample_yaw(size, device = None):

    quat = torch.zeros(size=(size, 4), device=device)
    yaw = torch.rand(size) * 2 * math.pi
    quat[:, 1] = torch.cos(yaw / 2)
    quat[:, 2] = torch.sin(yaw / 2)
    return quat

@configclass
class RobomasterGlideEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 30 # 10 Hz
    action_scale_x_pos = 3.5
    action_scale_x_neg = 0.5
    action_scale_y = 2.0
    action_scale_ang = 3.14

    action_space = 3
    observation_space = 3
    state_space = 0

    num_envs = 1024
    env_spacing = 10.0

    fin_dist = 0.25

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 300, # 300 Hz
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0, # TODO check friction
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0, # TODO check friction
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ROBOMASTER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 0.6
    moment_scale = 0.01

    # -- Goals
    goal_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=fin_dist,
                height=0.2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    # walls
    arena: RigidObjectCfg = ARENA_CFG.replace(prim_path="/World/envs/env_.*/Arena")

    # objects
    num_objects = 0
    objects_cfgs = []
    object_prim_paths = []
    # Rigid Object
    for i in range(num_objects):
        object_prim_path = f"/World/envs/env_.*/object_{i}"
        objects_cfgs.append(CUBE_CFG.replace(prim_path=object_prim_path))
        object_prim_paths.append(object_prim_path)
    
    #lidar config
    lidar_scanner_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        mesh_prim_paths=["/World/ground"] + object_prim_paths,
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-135.0, 135.0), horizontal_res=0.12),
        offset=OffsetCfg(pos=(0.1, 0.0, 0.083)),
        attach_yaw_only=False,
        debug_vis=False,
        max_distance=10.0
    )

    # TODO reward sclaes
    # reward scales
    # lin_vel_reward_scale = 1.0
    # yaw_rate_reward_scale = 0.5
    # z_vel_reward_scale = -2.0
    # ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -2.5e-7
    # action_rate_reward_scale = -0.01
    # feet_air_time_reward_scale = 0.5
    # undersired_contact_reward_scale = -1.0
    # flat_orientation_reward_scale = -5.0


class RobomasterGlideEnv(DirectRLEnv):
    cfg: RobomasterGlideEnvCfg

    def __init__(self, cfg: RobomasterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self._dist_to_goal_buf = torch.zeros(self.num_envs, device=self.device)

        # randomize goals
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos[:, :2].uniform_(-(self.cfg.env_spacing - (2*self.cfg.fin_dist)) / 2, (self.cfg.env_spacing - (2*self.cfg.fin_dist)) / 2)
        self.goal_pos += self.scene.env_origins
        self.goal_pos[:, 2] = 0.1 # set goal on ground

        # TODO debug
        self._cur_step = 0

        # TODO logging
        # Logging
        # self._episode_sums = {
        #     key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        #     for key in [
        #         "track_lin_vel_xy_exp",
        #         "track_ang_vel_z_exp",
        #         "lin_vel_z_l2",
        #         "ang_vel_xy_l2",
        #         "dof_torques_l2",
        #         "dof_acc_l2",
        #         "action_rate_l2",
        #         "feet_air_time",
        #         "undesired_contacts",
        #         "flat_orientation_l2",
        #     ]
        # }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._arena = RigidObject(self.cfg.arena)
        self.scene.rigid_objects["arena"] = self._arena

        self._objects = []
        for i in range (self.cfg.num_objects):
            object_cfg = self.cfg.objects_cfgs[i]
            object = RigidObject(object_cfg)
            self._objects.append(object)
            self.scene.rigid_objects[f"object_{i}"] = object

        self._goal_viz = VisualizationMarkers(self.cfg.goal_marker_cfg)

        # self._lidar_scanner = RayCaster(self.cfg.lidar_scanner_cfg)
        # self.scene.sensors["lidar_scanner"] = self._lidar_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._cur_step += 1
        self._actions = actions.clone()

        self._thrust[:, 0, :2] = self.cfg.thrust_to_weight * self._robot_weight * self._actions[:, :2]
        self._moment[:, 0, 2] = self.cfg.moment_scale * self._actions[:, 2]

    def _apply_action(self):     
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        quad = self._robot.data.root_quat_w
        w = quad[:, 0]
        x = quad[:, 1]
        y = quad[:, 2]
        z = quad[:, 3]        
        robo_yaw = torch.atan2(2.0 * (x*y + w*z), w*w + x*x - y*y - z*z).unsqueeze(1)

        dist_to_goal = torch.norm(self.goal_pos - self._robot.data.root_pos_w, dim=-1).unsqueeze(1) 

        rel_goal_pos = self.goal_pos - self._robot.data.root_pos_w
        #rotate around z-axis to align with robot
        rel_goal_pos_rot = torch.zeros(self.num_envs, 2, device=self.device)
        rel_goal_pos_rot[:, 0] = rel_goal_pos[:, 0] * torch.cos(-robo_yaw.squeeze()) - rel_goal_pos[:, 1] * torch.sin(-robo_yaw.squeeze())
        rel_goal_pos_rot[:, 1] = rel_goal_pos[:, 0] * torch.sin(-robo_yaw.squeeze()) + rel_goal_pos[:, 1] * torch.cos(-robo_yaw.squeeze())

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    rel_goal_pos_rot,
                    dist_to_goal
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        dist_to_goal = torch.norm(self.goal_pos - self._robot.data.root_pos_w, dim=-1)
        

        if self._dist_to_goal_buf is None:
            self._dist_to_goal_buf = dist_to_goal
        reward = (self._dist_to_goal_buf - dist_to_goal) * 10.0
        # reward += (5.0 - dist_to_goal) * 1.0
        self._dist_to_goal_buf = dist_to_goal
        
        reward = torch.where(dist_to_goal < self.cfg.fin_dist, 1000, reward)
        
        # penalty for backwards or sideways movement
        # reward += torch.where(self._actions[:,0] < 0, self._actions[:,0] * 10.0, 0)
        # reward += (torch.abs(self._actions[:,1]) * -0.1)
        
        # penalty for change in action for smoother actions
        reward += (self._previous_actions - self._actions).pow(2).sum(dim=1) * -1.0
        self._previous_actions = self._actions.clone()        

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        #TODO calulate not in every function
        dist_to_goal = torch.sqrt(torch.square(self.goal_pos[:, :2] - self._robot.data.root_pos_w[:,:2]).sum(-1))

        ones = torch.ones_like(time_out)
        died = torch.zeros_like(time_out)

        # check distance to goal
        died = torch.where(dist_to_goal > self.cfg.env_spacing , ones, died)

        # check if robot reached goal
        died = torch.where(dist_to_goal < self.cfg.fin_dist, ones, died)


        # Check if any z-coordinate is greater than 0.2
        indices = torch.nonzero(self._robot.data.root_pos_w[:, 2] > 0.2).squeeze()
        if indices.any():
            # Get the indices where z > 0.2
            print(f"[ROBO WARNING] Indices where z > 0.2: {indices.tolist()} @ step: {self._cur_step}")
            max_z_above_0_2 = self._robot.data.root_pos_w[indices, 2].max()
            print(f"[ROBO WARNING] Max z: {max_z_above_0_2.item()}")
            
        died = torch.where(self._robot.data.root_pos_w[:, 2] > 0.2, ones, died)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        num_resets = len(env_ids)

        self._robot.reset(env_ids)
        if num_resets == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands
        env_positions = self._terrain.env_origins[env_ids]

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += env_positions
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # randomize goals
        goal_pos = torch.zeros(num_resets, 3, device=self.device)

        goal_pos = sample_circle((self.cfg.env_spacing * 0.75) / 2, 1.0, size=goal_pos.size(), z=0.1, device=self.device)
        # goal_pos[:, :2].uniform_(-(self.cfg.env_spacing - (2*self.cfg.fin_dist)) / 2, (self.cfg.env_spacing - (2*self.cfg.fin_dist)) / 2)
        goal_pos += env_positions
        # goal_pos[:, 2] = 0.1 # set goal on ground
        self.goal_pos[env_ids] = goal_pos
        self._goal_viz.visualize(self.goal_pos)


        for object in self._objects:
            object_pose = torch.zeros(num_resets, 7, device=self.device)
            object_pose[:, :3] = sample_circle((self.cfg.env_spacing * 0.75) / 2, 1.0, size=object_pose[:, :3].size(), z=0.1, device=self.device)
            object_pose[:, :3] += env_positions
            object_pose[:, 3:] = sample_yaw(num_resets, device=self.device)

            object.write_root_pose_to_sim(object_pose, env_ids)


        super()._reset_idx(env_ids)
        # TODO logging
        # # Logging
        # extras = dict()
        # for key in self._episode_sums.keys():
        #     episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
        #     extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
        #     self._episode_sums[key][env_ids] = 0.0
        # self.extras["log"] = dict()
        # self.extras["log"].update(extras)
        # extras = dict()
        # extras["Episode Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # self.extras["log"].update(extras)
