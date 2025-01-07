# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
import os
import random

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
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
robomaster_usd_path = model_dir_path + "/robomaster_model/usd_files/robomaster_usd_texturesV2/Robomaster_visuals_v2_modified.usd"
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
        activate_contact_sensors=True,
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


box_01_path = model_dir_path + "/rl_assets/Box_A02_60x40x28cm_PR_V_NVD_01.usd"
box_02_path = model_dir_path + "/rl_assets/Box_A10_40x30x34cm_PR_V_NVD_01.usd"
box_03_path = model_dir_path + "/rl_assets/PlywoodCrate_B03_200x100x100cm_PR_NV_01.usd"
pallet_path = model_dir_path + "/rl_assets/Pallet_Asm_A02_91x91x51cm_PR_V_NVD_01.usd"
drum_path = model_dir_path + "/rl_assets/SteelDrum_A01_PR_NVD_01.usd"
shelf_path = model_dir_path + "/rl_assets/Simple_Shelf.usd"
BOX_01_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objects/box_01",
        spawn=sim_utils.UsdFileCfg(
            usd_path=box_01_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
BOX_02_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objects/box_02",
        spawn=sim_utils.UsdFileCfg(
            usd_path=box_02_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
BOX_03_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objects/box_03",
        spawn=sim_utils.UsdFileCfg(
            usd_path=box_03_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
PALLET_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objects/pallet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=pallet_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
DRUM_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objects/pallet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=drum_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
SHELF_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objects/shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=shelf_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.66, 0.66, 0.66)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
        # init_state=RigidObjectCfg.InitialStateCfg(),
    )

@configclass
class RobomasterEnvCfg(DirectRLEnvCfg):

    # TODO flag for when video is recorded
    # viewer: ViewerCfg = ViewerCfg(
    #     eye=(10.0, 10.0, 10.0),
    #     lookat=(0.0, 0.0, 0.0),
    #     resolution=(1920, 1080),
    # )
    
    # env
    episode_length_s = 20.0
    decimation = 30 # 10 Hz
    action_scale_x_pos = 3.5
    action_scale_x_neg = 0.5
    action_scale_y = 2.0
    action_scale_ang = 3.14
    
    num_objects = 6
    
    action_space = 3
    observation_space = {
    "lidar": [3,2250], # TODO get lidar raycount from sensor config
    "sensor": 3
    }
    state_space = 0 # only used for RNNs, defined to avoid warning

    num_envs = 1024
    env_spacing = 10.0

    fin_dist = 0.25

    # kinematics from https://research.ijcaonline.org/volume113/number3/pxc3901586.pdf
    wheel_radius = 0.05  # radius of the wheel
    wheel_lx = 0.1  # distance between wheels and the base in x
    wheel_ly = 0.1  # distance between wheels and the base in y
    dist = wheel_lx + wheel_ly

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
        # TODO flag for when video is recorded
        # TODO only for visualization, reduces performance
        # render=sim_utils.RenderCfg(
        #     samples_per_pixel=2,
        #     enable_ambient_occlusion=True,
        #     dlss_mode=2,
        #     enable_reflections=True,
        #     enable_translucency=True,
        #     enable_global_illumination=True,
        # )
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

    shelf_cfg = SHELF_CFG.replace(prim_path="/World/envs/env_.*/Shelf")

    # walls
    arena: RigidObjectCfg = ARENA_CFG.replace(prim_path="/World/envs/env_.*/Arena")

    # objects
    objects_cfgs = []
    lidar_prim_paths = [
        "/World/envs/env_.*/Arena",
        "/World/envs/env_.*/Shelf",
        ]
    
    # objects = [BOX_01_CFG, BOX_02_CFG, BOX_03_CFG, PALLET_CFG, DRUM_CFG]
    objects = [BOX_01_CFG, BOX_01_CFG, BOX_02_CFG, DRUM_CFG, DRUM_CFG, PALLET_CFG]
    # Rigid Object
    for i in range(num_objects):
        object_prim_path = f"/World/envs/env_.*/Object_{i}"
        # cfg = random.choice(objects) # TODO better use torch? so seed is set?
        cfg = objects[i%len(objects)]
        objects_cfgs.append(cfg.replace(prim_path=object_prim_path))
        lidar_prim_paths.append(object_prim_path)
    
    #lidar config
    lidar_scanner_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        mesh_prim_paths=lidar_prim_paths,
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-135.0, 135.0), horizontal_res=0.12),
        offset=OffsetCfg(pos=(0.1, 0.0, 0.083)),
        attach_yaw_only=True,
        debug_vis=False, # TODO flag for when video is recorded
        max_distance=5.0
    )

    # contact sensor config
    # TODO check update_period, history_length, filter needed?
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        # filter_prim_paths_expr=["/World/envs/env_.*/Object_.*", "/World/envs/env_.*/Shelf", "/World/envs/env_.*/Arena"],
        debug_vis=False, # TODO flag for when video is recorded
    )

    # TODO reward sclaes
    # reward scales


class RobomasterEnv(DirectRLEnv):
    cfg: RobomasterEnvCfg

    def __init__(self, cfg: RobomasterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Kinematic model from https://research.ijcaonline.org/volume113/number3/pxc3901586.pdf
        self.kinematic = 1 / self.cfg.wheel_radius * torch.tensor([
            [1.0, 1.0, -self.cfg.dist],     # left front
            [1.0, -1.0, -self.cfg.dist],    # right front
            [1.0, -1.0, self.cfg.dist],     # left back
            [1.0, 1.0, self.cfg.dist],      # right back
            ],
            device=self.device)
        
        self._dist_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._dist_to_goal_buf = torch.zeros(self.num_envs, device=self.device)
        self._dist_to_objects = torch.zeros(self.num_envs, self.cfg.num_objects, device=self.device)
        self._lidar_buf = torch.zeros(self.num_envs, *tuple(self.cfg.observation_space['lidar']), device=self.device)

        # Get specific body indices
        self._joint_ids, _ = self._robot.find_joints(["base_lf", "base_rf", "base_lb", "base_rb"])

        # randomize goals
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos[:, :2].uniform_(-(self.cfg.env_spacing - (2*self.cfg.fin_dist)) / 2, (self.cfg.env_spacing - (2*self.cfg.fin_dist)) / 2)
        self.goal_pos += self.scene.env_origins
        self.goal_pos[:, 2] = 0.1 # set goal on ground

        # TODO debug
        self._cur_step = 0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "delta_goal_dist_lin",
                "object_dist_penalty",
                "finished",
                "crash",
            ]
        }

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
        self._shelf = RigidObject(self.cfg.shelf_cfg)

        self._lidar_scanner = RayCaster(self.cfg.lidar_scanner_cfg)
        self.scene.sensors["lidar_scanner"] = self._lidar_scanner

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

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

        actions[:, 0] = torch.where(actions[:, 0] > 0, actions[:, 0] * self.cfg.action_scale_x_pos , actions[:, 0] * self.cfg.action_scale_x_neg)
        actions[:, 1] = actions[:, 1] * self.cfg.action_scale_y
        actions[:, 2] = actions[:, 2] * self.cfg.action_scale_ang

        actions = actions.unsqueeze(2)
        self._processed_actions = torch.matmul(self.kinematic, actions).squeeze()

    def _apply_action(self):
        self._robot.set_joint_velocity_target(self._processed_actions, joint_ids = self._joint_ids)

    def _get_observations(self) -> dict:
        quad = self._robot.data.root_quat_w
        w = quad[:, 0]
        x = quad[:, 1]
        y = quad[:, 2]
        z = quad[:, 3]        
        robo_yaw = torch.atan2(2.0 * (x*y + w*z), w*w + x*x - y*y - z*z).unsqueeze(1)

        self._dist_to_goal = torch.norm(self.goal_pos - self._robot.data.root_pos_w, dim=-1) 

        rel_goal_pos = self.goal_pos - self._robot.data.root_pos_w
        #rotate around z-axis to align with robot
        rel_goal_pos_rot = torch.zeros(self.num_envs, 2, device=self.device)
        rel_goal_pos_rot[:, 0] = rel_goal_pos[:, 0] * torch.cos(-robo_yaw.squeeze()) - rel_goal_pos[:, 1] * torch.sin(-robo_yaw.squeeze())
        rel_goal_pos_rot[:, 1] = rel_goal_pos[:, 0] * torch.sin(-robo_yaw.squeeze()) + rel_goal_pos[:, 1] * torch.cos(-robo_yaw.squeeze())

        # normalize lidar data
        lidar_data = self._lidar_scanner.data.ray_distances / self.cfg.lidar_scanner_cfg.max_distance
        # lidar_data = self._lidar_scanner.data.ray_distances

        # Shift the buffer to the back
        self._lidar_buf[:, 1:] = self._lidar_buf[:, :-1]
        # Insert the new scan at the front
        self._lidar_buf[:, 0] = lidar_data

        dist_to_objecs = torch.empty(self.num_envs, 0, device=self.device)
        # only for testing with GT postions of obstacles
        for object in self._objects:
            dist = torch.norm(object.data.root_pos_w - self._robot.data.root_pos_w, dim=-1).unsqueeze(1)
            dist_to_objecs = torch.cat((dist_to_objecs, dist), dim=1)
        
        self._dist_to_objecs = dist_to_objecs

        obs = {
            "lidar": self._lidar_buf,
            "sensor":             
                # TODO check this again, should also work with just the rel_goal_pos_rot ?
                # rel_goal_pos_rot,
                torch.cat(
                    [
                        tensor
                        for tensor in (
                            rel_goal_pos_rot,
                            self._dist_to_goal.unsqueeze(1),
                        )
                        if tensor is not None
                    ],
                    dim=-1,
                ),
        }

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:        

        if self._dist_to_goal_buf is None:
            self._dist_to_goal_buf = self._dist_to_goal
        delta_goal_dist_lin = (self._dist_to_goal_buf - self._dist_to_goal)
        
        # reward += (5.0 - dist_to_goal) * 1.0
        self._dist_to_goal_buf = self._dist_to_goal

        # penalty for distance to objects
        # reward += -1/(self._dist_to_objecs ** 2).sum(dim=-1) * 0.5
        # object_dist_penalty = -1/(torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values ** 2) * 0.5
        # reward += torch.where(torch.any(self._dist_to_objecs < 1.5, dim = -1), torch.min(self._dist_to_objects, dim = -1)[0] - 2.0, 0)
        object_dist_penalty = torch.where(torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values < 1.5, 
                                          (torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values - 1.5)/1.5,
                                          0)

        finished = torch.where(self._dist_to_goal < self.cfg.fin_dist, 1, 0)
        
        # contacts
        is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w, dim=-1), dim=1)[0] > 0.0
        )
        crash = torch.where(is_contact, 1, 0) # TODO check this with an contact sensor

        # TODO look into anymal example for scaling and step_dt
        rewards = {
            "delta_goal_dist_lin": delta_goal_dist_lin * 10.0,
            "object_dist_penalty": object_dist_penalty * 0.5,
            "finished": finished * 10.0,
            "crash": crash * -10.0,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            # print(f"Reward {key}: {value}")
            # print(f"Reward sum {key}: {self._episode_sums[key]}")

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ones = torch.ones_like(time_out)
        died = torch.zeros_like(time_out)
        
        # TODO check this with a contact sensor
        # check distance to objects
        # died = torch.where(torch.any(self._dist_to_objecs < 0.75, dim=-1), ones, died)
        # check distance with lidar, cant do this because of shelf
        # died = torch.where(torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values < self.cfg.fin_dist, ones, died)

        # contacts
        is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w, dim=-1), dim=1)[0] > 0.0
        )
        died = torch.where(is_contact, ones, died)


        # check if robot reached goal
        reached_goal = self._dist_to_goal < self.cfg.fin_dist
        died = torch.where(reached_goal, ones, died)
            
        # reset flying robots
        # TODO optinal? check if robo is flipped
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

        for object in self._objects:
            object_pose = torch.zeros(num_resets, 7, device=self.device)
            # object_pose[:, :3] = sample_circle((self.cfg.env_spacing * 0.75) / 2, ((self.cfg.env_spacing * 0.75) / 2) - 1.5, size=object_pose[:, :3].size(), z=0.1, device=self.device)
            object_pose[:, :3] = sample_circle((self.cfg.env_spacing * 0.75) / 2, 1.0, size=object_pose[:, :3].size(), z=0.0, device=self.device)
            object_pose[:, :3] += env_positions
            object_pose[:, 3:] = sample_yaw(num_resets, device=self.device)

            object.write_root_pose_to_sim(object_pose, env_ids)

        # randomize goals
        goal_pos = self.goal_pos.clone()
        env_positions = self._terrain.env_origins

        close = True
        ids = env_ids.clone()
        count = 0
        while close:
            if count > 1000:
                print(f"Resetting goals {count}, {len(ids)} goals left")
            count += 1
            goal_pos[ids] = sample_circle(((self.cfg.env_spacing * 0.75) / 2) , 1.0, size=goal_pos[ids].size(), z=0.1, device=self.device)
            goal_pos[ids] += env_positions[ids]

            #reset ids and find close goals
            ids = torch.empty(0, device=self.device, dtype=torch.long)
            for object in self._objects:
                # calculate dist to objects
                dist = torch.norm(object.data.root_pos_w[env_ids] - goal_pos[env_ids], dim=-1)
                # get ids of close goals
                ids = torch.unique(torch.cat((ids, torch.nonzero(dist < 1.0).view(-1))))

            ids = env_ids[ids]
            close = len(ids) > 0

        # goal_pos = sample_circle(((self.cfg.env_spacing * 0.75) / 2) - 1.5 , 1.0, size=goal_pos.size(), z=0.1, device=self.device)
        self.goal_pos[env_ids] = goal_pos[env_ids]
        # self._goal_viz.visualize(self.goal_pos)

        goal_pose = torch.zeros(num_resets, 7, device=self.device)
        goal_pose[:, :3] = self.goal_pos[env_ids]
        goal_pose[:, 3:] = sample_yaw(num_resets, device=self.device) # TODO do we need this somewhere else?
        self._shelf.write_root_pose_to_sim(goal_pose, env_ids)

        super()._reset_idx(env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/fin_or_crash"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

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
    quat[:, 0] = torch.cos(yaw / 2)
    quat[:, 3] = torch.sin(yaw / 2)
    return quat