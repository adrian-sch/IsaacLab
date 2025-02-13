
import math
import yaml
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns, OffsetCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.utils.io import load_yaml

from .robomaster_env_assets import ROBOMASTER_CFG, ARENA_CFG, BOX_01_CFG, BOX_02_CFG, BOX_03_CFG, PALLET_CFG, DRUM_CFG, SHELF_LEG_CFG

@configclass
class RobomasterEnvCfg(DirectRLEnvCfg):
    # TODO move all cfgs to yaml and log it when training
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(current_dir, "robomaster_env_cfg.yaml")
    cfg_full = load_yaml(cfg_file)
    cfg = cfg_full.get("params", {})

    # TODO flag for when video is recorded
    viewer: ViewerCfg = ViewerCfg(
        eye=(30.0, 0.0, 7.5),
        lookat=(17.5, 0.0, 0.0),
        resolution=(1920, 1080),
    )
    
    # env
    episode_length_s = cfg["episode_length_s"]
    dt = 1.0/cfg["simulation_rate"]
    decimation = int(1/(dt*cfg["inference_rate"]))     # 10 Hz
    action_scale_x_pos = cfg["action_scale_x_pos"]
    action_scale_x_neg = cfg["action_scale_x_neg"]
    action_scale_y = cfg["action_scale_y"]
    action_scale_ang = cfg["action_scale_ang"]
    
    num_objects = cfg["num_objects"]

    shelf_width = cfg["shelf_width"]
    shelf_length = cfg["shelf_length"]
    shelf_scale = cfg["shelf_scale"]               # starting scale of shelf, shrink by x every y episodes untill 1
    shelf_shrink_steps = cfg["shelf_shrink_epidoes"] * cfg["horizon_length"]     # shrink shelf every y steps, see rl_config for steps per episode (horizon_length)
    shelf_shrink_by = cfg["shelf_shrink_by"]          # shrink self by x: shelf_scale -= shelf_shrink_by

    lidar_skip_rays = cfg["lidar_skip_rays"]             # reduces the amount of rays to simulate a lower resolution and reduce computation, when 0 all rays are used
    lidar_history_length = cfg["lidar_history_length"] 
    
    action_space = 3

    num_rays = int((abs(cfg["lidar_horizontal_fov_range"][0]) + abs(cfg["lidar_horizontal_fov_range"][1]))/cfg["lidar_horizontal_res"])
    
    # If the horizontal field of view is 360 degrees, exclude the last point to avoid overlap
    if abs(abs(cfg["lidar_horizontal_fov_range"][0] - cfg["lidar_horizontal_fov_range"][1]) - 360.0) < 1e-6:
        num_rays -= 1
    
    goal_only_critic = cfg["goal_only_critic"]
    observation_space = {
        "lidar": [lidar_history_length,int(num_rays/(int(lidar_skip_rays+1)))], # TODO get lidar raycount from sensor config
        "sensor": 3 if goal_only_critic else 7,
        # "goal": 4 if goal_only_critic else 0,
    }
    if goal_only_critic:
        observation_space["goal"] = 4
    state_space = 0 # only used for RNNs, defined to avoid warning

    num_envs = cfg["num_envs"]
    env_spacing = cfg["env_spacing"]

    fin_dist = cfg["fin_dist"]
    fin_angle = cfg["fin_angle"]
    fin_lin_vel = cfg["fin_lin_vel"]
    fin_ang_vel = cfg["fin_ang_vel"]
    fin_duration = cfg["fin_duration"]

    # kinematics from https://research.ijcaonline.org/volume113/number3/pxc3901586.pdf
    wheel_radius = cfg["wheel_radius"]  # radius of the wheel
    wheel_lx = cfg["wheel_lx"]  # distance between wheels and the base in x
    wheel_ly = cfg["wheel_ly"]  # distance between wheels and the base in y
    dist = wheel_lx + wheel_ly

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
        ),
        physx=PhysxCfg(
            gpu_max_rigid_patch_count= 2 ** 20,
            gpu_temp_buffer_capacity= 2 ** 28,
            gpu_heap_capacity= 2 ** 30,
        )
    )
    # this will be set in the env script when not training for nicer visualization
    debug_render_cfg = sim_utils.RenderCfg(
            samples_per_pixel=2,
            enable_ambient_occlusion=True,
            dlss_mode=2,
            enable_reflections=True,
            enable_translucency=True,
            enable_global_illumination=True,
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
    # goal_marker_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/goal_marker",
    #     markers={
    #         "cylinder": sim_utils.CylinderCfg(
    #             radius=fin_dist,
    #             height=0.2,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #         ),
    #     },
    # )

    shelf_cfgs = {
        "rf_leg": SHELF_LEG_CFG.replace(prim_path="/World/envs/env_.*/right_front_leg"),
        "rb_leg": SHELF_LEG_CFG.replace(prim_path="/World/envs/env_.*/right_back_leg"),
        "lf_leg": SHELF_LEG_CFG.replace(prim_path="/World/envs/env_.*/left_front_leg"),
        "lb_leg": SHELF_LEG_CFG.replace(prim_path="/World/envs/env_.*/left_back_leg"),
    }

    # walls
    arena: RigidObjectCfg = ARENA_CFG.replace(prim_path="/World/envs/env_.*/Arena")

    # objects
    objects_cfgs = []
    lidar_prim_paths = [
        "/World/envs/env_.*/Arena",
        "/World/envs/env_.*/right_front_leg",
        "/World/envs/env_.*/right_back_leg",
        "/World/envs/env_.*/left_front_leg",
        "/World/envs/env_.*/left_back_leg",
        ]
    
    # objects = [BOX_01_CFG, BOX_02_CFG, BOX_03_CFG, PALLET_CFG, DRUM_CFG]
    objects = [BOX_01_CFG, BOX_01_CFG, BOX_02_CFG, DRUM_CFG, DRUM_CFG, PALLET_CFG]
    # Rigid Object
    for i in range(num_objects):
        object_prim_path = f"/World/envs/env_.*/Object_{i}"
        rigid_object_cfg = objects[i%len(objects)]
        objects_cfgs.append(rigid_object_cfg.replace(prim_path=object_prim_path))
        lidar_prim_paths.append(object_prim_path)
    
    #lidar config
    lidar_scanner_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        mesh_prim_paths=lidar_prim_paths,
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=tuple(cfg["lidar_horizontal_fov_range"]), horizontal_res=cfg["lidar_horizontal_res"] * int(lidar_skip_rays + 1)),
        offset=OffsetCfg(pos=(0.0, 0.0, 0.211)),
        attach_yaw_only=True,
        debug_vis=False,
        drift_range=tuple(cfg["lidar_drift_range"]),
        base_noise=cfg["lidar_base_noise"],
        range_dependet_noise=cfg["lidar_range_dependet_noise"],
        max_distance=cfg["lidar_max_distance"],
    )

    # contact sensor config
    # TODO check update_period, history_length, filter needed?
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        # filter_prim_paths_expr=["/World/envs/env_.*/Object_.*", "/World/envs/env_.*/Shelf", "/World/envs/env_.*/Arena"],
        debug_vis=False,
    )

    # reward scales
    delta_goal_dist_lin_scale = cfg["delta_goal_dist_lin_scale"]
    delta_goal_angel_lin_scale = cfg["delta_goal_angel_lin_scale"]
    object_dist_penalty_exp_scale = cfg["object_dist_penalty_exp_scale"]
    vel_lin_scale = cfg["vel_lin_scale"]
    vel_ang_scale = cfg["vel_ang_scale"]
    action_rate_scale = cfg["action_rate_scale"]
    goal_dist_lin_scale = cfg["goal_dist_lin_scale"]
    goal_angle_lin_scale = cfg["goal_angle_lin_scale"]
    at_goal_scale = cfg["at_goal_scale"]
    finished_scale = cfg["finished_scale"]
    contacts_scale = cfg["contacts_scale"]