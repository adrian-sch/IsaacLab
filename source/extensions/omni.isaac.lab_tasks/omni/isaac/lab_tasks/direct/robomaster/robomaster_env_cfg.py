
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns, OffsetCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass

from .robomaster_env_assets import ROBOMASTER_CFG, ARENA_CFG, BOX_01_CFG, BOX_02_CFG, BOX_03_CFG, PALLET_CFG, DRUM_CFG, SHELF_LEG_CFG

@configclass
class RobomasterEnvCfg(DirectRLEnvCfg):

    # TODO flag for when video is recorded
    viewer: ViewerCfg = ViewerCfg(
        eye=(25.0, 25.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        resolution=(1920, 1080),
    )
    
    # env
    episode_length_s = 20.0
    dt = 1.0/120.0              # 120 Hz
    decimation = int(1/(dt*10)) # 10 Hz
    action_scale_x_pos = 2.0
    action_scale_x_neg = 0.5
    action_scale_y = 1.25
    action_scale_ang = 3.14
    
    num_objects = 4

    shelf_width = 0.6
    shelf_length = 0.25
    shelf_scale = 1.0

    lidar_history_length = 3
    
    action_space = 3
    observation_space = {
    "lidar": [lidar_history_length,int(2250/(int(lidar_skip_rays+1)))], # TODO get lidar raycount from sensor config
    "sensor": 3,
    "goal": 4,
    }
    state_space = 0 # only used for RNNs, defined to avoid warning

    num_envs = 1024
    env_spacing = 10.0

    fin_dist = 0.10     # 10 cm
    fin_angle = 0.085   # approx 5 degrees

    # kinematics from https://research.ijcaonline.org/volume113/number3/pxc3901586.pdf
    wheel_radius = 0.05  # radius of the wheel
    wheel_lx = 0.1  # distance between wheels and the base in x
    wheel_ly = 0.1  # distance between wheels and the base in y
    dist = wheel_lx + wheel_ly

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
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
        # cfg = random.choice(objects) # TODO better use torch? so seed is set?
        cfg = objects[i%len(objects)]
        objects_cfgs.append(cfg.replace(prim_path=object_prim_path))
        lidar_prim_paths.append(object_prim_path)
    
    #lidar config
    lidar_scanner_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        mesh_prim_paths=lidar_prim_paths,
        # TODO skip rays cfg
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-135.0, 135.0), horizontal_res=0.12 * int(lidar_skip_rays + 1)),
        offset=OffsetCfg(pos=(0.1, 0.0, 0.083)),
        attach_yaw_only=True,
        debug_vis=False, # TODO flag for when video is recorded
        # TODO add noise back when we learned somthing without
        # drift_range=(-0.05, 0.05), # TODO check drift range
        # accuracy=0.05,
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