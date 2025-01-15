
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

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
            # solver_position_iteration_count=8, # dont know from where i got this
            # solver_velocity_iteration_count=2, # dont know from where i got this
            solver_position_iteration_count=4, # from anymal example
            solver_velocity_iteration_count=0,
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
            damping=0.01,
            # damping=1000.0, # dont know why this is here, maybe it has a reason
            effort_limit=1.2,
            velocity_limit=90.0,
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
box_02_path = model_dir_path + "/rl_assets/Box_A10_40x30x34cm_PR_V_NVD_01.usd"
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
box_03_path = model_dir_path + "/rl_assets/PlywoodCrate_B03_200x100x100cm_PR_NV_01.usd"
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
pallet_path = model_dir_path + "/rl_assets/Pallet_Asm_A02_91x91x51cm_PR_V_NVD_01.usd"
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
drum_path = model_dir_path + "/rl_assets/SteelDrum_A01_PR_NVD_01.usd"
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
shelf_path = model_dir_path + "/rl_assets/Simple_Shelf.usd"
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
shelf_leg_path = model_dir_path + "/rl_assets/square_shelf_leg_30mm.usd"
SHELF_LEG_CFG = SHELF_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objects/shelf/leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=shelf_leg_path,
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
