# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the ray-caster sensor.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_ray_caster.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.timer import Timer


# TODO add Roboamster config        
model_dir_path = os.path.abspath("../isaac_models")
arena_usd_path = model_dir_path + "/Isaac_RL_Stage_Blender/rl_stage.usd"

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


def define_sensor() -> RayCaster:
    """Defines the ray-caster sensor to add to the scene."""
    # Create a ray-caster sensor
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin.*/ball",
        mesh_prim_paths=[
            "/World/Arena",
            "/World/cube", 
            # "/World/cube2", 
            ],
        # mesh_prim_paths=[],
        # mesh_prim_paths=["/World/cube", "/World/ground", "/World/cube2", "/World/Arena"],
        pattern_cfg=patterns.LidarPatternCfg(channels=32, vertical_fov_range=(-10.0, 10.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=1.2),
        attach_yaw_only=True,
        debug_vis=True,
        max_distance=10.0
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    return ray_caster


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    # -- Rough terrain
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)
    # -- Light
    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)

    spawn_arena_cfg = ARENA_CFG.replace(prim_path="/World/Arena")
    arena_object = RigidObject(cfg=spawn_arena_cfg)

    cone_spawn_cfg = sim_utils.MeshCuboidCfg(
        size=(1.0, 1.0, 1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cone_spawn_cfg.func(
        "/World/cube", cone_spawn_cfg, translation=(2.0, 2.0, 0.5), orientation=(0, 0.0436194, 0, 0.9990482)
    )
    cone2_spawn_cfg = sim_utils.MeshCuboidCfg(
        size=(2.0, 2.0, 1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cone2_spawn_cfg.func(
        "/World/cube2", cone2_spawn_cfg
    )

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[5.0, 5.0, 5.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
    # -- Balls
    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )
    balls = RigidObject(cfg)
    # -- Sensors
    ray_caster = define_sensor()

    # return the scene information
    scene_entities = {"balls": balls, "ray_caster": ray_caster, "arena": arena_object}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # Extract scene_entities for simplified notation
    ray_caster: RayCaster = scene_entities["ray_caster"]
    balls: RigidObject = scene_entities["balls"]

    # define an initial position of the sensor
    ball_default_state = balls.data.default_root_state.clone()

    # ball_default_state[:, :3] = torch.rand_like(ball_default_state[:, :3]) * 10
    ball_default_state[:, 0] = 0.1
    ball_default_state[:, 1] = 0.1
    ball_default_state[:, 2] = 10.0



    print(f"Initial ball state: {ball_default_state}")

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # Reset the scene
        if step_count % 1000 == 0:
            # reset the balls
            balls.write_root_state_to_sim(ball_default_state)
            # reset the sensor
            ray_caster.reset()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the ray-caster
        # with Timer(
        #     f"Ray-caster update with {4} x {ray_caster.num_rays} rays with max height of"
        #     f" {torch.max(ray_caster.data.pos_w).item():.2f}"
        # ):
        
        ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)

        # print(f"Ray-caster: {ray_caster.data}")

        # Update counter
        step_count += 1


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.0, 15.0, 15.0], [0.0, 0.0, -2.5])
    # Design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()