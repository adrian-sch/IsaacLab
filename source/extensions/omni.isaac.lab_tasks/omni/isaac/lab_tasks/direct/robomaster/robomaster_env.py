# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .robomaster_env_cfg import RobomasterEnvCfg
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml


class RobomasterEnv(DirectRLEnv):
    cfg: RobomasterEnvCfg

    def __init__(self, cfg: RobomasterEnvCfg, render_mode: str | None = None, **kwargs):
        if kwargs.get("train", False):
            # dump the configuration into log-directory when training
            dump_yaml(os.path.join(kwargs.get("log_root_path", ""), kwargs.get("log_dir", ""), "params", "robomaser_env.yaml"), cfg.cfg_full)
        else:
            # only do scaling when training
            self.cfg.shelf_scale = 1.0

        if kwargs.get("debug", False):
            print(f"RobomasterEnv: {self.cfg}")
            self.cfg.lidar_scanner_cfg.debug_vis = True
            self.cfg.contact_sensor.debug_vis = True

            # set nicer render settings
            self.cfg.sim.render = cfg.debug_render_cfg

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
        self._angle_error_goal = torch.zeros(self.num_envs, device=self.device)
        self._dist_to_goal_buf = torch.zeros(self.num_envs, device=self.device)
        self._dist_to_objects = torch.zeros(self.num_envs, self.cfg.num_objects, device=self.device)
        self._lidar_buf = torch.zeros(self.num_envs, *tuple(self.cfg.observation_space['lidar']), device=self.device)
        self._is_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self._cur_step = 0
        self.shelf_scale = self.cfg.shelf_scale
        
        # Get specific body indices
        self._joint_ids, _ = self._robot.find_joints(["base_lf", "base_rf", "base_lb", "base_rb"])

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "delta_goal_dist_lin",
                "object_dist_penalty_exp",
                # "action_rate",
                "goal_dist_lin",
                # "goal_vel_lin",
                "goal_angle_lin",
                "finished",
                "contacts",
            ]
        }
                
        self._finished = 0
        self._contact = 0

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

        # self._goal_viz = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self._shelf = {}
        for key, shelf_leg_cfg in self.cfg.shelf_cfgs.items():
            shelf_leg = RigidObject(shelf_leg_cfg)
            self._shelf[key] = shelf_leg
            self.scene.rigid_objects[key] = shelf_leg

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
        self._previous_actions = self._actions.clone()

        # robo_odom = self._robot.data.root_pos_w[:, :2]
        robo_lin_vel = self._robot.data.root_lin_vel_b[:, :2]
        robo_ang_vel = self._robot.data.root_ang_vel_b[:, 2]
        
        robo_yaw = yaw_from_quad(self._robot.data.root_quat_w)
        goal_yaw = yaw_from_quad(self.goal_pose[:, 3:])

        self._dist_to_goal = torch.norm(self.goal_pose[:, :3] - self._robot.data.root_pos_w, dim=-1) 
        
        # angle error to goal
        angle_diff = goal_yaw - robo_yaw
        self._angle_error_goal = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        rel_goal_pos = self.goal_pose[:, :3] - self._robot.data.root_pos_w
        #rotate around z-axis to align with robot
        rel_goal_pos_rot = torch.zeros(self.num_envs, 2, device=self.device)
        rel_goal_pos_rot[:, 0] = rel_goal_pos[:, 0] * torch.cos(-robo_yaw.squeeze()) - rel_goal_pos[:, 1] * torch.sin(-robo_yaw.squeeze())
        rel_goal_pos_rot[:, 1] = rel_goal_pos[:, 0] * torch.sin(-robo_yaw.squeeze()) + rel_goal_pos[:, 1] * torch.cos(-robo_yaw.squeeze())

        # normalize lidar data
        lidar_data = self._lidar_scanner.data.ray_distances / self.cfg.lidar_scanner_cfg.max_distance
        # Shift the buffer to the back and nsert the new scan at the front
        self._lidar_buf[:, 1:] = self._lidar_buf[:, :-1]
        self._lidar_buf[:, 0] = lidar_data

        # TODO do we we need this anymore?
        # dist_to_objecs = torch.empty(self.num_envs, 0, device=self.device)
        # # only for testing with GT postions of obstacles
        # for object in self._objects:
        #     dist = torch.norm(object.data.root_pos_w - self._robot.data.root_pos_w, dim=-1).unsqueeze(1)
        #     dist_to_objecs = torch.cat((dist_to_objecs, dist), dim=1)
        
        # self._dist_to_objecs = dist_to_objecs

        obs = {
            "lidar": self._lidar_buf,
            "sensor": 
                torch.cat(
                    [
                        tensor
                        for tensor in (
                            # robo_odom, # TODO I think this is no good for learning, because its absolute and "random"
                            # robo_yaw, # TODO see upper comment
                            robo_lin_vel,
                            robo_ang_vel.unsqueeze(1),
                        )
                        if tensor is not None
                    ],
                    dim=-1,
                ),
        }
        if not self.cfg.goal_only_critic:
            obs["goal"] = torch.cat(
                [
                    tensor
                    for tensor in (
                        rel_goal_pos_rot,
                        self._dist_to_goal.unsqueeze(1),
                        self._angle_error_goal,
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
        else:
            # extend sensor with goal information
            obs["sensor"] = torch.cat(
                [
                    tensor
                    for tensor in (
                        obs["sensor"],
                        rel_goal_pos_rot,
                        self._dist_to_goal.unsqueeze(1),
                        self._angle_error_goal,
                    )
                    if tensor is not None
                ],
                dim=-1,
            )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:        

        if self._dist_to_goal_buf is None:
            self._dist_to_goal_buf = self._dist_to_goal
        delta_goal_dist_lin = (self._dist_to_goal_buf - self._dist_to_goal)
        
        # reward += (5.0 - dist_to_goal) * 1.0
        self._dist_to_goal_buf = self._dist_to_goal

        # penalty for change in action for smoother actions
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        self._previous_actions = self._actions.clone()
        
        # penalty for distance to objects based on lidar
        object_dist_penalty_exp = 1/(torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values ** 3)
        object_dist_penalty_lin = torch.where(torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values < 1.5, 
                                          (torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values - 1.5)/1.5,
                                          0)

        velocity = torch.norm(self._robot.data.root_lin_vel_w, dim=-1) + torch.abs(self._robot.data.root_ang_vel_w[:, 2])
        
        goal_dist_lin = torch.max(1-self._dist_to_goal, torch.zeros_like(self._dist_to_goal))
        
        # adjust error for 180-degree rotation
        angle_error = torch.where(torch.abs(self._angle_error_goal) < math.pi/2,                                                    # when angle is smaller than pi/2
                                  self._angle_error_goal,                                                                           # use the angle
                                  (-torch.sign(self._angle_error_goal)) * (math.pi - torch.abs(self._angle_error_goal))).squeeze()  # else rotate 180 degrees
        goal_angle_lin = torch.where(self._dist_to_goal < 1.0, 
                                     1 - torch.abs(angle_error/(math.pi/2)),            
                                     torch.zeros_like(angle_error)) 
        goal_vel_lin = torch.where(self._dist_to_goal < 1.0, 
                                   velocity, 
                                   torch.zeros_like(velocity))
        
        # print(f"Goal distance: {self._dist_to_goal[0]}")
        # print(f"Goal angle: {angle_error[0]}")
        # print(f"Goal velocity: {velocity[0]}")

        # TODO get velocity threshold from cfg
        self._is_finished = torch.logical_and(
            torch.logical_and(self._dist_to_goal < self.cfg.fin_dist, angle_error < self.cfg.fin_angle),
            velocity < 0.1
        )
        
        # check for contacts
        self._is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w, dim=-1), dim=1)[0] > 0.0
        )

        # TODO move scaling to cfg
        rewards = {
            "delta_goal_dist_lin": delta_goal_dist_lin * self.cfg.delta_goal_dist_lin_scale * self.step_dt,
            "object_dist_penalty_exp": object_dist_penalty_exp * self.cfg.object_dist_penalty_exp_scale * self.step_dt,
            # "action_rate": action_rate * self.cfg.action_rate_scale * self.step_dt,
            "goal_dist_lin": goal_dist_lin * self.cfg.goal_dist_lin_scale * self.step_dt,
            "goal_angle_lin": goal_angle_lin * self.cfg.goal_angle_lin_scale * self.step_dt,
            # "goal_vel_lin": goal_vel_lin * self.cfg.goal_vel_lin_scale * self.step_dt,
            "finished": self._is_finished * self.cfg.finished_scale * self.step_dt,
            "contacts": self._is_contact * self.cfg.contacts_scale * self.step_dt,
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
        
        # TODO maybe we can allow contacts with small force?
        # contacts
        died = torch.where(self._is_contact, ones, died)
        self._contact = torch.sum(self._is_contact)

        # check if robot reached goal and is standing still
        died = torch.where(self._is_finished, ones, died)
        self._finished = torch.sum(self._is_finished)

        # reset flying robots
        # TODO still needed with contact sensor?
        # TODO optinal, check if robo is flipped
        died = torch.where(self._robot.data.root_pos_w[:, 2] > 0.2, ones, died)

        # TODO is this the best position for this?
        # update shelf scale
        if (self._cur_step % self.cfg.shelf_shrink_steps == 0):

            self.shelf_scale = max(1.0, self.shelf_scale - self.cfg.shelf_shrink_by)
            print(f"Shrinking shelf, new shelf scale: {self.shelf_scale}")

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
            object_pose = sample_pose((self.cfg.env_spacing * 0.75) / 2, 1.0, num_envs=num_resets, z=0.0, device=self.device)
            object_pose[:, :3] += env_positions

            object.write_root_pose_to_sim(object_pose, env_ids)

        # randomize shelf position
        goal_pose = self.goal_pose.clone()
        env_positions = self._terrain.env_origins

        # make sure goals are not too close to other objects
        ids = env_ids.clone()
        while True:
            goal_pose[ids] = sample_pose((self.cfg.env_spacing * 0.75) / 2, 1.0, num_envs=len(ids), z=0.1, device=self.device)
            goal_pose[ids, :3] += env_positions[ids]

            #reset ids and find close goals
            ids = torch.empty(0, device=self.device, dtype=torch.long)
            for object in self._objects:
                # calculate dist to objects
                dist = torch.norm(object.data.root_pos_w[env_ids] - goal_pose[env_ids, :3], dim=-1)
                # get ids of close goals
                ids = torch.unique(torch.cat((ids, torch.nonzero(dist < 1.0).view(-1))))

            ids = env_ids[ids]

            if len(ids) == 0:
                break

        # place shelf
        self.goal_pose[env_ids] = goal_pose[env_ids]
        self.place_shelf(env_ids, goal_pose[env_ids])

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
        # extras["Episode_Termination/fin_or_crash"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/finished"] = self._finished
        extras["Episode_Termination/contacts"] = self._contact
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def place_shelf(self, env_ids, goal_pose: torch.Tensor):
        legs_pos = {
            "rf_leg": torch.tensor([self.cfg.shelf_length/2, -self.cfg.shelf_width/2], device=self.device),
            "rb_leg": torch.tensor([-self.cfg.shelf_length/2, -self.cfg.shelf_width/2], device=self.device),
            "lf_leg": torch.tensor([self.cfg.shelf_length/2, self.cfg.shelf_width/2], device=self.device),
            "lb_leg": torch.tensor([-self.cfg.shelf_length/2, self.cfg.shelf_width/2], device=self.device),
        }
        
        for key, leg_pos in legs_pos.items():
            leg_pos = leg_pos * self.shelf_scale
            leg_pos = rotate_vec_2d(leg_pos.unsqueeze(0), goal_pose[:, 3:]).squeeze()
            
            leg_pose = goal_pose.clone()
            leg_pose[:, :2] += leg_pos
            
            self._shelf[key].write_root_pose_to_sim(leg_pose, env_ids)


def sample_pose(max_radius, min_radius, num_envs, z = 0.0, device = None):
    pose = torch.zeros((num_envs, 7), device=device)
    pose[:, :3] = sample_circle(max_radius, min_radius, num_envs=num_envs, z=z, device=device)
    pose[:, 3:] = sample_yaw(num_envs, device=device)
    return pose

def sample_circle(max_radius, min_radius, num_envs, z = 0.0, device = None):
    # sample uniformly from a circle with a maximum radius of max_radius with a height of z from a circle aligned with the z-axis 
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly

    r = torch.sqrt(torch.rand(num_envs) * (max_radius**2 - min_radius**2) + min_radius**2)

    # r = max_radius * torch.sqrt(torch.rand(size[0]))
    theta = 2 * math.pi * torch.rand(num_envs)

    points = torch.zeros((num_envs, 3), device=device)
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

def yaw_from_quad(quad):
        w = quad[:, 0]
        x = quad[:, 1]
        y = quad[:, 2]
        z = quad[:, 3]        
        yaw = torch.atan2(2.0 * (x*y + w*z), w*w + x*x - y*y - z*z).unsqueeze(1)

        return yaw

def rotate_vec_2d(vec, quad):
    # rotate a vector around the z-axis
    yaw = yaw_from_quad(quad)
    
    x = vec[:, 0]
    y = vec[:, 1]
    x_rot = x * torch.cos(yaw) - y * torch.sin(yaw)
    y_rot = x * torch.sin(yaw) + y * torch.cos(yaw)

    return torch.cat((x_rot.unsqueeze(1), y_rot.unsqueeze(1)), dim=-1)