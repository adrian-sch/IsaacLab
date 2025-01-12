# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math

# import matplotlib
# matplotlib.use('Qt5Agg')  # You can also try 'Qt5Agg' or 'GTK3Agg'
# import matplotlib.pyplot as plt

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.assets import RigidObject

from .robomaster_env_cfg import RobomasterEnvCfg

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
        self._is_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
                "object_dist_penalty_exp",
                # "object_dist_penalty_lin",
                "finished",
                "contacts",
            ]
        }
        self._finished = 0
        self._contact = 0

        # # Initialize the plot
        # plt.ion()  # Turn on interactive mode
        # self.fig, self.ax = plt.subplots()
        # self.target_vel_line, = self.ax.plot([], [], 'r-', label='Target Velocities')
        # self.joint_vel_line, = self.ax.plot([], [], 'b-', label='Joint Velocities')
        # self.ax.legend()
        # self.ax.set_xlim(0, 100)  # Adjust as needed
        # self.ax.set_ylim(-100, 100)  # Adjust as needed
        # self.target_vel_data = []
        # self.joint_vel_data = []
        # self.time_data = []
        # plt.show(block=False)  # Show the plot window without blocking the code execution



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
        # print("Actions: ", self._processed_actions[0])
        # print("joint_vels: ", self._robot.data.joint_vel[0, self._joint_ids])
        # print("target_vels: ", self._robot.data.joint_vel_target[0, self._joint_ids])

        # # Update the plot data
        # self.time_data.append(len(self.time_data))
        # self.target_vel_data.append(self._robot.data.joint_vel_target[0, self._joint_ids].cpu().numpy())
        # self.joint_vel_data.append(self._robot.data.joint_vel[0, self._joint_ids].cpu().numpy())

        # # Update the plot
        # self.target_vel_line.set_xdata(self.time_data)
        # self.target_vel_line.set_ydata([vel[0] for vel in self.target_vel_data])
        # self.joint_vel_line.set_xdata(self.time_data)
        # self.joint_vel_line.set_ydata([vel[0] for vel in self.joint_vel_data])


        # # Adjust x-axis limits to scroll
        # if len(self.time_data) > 100:  # Adjust the window size as needed
        #     self.ax.set_xlim(self.time_data[-100], self.time_data[-1])

        # self.ax.relim()
        # self.ax.autoscale_view()
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

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
        # reward += torch.where(torch.any(self._dist_to_objecs < 1.5, dim = -1), torch.min(self._dist_to_objects, dim = -1)[0] - 2.0, 0)
        
        # penalty for distance to objects based on lidar
        object_dist_penalty_exp = -1/(torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values ** 3)
        object_dist_penalty_lin = torch.where(torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values < 1.5, 
                                          (torch.min(self._lidar_scanner.data.ray_distances, dim= -1).values - 1.5)/1.5,
                                          0)

        # finieshed
        finished = torch.where(self._dist_to_goal < self.cfg.fin_dist, 1, 0)
        
        # check for contacts
        self._is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w, dim=-1), dim=1)[0] > 0.0
        )

        # TODO look into anymal example for scaling and step_dt
        rewards = {
            "delta_goal_dist_lin": delta_goal_dist_lin * 50.0,
            "object_dist_penalty_exp": object_dist_penalty_exp,
            # "object_dist_penalty_lin": object_dist_penalty_lin,
            "finished": finished * 10000.0,
            "contacts": self._is_contact * -1000.0,
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
        died = torch.where(self._is_contact, ones, died)
        self._contact = torch.sum(self._is_contact)

        # check if robot reached goal
        reached_goal = self._dist_to_goal < self.cfg.fin_dist
        self._finished = torch.sum(reached_goal)
        died = torch.where(reached_goal, ones, died)
        
        if torch.any(died):
            print(f"Contacts in current Step: {self._contact}")
            print(f"Finished in current Step: {self._finished}")

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

        # place goal visualization
        self.goal_pos[env_ids] = goal_pos[env_ids]
        self._goal_viz.visualize(self.goal_pos)

        # place shelf
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
        # extras["Episode_Termination/fin_or_crash"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/finished"] = self._finished
        extras["Episode_Termination/contacts"] = self._contact
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