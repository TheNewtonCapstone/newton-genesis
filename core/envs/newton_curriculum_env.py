import math
from typing import Dict, Any, Optional

import genesis as gs
import torch
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from core.config import TerrainConfig
from core.controllers import KeyboardController
from core.domain_randomizer import DomainRandomizer, gaussian_noise
from core.terrain import Terrain


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class NewtonCurriculumEnv:
    @classmethod
    def get_default_env_config(cls) -> Dict[str, Any]:
        """Get default environment configuration."""
        return {
            "num_actions": 8,  # 8 DOF
            "default_joint_angles": {  # [rad]
                "FL_HFE": 0.8,
                "FR_HFE": 0.8,
                "HL_HFE": 0.8,
                "HR_HFE": 0.8,
                "FL_KFE": -1.4,
                "FR_KFE": -1.4,
                "HL_KFE": -1.4,
                "HR_KFE": -1.4,
            },
            "dof_names": [
                "FL_HFE",
                "FL_KFE",
                "FR_HFE",
                "FR_KFE",
                "HL_HFE",
                "HL_KFE",
                "HR_HFE",
                "HR_KFE",
            ],
            "contact_names": [
                "base_link",
                "FR_SHOULDER",
                "FL_SHOULDER",
                "HR_SHOULDER",
                "HL_SHOULDER",
                "FL_UPPER_LEG",
                "FR_UPPER_LEG",
                "HL_UPPER_LEG",
                "HR_UPPER_LEG",
            ],
            "feet_names": [
                "FL_LOWER_LEG",
                "FR_LOWER_LEG",
                "HL_LOWER_LEG",
                "HR_LOWER_LEG",
            ],
            "links_to_keep": [
                "FL_UPPER_LEG",
                "FR_UPPER_LEG",
                "HL_UPPER_LEG",
                "HR_UPPER_LEG",
                "FL_LOWER_LEG",
                "FR_LOWER_LEG",
                "HL_LOWER_LEG",
                "HR_LOWER_LEG",
            ],
            # PD
            "kp": 10.0,
            "kd": 0.5,
            # termination
            "termination_if_roll_greater_than": 10,  # degree
            "termination_if_pitch_greater_than": 10,
            # base pose
            "base_init_pos": [0.0, 0.0, 0.30],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 20.0,
            "resampling_time_s": 4.0,
            "action_scale": 0.25,
            "simulate_action_latency": True,
            "clip_actions": 100.0,
            "random_reset_pose": False,
        }

    @classmethod
    def get_default_obs_config(cls) -> Dict[str, Any]:
        """Get default observation configuration."""
        return {
            "num_obs": 33,  # 8 DOF
            "obs_scales": {
                "lin_vel": 2.0,
                "ang_vel": 0.25,
                "dof_pos": 1.0,
                "dof_vel": 0.05,
            },
        }

    @classmethod
    def get_default_reward_config(cls) -> Dict[str, Any]:
        """Get default reward configuration."""
        return {
            "tracking_sigma": 0.25,
            "base_height_target": 0.3,
            "feet_height_target": 0.1,
            "reward_scales": {
                "tracking_lin_vel": 2.0,
                "tracking_ang_vel": 0.2,
                "lin_vel_z": -1.0,
                "base_height": -50.0,
                "action_rate": -0.05,
                "similar_to_default": -0.05,
                # "feet_height": -4.0,
            },
        }

    @classmethod
    def get_default_command_config(cls) -> Dict[str, Any]:
        """Get default command configuration."""
        return {
            "num_commands": 3,
            "lin_vel_x_range": [-1.0, 1.0],
            "lin_vel_y_range": [-1.0, 1.0],
            "ang_vel_range": [0.0, 0.0],
        }

    def __init__(self,
                 num_envs: int,
                 terrain_cfg: TerrainConfig,
                 env_cfg: Optional[Dict[str, Any]] = None,
                 obs_cfg: Optional[Dict[str, Any]] = None,
                 reward_cfg: Optional[Dict[str, Any]] = None,
                 command_cfg: Optional[Dict[str, Any]] = None,
                 urdf_path: str = "assets/newton/newton.urdf",
                 enable_lstm: bool = False,
                 show_viewer: bool = False,
                 device: str = "cuda"):

        self.device = torch.device(device)
        self.enable_lstm = enable_lstm
        self.urdf_path = urdf_path

        # Load default configurations
        self.env_cfg = self.get_default_env_config() if env_cfg is None else env_cfg
        self.obs_cfg = self.get_default_obs_config() if obs_cfg is None else obs_cfg
        self.reward_cfg = self.get_default_reward_config() if reward_cfg is None else reward_cfg
        self.command_cfg = self.get_default_command_config() if command_cfg is None else command_cfg

        self.dt = 0.02  # control frequency on real robot is 50hz
        self.num_envs = num_envs
        self.num_obs = self.obs_cfg["num_obs"]
        self.simulate_action_latency = self.env_cfg.get("simulate_action_latency", False)
        self.num_actions = self.env_cfg["num_actions"]
        self.num_commands = self.command_cfg["num_commands"]
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)
        self.num_privileged_obs = None

        self.obs_scales = self.obs_cfg["obs_scales"]
        self.reward_scales = self.reward_cfg["reward_scales"]

        self.keyboard_controller = KeyboardController(command_scale=self.command_cfg["lin_vel_x_range"][1])

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=16),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.terrain = Terrain(terrain_cfg)
        self.terrain.build()
        self.subterrain_origins = self.terrain.get_subterrain_origins()
        self.scene.add_entity(
            gs.morphs.Terrain(
                horizontal_scale=self.terrain.horizontal_scale,
                vertical_scale=self.terrain.vertical_scale,
                height_field=self.terrain.get_height_field(),
            )
        )
        self.curriculum_levels = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)

        self.base_init_pos = torch.tensor(self.subterrain_origins[0], device=self.device)
        self.base_init_pos[2] += 0.3
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_path,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

        self.domain_randomizer = DomainRandomizer(self.scene, self.robot, self.num_envs, self.env_cfg["dof_names"])
        self.step_idx = 0

        # To visualize the origins of each subterrain
        self.scene.draw_debug_spheres(poss=self.subterrain_origins, radius=0.2)

        newton_spawn_position = self.subterrain_origins.copy()
        newton_spawn_position[:, 2] += 0.3
        self.scene.draw_debug_spheres(poss=newton_spawn_position, radius=0.2)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()
        self.step_idx += 1

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        base_ang_vel = gaussian_noise(self.base_ang_vel, std=0.5)
        projected_gravity = gaussian_noise(self.projected_gravity, std=0.1)
        dof_pos = gaussian_noise(self.dof_pos, std=0.25)
        dof_vel = gaussian_noise(self.dof_vel, std=0.5)

        # compute observations
        self.obs_buf = torch.cat(
            [
                base_ang_vel * self.obs_scales["ang_vel"],  # 3
                projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Push the robots
        if self.step_idx % 100 == 0:
            self.domain_randomizer.push_xy()

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )


        self._update_terrain_curriculum(envs_idx)

        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.domain_randomizer.randomize(envs_idx)
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _update_terrain_curriculum(self, indices=None) -> None:
        if indices is None:
            return

        obs = self.get_observations()
        flat_origins = torch.tensor(
            self.terrain.get_subterrain_origins(),
            dtype=torch.float32,
            device=self.device,
        ).clone()
        flat_origins[:, 2] += 0.3
        sub_terrain_length = self.terrain.subterrain_size[0]

        level_indices = self.curriculum_levels[indices].long()

        # The levl is updated based on the distance traversed by the agent
        distance = self.robot.get_pos()[indices, :2] - flat_origins[level_indices, :2]
        distance = torch.norm(distance, dim=1)
        move_up = distance >= sub_terrain_length / 2
        move_down = distance < sub_terrain_length / 3

        # Update the Newton levels
        self.curriculum_levels[indices] += 1 * move_up - 1 * move_down

        # Ensure levels stay within bounds
        max_level = self.terrain.num_sub_terrains - 1  # Max valid sub-terrain index
        self.curriculum_levels[indices] = torch.clamp(
            self.curriculum_levels[indices],
            min=0,
            max=max_level,
        )

        self.curriculum_levels[indices] = torch.where(self.curriculum_levels[indices] >= max_level,
                                                      torch.randint_like(self.curriculum_levels[indices], max_level),
                                                      torch.clip(self.curriculum_levels[indices], 0, max_level))

        # Ensure newton_levels is a valid index type
        level_indices = self.curriculum_levels[indices].long()

        # Get new spawn positions based on the levels
        new_spawn_positions = flat_origins[level_indices, :]

        # Update the initial positions in the environment
        self.robot.set_pos(pos=new_spawn_positions, envs_idx=indices)
