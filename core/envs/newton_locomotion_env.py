from typing import List

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from core.controllers.keyboard_controller import KeyboardController
from core.logger.logger import Logger

from core.domain_randomizer import DomainRandomizer


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class NewtonLocomotionEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, urdf_path=None, show_viewer=False, enable_lstm = False, device="cuda"):
        self.device = torch.device(device)
        self.keyboard_controller = KeyboardController(command_scale=command_cfg["lin_vel_x_range"][1])

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.enable_lstm = enable_lstm

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=4),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        if urdf_path is None:
            print("URDF Path not provided")
            return

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        self.cam = self.scene.add_camera(
            res=(640, 480),
            pos=(4.5, 0.0, 3.5),
            lookat=(0, 0, 0.5),
            fov=40,
            GUI=True,
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.feet_links = [self.robot.get_link(name).idx for name in self.env_cfg["feet_names"]]
        self.feet_links = torch.tensor(self.feet_links)

        if self.enable_lstm:
            from core.actuators import LSTMActuator

            self.hfe_actuators: List[LSTMActuator] = []
            self.kfe_actuators: List[LSTMActuator] = []
            hfe_model_path = "assets/newton/models/lstm_hfe.pth"
            kfe_model_path = "assets/newton/models/lstm_kfe.pth"
            model_params = {
                "hidden_size": 32,
                "num_layers": 1
            }

            for i in range(4):
                actuator = LSTMActuator(
                    scene=self.scene,
                    motor_model_path=hfe_model_path,
                    model_params=model_params,
                    device=gs.device,
                )
                actuator.build()
                self.hfe_actuators.append(actuator)

            for i in range(4):
                actuator = LSTMActuator(
                    scene=self.scene,
                    motor_model_path=kfe_model_path,
                    model_params=model_params,
                    device=gs.device,
                )
                actuator.build()
                self.kfe_actuators.append(actuator)
        else:
        # PD control parameters
            self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
            self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)
            self.robot.set_dofs_force_range([-self.env_cfg["clip_actions"]] * self.num_actions, [self.env_cfg["clip_actions"]] * self.num_actions, self.motor_dofs)


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

        self.domain_randomizer = DomainRandomizer(self.scene, self.robot, self.num_envs , self.env_cfg["dof_names"])
        self.logger = Logger(self.scene)
        self.step_idx = 0

    def update_commands(self, envs_idx):
        command = self.keyboard_controller.get_command()
        if command is not None:
            self.commands[0, 0] = command[0]
            self.commands[0, 1] = command[1]
            self.commands[0, 2] = command[2]

        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(self.command_cfg["lin_vel_x_range"][0], self.command_cfg["lin_vel_x_range"][1], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(self.command_cfg["lin_vel_y_range"][0], self.command_cfg["lin_vel_y_range"][1], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(self.command_cfg["ang_vel_range"][0], self.command_cfg["ang_vel_range"][1], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        target_dof_pos[:] = torch.fmod(target_dof_pos[:] + math.pi, 2 * math.pi) - math.pi
        # if target_dof_pos[:]

        if self.enable_lstm:
            # Implementing Actuators
            output_current_positions = self.robot.get_dofs_position(self.motor_dofs)
            output_current_velocities = self.robot.get_dofs_velocity(self.motor_dofs)

            efforts_to_apply = torch.zeros_like(self.actions)

            for i in range(self.env_cfg["num_actions"]):
                if i % 2 == 0:
                    actuator = self.hfe_actuators[i // 2]
                else:
                    actuator = self.kfe_actuators[i // 2]
                efforts = actuator.step(
                    output_current_positions=output_current_positions[:, i],
                    output_current_velocities=output_current_velocities[:, i],
                    output_target_positions=target_dof_pos[:, i],
                )

                efforts_to_apply[:, i] = efforts

            self.robot.control_dofs_force(efforts_to_apply, self.motor_dofs)
        else:
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
        self.update_commands(envs_idx)

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

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Push the robots
        if self.step_idx % 100 == 0:
            self.domain_randomizer.push_xy()

        if self.logger:
            # base position and orientation
            self.logger.log_base_pos_and_ori(self.base_pos[0], self.base_quat[0])

            # joint velocities
            self.logger.log_joint_velocities(self.dof_vel[0])

            # joint positions
            self.logger.log_joint_positions(self.dof_pos[0])

            # joint efforts
            joint_efforts = self.robot.get_dofs_force(self.motor_dofs)[0]
            self.logger.log_joint_efforts(joint_efforts)

            # actions
            self.logger.log_actions(self.actions[0])

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

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)
        if self.env_cfg["random_reset_pose"]:
            random_dof_pos = (torch.rand_like(self.dof_pos[envs_idx]) * torch.pi * 2) - torch.pi
            self.robot.set_dofs_position(random_dof_pos, self.motor_dofs, envs_idx=envs_idx)

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

        self.update_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def _get_contact_termination(self):
        # check termination and reset
        contacts = self.robot.get_contacts(self.plane)
        robot_contacts = torch.tensor(contacts["link_b"])
        termination_contacts = torch.isin(robot_contacts, self.feet_links).any(dim=1).to(self.device)
        return termination_contacts


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

    def _reward_feet_height(self):
        # Penalize feet height away from target
        ankle_height = self.robot.get_links_pos()[:, -4:, 2]
        ankle_angle_quat = self.robot.get_links_quat()[:, -4:]
        ankle_angle = quat_to_xyz(ankle_angle_quat)
        feet_height = ankle_height + torch.sin(ankle_angle[:, :, 1]) * 0.175

        return torch.sum(torch.square(feet_height - self.reward_cfg["feet_height_target"]), dim=1)