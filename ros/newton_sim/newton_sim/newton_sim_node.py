import  argparse
import os
import torch
import numpy as np
import genesis as gs
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from std_msgs.msg import  Float32MultiArray

# import your simulation components
from rsl_rl.runners import OnPolicyRunner
from core.envs.newton_locomotion_env import NewtonLocomotionEnv
from core.config.terrain import TerrainConfig
from core.logger.logger import Logger


class NewtonRosNode(Node):
    def __init__(self, args = None):
        super().__init__('newton_node')
        self.env_cfg, self.obs_cfg, self.reward_cfg, self.command_cfg = self.get_cfgs()
        self.terrain_cfg = TerrainConfig()
        print(os.getcwd())
        # node we create only one env for ros2
        # HACK : give the relative path to the urdf file from the current directory.
        # better to use absolute path and update the python path

        self.env = NewtonLocomotionEnv(
            num_envs=1,
            env_cfg=self.env_cfg,
            obs_cfg=self.obs_cfg,
            reward_cfg=self.reward_cfg,
            command_cfg=self.command_cfg,
            urdf_path="../../assets/newton/newton.urdf",
            enable_lstm=False,
            show_viewer=True,
            device=gs.device
        )



        self.logger = Logger(self.env.scene, log_dir="data_logs")
        # // load model
        model_path = "../../logs/newton-walking/model_1000.onnx"
        self.model_path = model_path

        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # if args.use_onnx:
            # # Get model directory and iteration from path
            # model_dir = os.path.dirname(self.model_path)
            # # Load policy from PyTorch
            # runner = OnPolicyRunner(self.env, {}, model_dir, device=gs.device)
            # runner.load(self.model_path)
            # self.policy = runner.get_inference_policy(device=gs.device)
            # self.use_onnx = False

        self.use_onnx = True # HARD CODED for now, we can use launch to extend this

        self.actions_pub = self.create_publisher(
            Float32MultiArray,
            'newton',
            10)

        self.obs, _ = self.env.reset()
        self.timer_period = self.env.dt
        self.timer = self.create_timer(self.timer_period, self.simulation_step)

        self.get_logger().info('Publishing model actions')


    def get_cfgs(self):
        """Get environment configurations"""

        env_cfg = {
            "num_actions": 8,  # 8 DOF
            "default_joint_angles": {
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
                "FL_HFE", "FL_KFE",
                "FR_HFE", "FR_KFE",
                "HL_HFE", "HL_KFE",
                "HR_HFE", "HR_KFE",
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
            "kp": 10.0,
            "kd": 0.5,
            "termination_if_roll_greater_than": 10,
            "termination_if_pitch_greater_than": 10,
            "base_init_pos": [0.0, 0.0, 0.40],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 20.0,
            "resampling_time_s": 4.0,
            "action_scale": 0.25,
            "simulate_action_latency": True,
            "clip_actions": 100.0,
            "random_reset_pose": False,
        }

        obs_cfg = {
            "num_obs": 33,
            "obs_scales": {
                "lin_vel": 2.0,
                "ang_vel": 0.25,
                "dof_pos": 1.0,
                "dof_vel": 0.05,
            },
        }

        reward_cfg = {
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
                "feet_height": -4.0,
            },
        }

        command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.0, 1.0],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [0.0, 0.0],
        }

        return env_cfg, obs_cfg, reward_cfg, command_cfg


    def simulation_step(self):
        with torch.no_grad():
            if self.use_onnx:
                # Convert PyTorch tensor to numpy for ONNX
                obs = self.obs.cpu().numpy().astype(np.float32)

                # Run ONNX inference
                actions_np = self.session.run([self.output_name], {self.input_name: obs})[0]

                # Convert back to tensor
                actions = torch.tensor(actions_np, dtype=torch.float32, device=self.obs.device)
            else:
                actions = self.policy(self.obs)

        # Publish model's recommended actions
        self.publish_actions(actions)

        # Step the simulation with the same actions
        self.obs, _, _, dones, _ = self.env.step(actions)

        # Reset if necessary
        if dones[0]:
            self.obs, _ = self.env.reset()

    def publish_actions(self, actions):
        """Publish the model's recommended actions to ROS2"""
        # Create message
        msg = Float32MultiArray()

        actions = actions[0].cpu().numpy().tolist()
        msg.data = actions
        # print(msg.data)

        # Publish
        self.actions_pub.publish(msg)



def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False,
                        help="Path to the model file (.pt or .onnx)")
    parser.add_argument("--use-onnx", action="store_true", default=False,
                        help="Use ONNX model instead of PyTorch")
    parser.add_argument("--enable-lstm", action="store_true", default=False,
                        help="Enable LSTM actuator")
    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    # Initialize Genesis
    gs.init(logging_level="warning")

    # Create the model publisher node
    node = NewtonRosNode(args)

    try:
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

