import argparse
import os
import pickle
import shutil
import torch

import genesis as gs
import onnxruntime as ort
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from core.envs.newton_curriculum_env import NewtonCurriculumEnv
from core.envs.newton_locomotion_env import NewtonLocomotionEnv
from core.config.terrain import TerrainConfig


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_HAA": 0.0,
            "FR_HAA": 0.0,
            "HL_HAA": 0.0,
            "HR_HAA": 0.0,
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
            "FR_HAA",
            "FR_HFE",
            "FR_KFE",
            "FL_HAA",
            "FL_HFE",
            "FL_KFE",
            "HR_HAA",
            "HR_HFE",
            "HR_KFE",
            "HL_HAA",
            "HL_HFE",
            "HL_KFE",
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
        # PD
        "kp":10.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 45,  # degree
        "termination_if_pitch_greater_than": 45,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.30],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        # "base_height_target": 0.3,
        "feet_height_target": 0.16,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            # "lin_vel_z": -1.0,
            # "base_height": 0.0,
            "action_rate": -0.005,
            "similar_to_default": -0.05,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.2, 0.5],
        "lin_vel_y_range": [-0.2, 0.2],
        "ang_vel_range": [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def setup_experiment(args):
    """Initializes logging and configuration settings."""
    gs.init(logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    if args.train and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_environment(args, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg):
    """Creates and returns the appropriate environment instance."""
    if args.curriculum:
        return NewtonCurriculumEnv(
            num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
            reward_cfg=reward_cfg, command_cfg=command_cfg, terrain_cfg=terrain_cfg,
            show_viewer=True, device=gs.device
        )
    return NewtonLocomotionEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        urdf_path="assets/newton/newton.urdf",
        show_viewer=True, device=gs.device
    )


def train_model(runner, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, log_dir, max_iterations):
    """Trains the model using OnPolicyRunner."""
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)


def evaluate_model(runner, env, max_iterations):
    """Loads the trained model and runs evaluation."""
    resume_path = os.path.join(runner.log_dir, f"model_{max_iterations}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


def evaluate_onnx_model(onnx_model_path, env, max_iterations):
    """Loads an ONNX model and runs evaluation in the simulation environment."""
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Get input and output names from ONNX model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    obs, _ = env.reset()

    with torch.no_grad():
        while True:
            # Convert PyTorch tensor to numpy (ONNX expects numpy arrays)
            obs_np = obs.cpu().numpy().astype(np.float32)

            # Run inference
            actions_np = session.run([output_name], {input_name: obs_np})[0]

            # Convert actions back to tensor
            actions = torch.tensor(actions_np, dtype=torch.float32, device=obs.device)

            obs, _, rews, dones, infos = env.step(actions)

def main():
    """Main function to parse arguments and run training or evaluation."""
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="newton-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--eval-onnx", action="store_true", default=False)
    parser.add_argument("--curriculum", action="store_true", default=False)
    args = parser.parse_args()

    # Setup Experiment
    log_dir = setup_experiment(args)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    terrain_cfg = TerrainConfig()

    # Create Environment
    env = create_environment(args, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg)

    # Train or Evaluate
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.train:
        train_model(runner, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, log_dir, args.max_iterations)
    else:
        if args.eval_onnx:
            evaluate_onnx_model("logs/newton-walking/model_200.onnx", env, args.max_iterations)
        else:
            evaluate_model(runner, env, args.max_iterations)


if __name__ == "__main__":
    main()
