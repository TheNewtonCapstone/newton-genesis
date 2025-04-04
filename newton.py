import argparse
import os
import pickle
import shutil
import torch

import genesis as gs
import onnxruntime as ort
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from core.envs import NewtonLocomotionAltEnv
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


def setup_experiment(args):
    """Initializes logging and configuration settings."""
    gs.init(logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    if args.train and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_environment(args, terrain_cfg=None):
    """Creates and returns the appropriate environment instance."""
    if args.curriculum:
        if terrain_cfg is None:
            terrain_cfg = TerrainConfig()
        return NewtonCurriculumEnv(
            num_envs=args.num_envs,
            terrain_cfg=terrain_cfg,
            show_viewer=True,
            device=gs.device
        )

    if args.task_name == "locomotion":
        return NewtonLocomotionEnv(
            num_envs=args.num_envs,
            urdf_path="assets/newton/newton.urdf",
            enable_lstm=args.enable_lstm,
            show_viewer=True,
            device=gs.device
        )
    elif args.task_name == "locomotion_alt":
        return NewtonLocomotionAltEnv(
            num_envs=args.num_envs,
            urdf_path="assets/newton/newton.urdf",
            enable_lstm=args.enable_lstm,
            show_viewer=True,
            device=gs.device
        )

def train_model(runner, env, train_cfg, log_dir, max_iterations):
    """Trains the model using OnPolicyRunner."""
    # Get the environment configurations directly from the environment
    env_cfg = env.env_cfg
    obs_cfg = env.obs_cfg
    reward_cfg = env.reward_cfg
    command_cfg = env.command_cfg

    # Save configurations
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
    parser.add_argument("--max_iterations", type=int, default=700)
    parser.add_argument("--task-name", type=str, default="locomotion")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--eval-onnx", action="store_true", default=False)
    parser.add_argument("--curriculum", action="store_true", default=False)
    parser.add_argument("--enable-lstm", action="store_true", default=False)
    parser.add_argument("--custom-config", action="store_true", default=False)
    args = parser.parse_args()

    # Setup Experiment
    log_dir = setup_experiment(args)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Create Environment - environment now manages its own configuration
    env = create_environment(args)

    # Train or Evaluate
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.train:
        train_model(runner, env, train_cfg, log_dir, args.max_iterations)
    else:
        if args.eval_onnx:
            evaluate_onnx_model("logs/newton-walking/model_1000.onnx", env, args.max_iterations)
        else:
            evaluate_model(runner, env, args.max_iterations)


if __name__ == "__main__":
    main()