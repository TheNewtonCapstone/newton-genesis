import argparse
import os
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner
from core.envs.newton_locomotion_env import NewtonLocomotionEnv

import genesis as gs
import pprint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="newton-walking")
    parser.add_argument("--ckpt", type=int, default=200)
    args = parser.parse_args()

    gs.init()

    log_dir = f"../../logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"../../logs/{args.exp_name}/cfgs.pkl", "rb"))
    pprint.pprint(env_cfg)
    pprint.pprint(obs_cfg)
    pprint.pprint(reward_cfg)
    pprint.pprint(command_cfg)

    reward_cfg["reward_scales"] = {}

    env = NewtonLocomotionEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_path="../../assets/newton/newton.urdf"
    )

    print("Exporting model to ONNX format...")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path, load_optimizer=False)
    actor_critic = runner.alg.actor_critic

    actor_critic.eval()
    actor_critic = actor_critic.to("cpu")

    dummy_input = torch.zeros(1, env.num_obs, device="cpu")

    # Define export path
    onnx_path = os.path.join(log_dir, f"model_{args.ckpt}.onnx")

    # Export only the actor part (policy network)
    torch.onnx.export(
        actor_critic.actor,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    main()
