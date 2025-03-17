import argparse
from typing import List

import numpy as np
import torch

import genesis as gs

from core.domain_randomizer import DomainRandomizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="warning")

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
        ),
    )

    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="../assets/newton/newton.urdf",
            pos=(0, 0, 0.4),
        ),
    )
    ########################## build ##########################
    n_envs = 2
    scene.build(n_envs=n_envs)

    ########################## domain randomization ##########################
    robot.set_friction_ratio(
        friction_ratio=0.5 + torch.rand(scene.n_envs, robot.n_links),
        link_indices=np.arange(0, robot.n_links),
    )

    # set mass of a single link
    link = robot.get_link("base_link")
    rigid = scene.sim.rigid_solver
    ori_mass = rigid.links_info.inertial_mass.to_numpy()
    print("original mass", link.get_mass(), ori_mass)

    joint_names = [
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
    ]
    motor_dofs = [robot.get_joint(name).dof_idx_local for name in joint_names]

    default_dof_pos = torch.tensor(
        [
            0.0,
            0.8,
            -1.5,
            0.0,
            0.8,
            -1.5,
            0.0,
            1.0,
            -1.5,
            0.0,
            1.0,
            -1.5,
        ],
        device=gs.device
    )

    from core.actuators import LSTMActuator

    actuators: List[LSTMActuator] = []
    motor_model_path="../assets/newton/models/lstm.pth"
    model_params= {
        "hidden_size": 32,
        "num_layers": 1
    }

    for i in range(12):
        actuator = LSTMActuator(
            scene=scene,
            motor_model_path=motor_model_path,
            model_params=model_params,
            device=gs.device,
        )
        actuator.build()
        actuators.append(actuator)

    num_actions = 12

    actions = torch.zeros((n_envs, num_actions), device=gs.device, dtype=gs.tc_float)


    idx = 0
    reset = False
    domain_rand = DomainRandomizer(scene, robot, n_envs, joint_names)

    while True:
        scene.step()
        idx += 1

        output_current_positions = robot.get_dofs_position(motor_dofs)
        output_current_velocities = robot.get_dofs_velocity(motor_dofs)
        output_target_positions = torch.zeros_like(actions)
        output_target_positions[:] = default_dof_pos

        efforts_to_apply = torch.zeros_like(actions)

        for i, actuator in enumerate(actuators):
            efforts, _ = actuator.step(
                output_current_positions=output_current_positions[:, i],
                output_current_velocities=output_current_velocities[:, i],
                output_target_positions=output_target_positions[:, i],
            )

            efforts_to_apply[:, i] = efforts

        robot.set_dofs_position(efforts_to_apply, motor_dofs, zero_velocity=False)

        # Push the base around every 1000 steps
        if idx % 100 == 0:
            if reset:
                domain_rand.reset()
            else:
                domain_rand.randomize()
                domain_rand.push_xy()

            reset = not reset




if __name__ == "__main__":
    main()
