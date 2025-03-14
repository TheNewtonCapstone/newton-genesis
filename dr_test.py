import argparse

import numpy as np
import torch

import genesis as gs

from core.domain_randomizer import DomainRandomizer


def rand_com_shift(n_envs):
    com_shift = 0.5 * (2 * torch.rand(n_envs, 1, 3) - 1)
    com_shift[:, :, 2] = 0.0  # No changes in z-axis
    return com_shift

def rand_mass_shift(n_envs):
    return 0.5 * (2 * torch.rand(n_envs, 1) - 1)


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
            file="assets/newton/newton.urdf",
            pos=(0, 0, 0.4),
        ),
    )
    ########################## build ##########################
    n_envs = 1
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

    robot.set_dofs_kp(np.full(12, 4.0), motor_dofs)
    robot.set_dofs_kv(np.full(12, 1), motor_dofs)
    default_dof_pos = np.array(
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
        ]
    )
    robot.control_dofs_position(default_dof_pos, motor_dofs)

    idx = 0
    reset = False
    domain_rand = DomainRandomizer(scene, robot, n_envs, joint_names)

    while True:
        scene.step()
        idx += 1

        # Push the base around every 1000 steps
        if idx % 1000 == 0:
            if reset:
                domain_rand.reset()
            else:
                domain_rand.randomize()

            reset = not reset




if __name__ == "__main__":
    main()
