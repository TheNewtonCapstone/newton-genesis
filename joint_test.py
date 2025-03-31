import argparse
import math

import genesis as gs
import numpy as np

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

    links_to_keep = [
        "FL_UPPER_LEG",
        "FR_UPPER_LEG",
        "HL_UPPER_LEG",
        "HR_UPPER_LEG",
        "FL_LOWER_LEG",
        "FR_LOWER_LEG",
        "HL_LOWER_LEG",
        "HR_LOWER_LEG", ]

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/newton/newton.urdf",
            pos=(0, 0, 0.4),
            quat=(0, 0, 0, 1),
            fixed=True,
            merge_fixed_links=True,
            links_to_keep=links_to_keep,
        ),
    )
    ########################## Build ##########################
    n_envs = 1
    scene.build(n_envs=n_envs)

    ########################## Controlling Joints ##########################

    joint_names = [
        "FR_KFE",
        "FL_KFE",
        "HR_KFE",
        "HL_KFE",
        "FR_HFE",
        "FL_HFE",
        "HR_HFE",
        "HL_HFE",
        "FR_HAA",
        "FL_HAA",
        "HR_HAA",
        "HL_HAA",
    ]
    motor_dofs = [robot.get_joint(name).dof_idx_local for name in joint_names]
    num_motors = len(motor_dofs)

    robot.set_dofs_kp(np.full(num_motors, 4.0), motor_dofs)
    robot.set_dofs_kv(np.full(num_motors, 1), motor_dofs)
    default_dof_pos = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    idx = 0
    initial_position = - math.pi
    curr_motor = 0
    step_pos = math.pi / 10

    while True:
        scene.step()
        idx += 1

        if idx % 100 == 0:
            default_dof_pos[curr_motor] = initial_position
            robot.control_dofs_position(default_dof_pos, motor_dofs)
            print("Joint: ", joint_names[curr_motor], "Position: ", initial_position)
            initial_position += step_pos

            if initial_position > 6.3:
                initial_position = -math.pi
                curr_motor += 1


if __name__ == "__main__":
    main()
