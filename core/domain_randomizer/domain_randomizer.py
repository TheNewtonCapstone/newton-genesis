import torch

class DomainRandomizer:
    def __init__(self, scene, robot, num_envs, joint_names):
        self.scene = scene
        self.robot = robot
        self.joint_names = joint_names
        self.num_envs = num_envs

        self.motor_dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in joint_names]
        self.base_dofs_idx = self.robot.base_joint.dof_idx

        self.base_mass_shift = torch.zeros(self.num_envs, 1)
        self.base_com_shift = torch.zeros(self.num_envs, 1, 3)

    def randomize(self):

        #### Randomizing the mass and COM
        mass_shift = self.rand_mass_shift()
        com_shift = self.rand_com_shift()

        # Shifts the mass of the base link
        self.robot.set_mass_shift(
            mass_shift= mass_shift,
            link_indices=[0],
        )

        # Shifts the COM of the base link
        self.robot.set_COM_shift(
            com_shift= com_shift,
            link_indices=[0]
        )

        #### Push robot around
        base_vel = 2 * torch.rand(self.num_envs, len(self.base_dofs_idx)) - 1
        self.robot.set_dofs_velocity(base_vel, self.base_dofs_idx)

    def reset(self):
        # Resetting shift
        self.reset_mass_shift()
        self.reset_com_shift()

    def rand_com_shift(self):
        com_shift = 0.5 * (2 * torch.randn_like(self.base_com_shift) - 1)
        com_shift[:, :, 2] = 0.0  # No changes in z-axis
        return com_shift

    def rand_mass_shift(self):
        return 0.5 * (2 * torch.randn_like(self.base_mass_shift) - 1)

    def reset_mass_shift(self):
        self.robot.set_mass_shift(
            mass_shift= self.base_mass_shift,
            link_indices=[0],
        )

    def reset_com_shift(self):
        self.robot.set_COM_shift(
            com_shift= self.base_com_shift,
            link_indices=[0],
        )