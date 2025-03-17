import torch
import random
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.materials.rigid import Rigid

class DomainRandomizer:
    def __init__(self, scene, robot, num_envs, joint_names):
        self.scene = scene
        self.robot = robot
        self.joint_names = joint_names
        self.num_envs = num_envs

        self.motor_dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in joint_names]
        self.base_dofs_idx = self.robot.base_joint.dof_idx
        self.base_link_idx = [1]
        self.step_idx = 0

        self.base_mass_shift = torch.zeros(self.num_envs, 1)
        self.base_com_shift = torch.zeros(self.num_envs, 1, 3)

        self.rigid_solver = None
        for solver in scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

    def randomize(self, envs_idx=None):

        #### Randomizing the mass and COM
        mass_shift = self.rand_mass_shift(envs_idx)
        com_shift = self.rand_com_shift(envs_idx)

        # Shifts the mass of the base link
        self.robot.set_mass_shift(
            mass_shift= mass_shift,
            link_indices=[0],
            envs_idx=envs_idx,
        )

        # Shifts the COM of the base link
        self.robot.set_COM_shift(
            com_shift= com_shift,
            link_indices=[0],
            envs_idx=envs_idx,
        )

        # Randomize friction
        self.rand_terrain_friction()


    def push_xy(self):
        # Generate random forces in x and y, keeping z = 0
        random_force = torch.randn(self.num_envs, 1, 3) * 75  # Scaling the randomness
        random_force[:, :, 2] = 0  # No force applied in the z-axis

        print("Random force applied:", random_force)

        self.rigid_solver.apply_links_external_force(
            force=random_force.numpy(),
            links_idx=self.base_link_idx,
        )

    def reset(self):
        # Resetting shift
        self.reset_mass_shift()
        self.reset_com_shift()
        print("Resetting the robot's shift")

    def rand_com_shift(self, envs_idx = None):
        if envs_idx is None:
            com_shift = 0.05 * torch.randn_like(self.base_com_shift)
        else:
            com_shift = 0.05 * torch.randn(envs_idx.shape[0], 1, 3)
        com_shift[:, :, 2] = 0.0  # No changes in z-axis
        return com_shift

    def rand_mass_shift(self, envs_idx = None):
        if envs_idx is None:
            mass_shift = 0.1 * torch.randn_like(self.base_mass_shift)
        else:
            mass_shift = 0.1 * torch.randn(envs_idx.shape[0], 1)
        return mass_shift

    def rand_terrain_friction(self):
        """
        Randomizes the friction of the terrain's rigid bodies.

        Args:
            scene (gs.Scene): The physics simulation scene containing terrain entities.
        """
        min_friction = 0.05  # Minimum allowed friction
        max_friction = 1.0  # Maximum allowed friction

        for entity in self.scene.entities:
            if hasattr(entity, "material") and isinstance(entity.material, Rigid):
                # Generate a random friction value within the allowed range
                new_friction = random.uniform(min_friction, max_friction)

                # Update the material's friction
                entity.set_friction(new_friction)  # Directly modifying the private variable

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