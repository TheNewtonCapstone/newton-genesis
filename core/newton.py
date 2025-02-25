import argparse
import os
import time

import numpy as np
import torch
import genesis as gs

from config.terrain import TerrainConfig
from terrain.terrain import Terrain

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0, -5.0, 10.0),
            camera_lookat=(5.0, 5.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
        ),
        vis_options=gs.options.VisOptions(),
    )

    terrain = Terrain(
        terrain_cfg=TerrainConfig(),
    )
    terrain.build()

    sub_terrain_origins = terrain.get_subterrain_origins()
    sub_terrain_origins[:, 2] += 0.3

    ########################## entities ##########################
    gs_terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=terrain.horizontal_scale,
            vertical_scale=terrain.vertical_scale,
            height_field=terrain.get_height_field(),
        ),
    )
    print(os.getcwd())
    newton = scene.add_entity(
        gs.morphs.URDF(
            file="assets/newton/newton.urdf",
            pos=sub_terrain_origins[0],
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    )

    ########################## build ##########################
    scene.build()

    # height_field = terrain.geoms[0].metadata["height_field"]
    # rows = horizontal_scale * torch.range(0, height_field.shape[0] - 1, 1).unsqueeze(1).repeat(
    #     1, height_field.shape[1]
    # ).unsqueeze(-1)
    # cols = horizontal_scale * torch.range(0, height_field.shape[1] - 1, 1).unsqueeze(0).repeat(
    #     height_field.shape[0], 1
    # ).unsqueeze(-1)
    # heights = vertical_scale * torch.tensor(height_field).unsqueeze(-1)
    #
    # poss = torch.cat([rows, cols, heights], dim=-1).reshape(-1, 3)
    # scene.draw_debug_spheres(poss=poss, radius=0.05, color=(0, 0, 1, 0.7))
    for _ in range(1000):
        time.sleep(0.5)
        scene.step()


if __name__ == "__main__":
    main()