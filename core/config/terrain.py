class TerrainConfig:
    # ---------------------------------------------------------------------------
    # Mesh and Scaling Configuration
    # ---------------------------------------------------------------------------
    mesh_type = 'trimesh'  # Options: 'none', 'plane', 'heightfield', or 'trimesh'
    horizontal_scale = 0.1  # Horizontal scale [m]
    vertical_scale = 0.005  # Vertical scale [m]
    border_size = 0.1      # Border size [m]

    # ---------------------------------------------------------------------------
    # Curriculum Settings
    # ---------------------------------------------------------------------------
    curriculum = True     # Enable curriculum learning
    # measure_heights = True  # Enable height measurements (for rough terrain only)

    # # Points used for measuring terrain heights
    # measured_points_x = [
    #     -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    #      0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7, 0.8
    # ]  # Represents a 1m x 1.6m rectangle (excluding the center line)
    # measured_points_y = [
    #     -0.5, -0.4, -0.3, -0.2, -0.1,
    #      0.0,  0.1,  0.2,  0.3,  0.4,  0.5
    # ]

    # ---------------------------------------------------------------------------
    # Terrain Type Selection and Configuration
    # ---------------------------------------------------------------------------
    selected = False      # If True, select a unique terrain type and pass all arguments
    terrain_kwargs = None  # Dictionary of additional arguments for the selected terrain
    max_init_terrain_level = 5  # Starting curriculum state (terrain difficulty level)

    # Terrain dimensions and layout configuration
    terrain_size = 8.0  # Terrain length [m]
    num_rows = 3       # Number of terrain rows (levels)
    num_cols = 3       # Number of terrain columns (types)

    # Terrain type proportions:
    # [smooth slope, rough slope, stairs up, stairs down, discrete, stepping_stones]
    terrain_proportions = [0.1, 0.1, 0.35, 0.45, 0.7]
    terrain_types = ['smooth_slope', 'stairs_up', 'stairs_down', 'discrete', 'stepping_stones']

    # ---------------------------------------------------------------------------
    # Trimesh Specific Settings
    # ---------------------------------------------------------------------------
    slope_threshold = 0.75  # For 'trimesh' type: slopes above this threshold will be corrected to vertical surfaces

    # ---------------------------------------------------------------------------
    # Physical Properties
    # ---------------------------------------------------------------------------
    static_friction = 1.0    # Static friction coefficient
    dynamic_friction = 1.0   # Dynamic friction coefficient
    restitution = 0.0        # Coefficient of restitution (bounciness)
