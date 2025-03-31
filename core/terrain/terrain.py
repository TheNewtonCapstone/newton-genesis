import numpy as np
from genesis.options.morphs import Terrain as TerrainMorph
from .generators import (
    pyramid_sloped_terrain,
    pyramid_stairs_terrain,
    random_uniform_terrain,
    discrete_obstacles_terrain,
    stepping_stones_terrain,
)
from core.config.terrain import TerrainConfig
from .sub_terrain import SubTerrain
# Note: Make sure that any utilities such as gs.raise_exception and mu.is_approx_multiple
# are imported or available in your working environment.

class Terrain:
    """
    The terrain can be constructed as a grid of sub-terrains or from a height field.
    It supports curriculum learning by varying terrain difficulty across sub-terrains.
    """

    def __init__(self, terrain_cfg: TerrainConfig, **data):

        # ---------------------------------------------------------------------------
        # Initialization: Basic parameters from configuration
        # ---------------------------------------------------------------------------

        super().__init__(**data)
        # Set subterrain size from configuration. Here we assume:
        #   index 0 -> width, index 1 -> length.
        self.subterrain_size = (terrain_cfg.terrain_size, terrain_cfg.terrain_size)
        self.n_subterrains = (terrain_cfg.num_rows, terrain_cfg.num_cols)
        self.horizontal_scale = terrain_cfg.horizontal_scale
        self.vertical_scale = terrain_cfg.vertical_scale

        # Additional terrain properties
        self.border_size = terrain_cfg.border_size
        self.slope_threshold = terrain_cfg.slope_threshold  # Note: using config's spelling

        # Curriculum flag and subterrain type proportions from config
        self.curriculum: bool = terrain_cfg.curriculum
        self._sub_terrain_type_proportion = terrain_cfg.terrain_proportions
        self._sub_terrain_types = terrain_cfg.terrain_types

        # ---------------------------------------------------------------------------
        # Compute derived quantities for subterrain vertex counts
        # ---------------------------------------------------------------------------

        self._num_rows: int = terrain_cfg.num_rows
        self._num_cols: int = terrain_cfg.num_cols

        # Number of vertices per subterrain (width and length)
        self._sub_terrain_num_width_vertex: int = int(self.subterrain_size[0] // self.horizontal_scale)
        self._sub_terrain_num_length_vertex: int = int(self.subterrain_size[1] // self.horizontal_scale)
        self._sub_terrain_num_border_vertex: int = int(self.border_size // self.horizontal_scale)

        # Save physical dimensions for later use in origin computation
        self._sub_terrain_width: float = self.subterrain_size[0]
        self._sub_terrain_length: float = self.subterrain_size[1]

        # Initialize the height field and update dependent properties
        self._height_field = None
        self._update_rows_cols_dependents()

    def build(self):
        """
        Build the terrain by constructing subterrains based on the curriculum.
        """
        self._construct_curriculum()

    # ---------------------------------------------------------------------------
    # Public API: Get Subterrain Origins
    # ---------------------------------------------------------------------------

    def get_subterrain_origins(self):
        """
        Returns the origins of each subterrain as an array of shape (-1, 3).
        """
        return self._sub_terrain_origins.reshape(-1, 3)

    def get_height_field(self):
        """
        Returns the height field of the terrain as a numpy array.
        """
        return self._height_field

    # ---------------------------------------------------------------------------
    # Curriculum Construction: Generate and add terrains based on difficulty
    # ---------------------------------------------------------------------------

    def _construct_curriculum(self):
        for j in range(self._num_cols):
            for i in range(self._num_rows):
                # Compute difficulty as a fraction of row index over total rows.
                difficulty = i / self._num_rows
                # Determine a choice value along the column index with a small offset.
                choice = self._sub_terrain_types[j]

                # Generate a subterrain based on curriculum parameters.
                terrain = self._generate_curriculum_terrains(choice, difficulty)
                self._add_sub_terrain(terrain, i, j)

    def _generate_curriculum_terrains(self, choice: str, difficulty: float):
        """
        Generate a single subterrain based on curriculum choice and difficulty.
        """
        # Create a new subterrain with preset vertex counts and scales.
        terrain = SubTerrain(
            width=self._sub_terrain_num_width_vertex,
            length=self._sub_terrain_num_width_vertex,  # Assuming square grid for subterrain vertices
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale,
        )

        # Compute parameters based on difficulty.
        slope = int(difficulty * 0.4)
        step_height = 0.05 + 0.1 * difficulty
        # discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty  # Not used in current generators
        pit_depth = 1.0 * difficulty   # Not used in current generators

        # Select terrain generator based on the provided choice and configured proportions.
        if choice == "smooth_slope":
            # For smooth/pyramid sloped terrain, adjust slope sign based on choice.
            slope *= -1
            pyramid_sloped_terrain(terrain, slope=slope, platform_size=2.0)
        elif choice == "rough_slope":
            pyramid_sloped_terrain(terrain, slope=slope, platform_size=2.0)
            random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=0.05,
                downsampled_scale=0.2,
            )
        elif choice in ["stairs_up", "stairs_down"]:
            # For stairs terrain, adjust step height sign based on choice.
            step_height *= -1 if choice == "stairs_down" else 1
            pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=2.0)
        elif choice == "discrete":
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            discrete_obstacles_terrain(
                terrain,
                step_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=2.0,
            )
        elif choice == "stepping_stones":
            stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=2.0,
            )
        else:
            raise ValueError(f"Invalid terrain choice: {choice}")

        return terrain

    # ---------------------------------------------------------------------------
    # Update Dependent Quantities Based on Rows and Columns
    # ---------------------------------------------------------------------------

    def _update_rows_cols_dependents(self) -> None:
        """
        Update quantities that depend on the number of subterrains.
        This includes total rows, columns, and the overall height field array.
        """
        self.num_sub_terrains = self._num_rows * self._num_cols
        # Initialize the subterrain origins array (each origin is [x, y, z])
        self._sub_terrain_origins = np.zeros((self._num_cols, self._num_rows, 3))

        # Total number of rows and columns in the full terrain grid (including borders)
        self._total_num_rows = int(self._num_rows * self._sub_terrain_num_length_vertex) + 2 * self._sub_terrain_num_border_vertex
        self._total_num_cols = int(self._num_cols * self._sub_terrain_num_width_vertex) + 2 * self._sub_terrain_num_border_vertex

        # Create the height field array with default zero heights.
        self._height_field: np.ndarray = np.zeros(
            (self._total_num_cols, self._total_num_rows),
            dtype=np.int16,
        )

    # ---------------------------------------------------------------------------
    # Add a Subterrain to the Global Height Field
    # ---------------------------------------------------------------------------

    def _add_sub_terrain(self, terrain: SubTerrain, row: int, col: int):
        """
        Inserts a generated subterrain into the global height field at the specified row and column.
        """
        i = col  # Column index in the grid
        j = row  # Row index in the grid

        # Determine starting and ending indices for the subterrain within the global height field
        start_x = self._sub_terrain_num_border_vertex + i * self._sub_terrain_num_length_vertex
        end_x = self._sub_terrain_num_border_vertex + (i + 1) * self._sub_terrain_num_length_vertex
        start_y = self._sub_terrain_num_border_vertex + j * self._sub_terrain_num_width_vertex
        end_y = self._sub_terrain_num_border_vertex + (j + 1) * self._sub_terrain_num_width_vertex

        # Update the height field with the subterrain's height data
        self._height_field[start_x:end_x, start_y:end_y] = terrain.height_field

        # Compute the environment origin for this subterrain
        env_origin_x = (i + 0.5) * self._sub_terrain_length
        env_origin_y = (j + 0.5) * self._sub_terrain_width

        # Determine a local region for estimating the z-origin (height) of the subterrain
        x1 = int((self._sub_terrain_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self._sub_terrain_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self._sub_terrain_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self._sub_terrain_width / 2.0 + 1) / terrain.horizontal_scale)

        env_origin_z = np.max(terrain.height_field[x1:x2, y1:y2]) * terrain.vertical_scale

        # Save the computed origin for later use (e.g., visualization or collision adjustments)
        self._sub_terrain_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
