import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
import logging
from typing import Any, Dict, Tuple, List


class TrajDrawer:
    def __init__(
            self, 
            nodes_filepath: str,
            background_image_path: str,
            bounding_box: tuple,  # Should be (x_min, y_min, x_max, y_max)
            logger: logging.Logger = None,
            node_id_col: str = 'node_index', 
            node_x_col: str = 'pos_x',
            node_y_col: str = 'pos_y'
            ):
        """
        Initializes the TrajDrawer.

        Args:
            nodes_filepath: Path to the CSV file containing node coordinates.
            background_image_path: Path to the TIF background image.
            bounding_box: Tuple defining the image extent (x_min, y_min, x_max, y_max).
            logger: Optional logger instance. If None, a default logger is created.
            node_id_col: Name of the column containing node IDs in the nodes CSV.
            node_x_col: Name of the column containing x-coordinates in the nodes CSV.
            node_y_col: Name of the column containing y-coordinates in the nodes CSV.
        """
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            # Avoid adding handler if logger already has handlers (e.g., in Jupyter)
            if not self.logger.hasHandlers():
                self.logger.addHandler(logging.StreamHandler())
        else: 
            self.logger = logger

        self.node_id_col = node_id_col
        self.node_x_col = node_x_col
        self.node_y_col = node_y_col

        self.nodes_dict: Dict[Any, Tuple[float, float]] = self._load_nodes(nodes_filepath)
        self.background_image: np.ndarray = self._load_background_image(background_image_path)
        self.bounding_box: tuple = self._set_bounding_box(bounding_box)

        self.fig = None
        self.ax = None
        self.scatter = None
        self.time_text = None

    def _load_nodes(self, nodes_filepath: str) -> Dict[Any, Tuple[float, float]]:
        """Loads node coordinates from CSV into a dictionary for fast lookup."""
        try:
            nodes_df = pd.read_csv(nodes_filepath)
            # Validate required columns
            if not {self.node_id_col, self.node_x_col, self.node_y_col}.issubset(nodes_df.columns):
                 raise ValueError(f"Nodes CSV must contain columns: '{self.node_id_col}', '{self.node_x_col}', '{self.node_y_col}'")

            nodes_dict = nodes_df.set_index(self.node_id_col)[[self.node_x_col, self.node_y_col]].apply(tuple, axis=1).to_dict()
            self.logger.info(f"Read {len(nodes_dict)} nodes from {nodes_filepath}.")
            return nodes_dict
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading nodes: {e}")
            raise
    
    def _load_background_image(self, background_image_path: str) -> np.ndarray:
        try:
            background_image: np.ndarray = plt.imread(background_image_path)
            self.logger.info(f"Loaded background image from {background_image_path}")
            return background_image
        except Exception as e:
            self.logger.error(f"Error loading background image {background_image_path}: {e}")
            raise
        
    def _set_bounding_box(self, bounding_box: tuple) -> tuple:
        """Validates and sets the bounding box."""
        if len(bounding_box) == 4 and all(isinstance(n, (int, float)) for n in bounding_box):
            self.logger.info(f"Bounding box set to: {bounding_box}")
            return bounding_box
        else:
            raise ValueError("Bounding box should be a tuple of four numbers: (x_min, y_min, x_max, y_max).")
        
    def _setup_plot(
            self, 
            title: str = "Fleet Trajectory",
            figsize: tuple = (10, 10),
            scatter_size: int = 50,
            edgecolors: str = 'k', linewidths: float = 0.5,
            text_position: tuple = (0.02, 0.95), text_ha: str = 'left', text_va: str = 'top', 
            text_fontsize: int = 10, text_color: str = 'black', text_backgroundcolor: str = 'white', text_zorder: int = 10,
            ):
        """Sets up the Matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        if self.background_image is not None:
                # aspect='auto' allows the image to stretch to the bounding box
                # extent defines the coordinate system for the image corners
            self.ax.imshow(self.background_image, extent=self.bounding_box, aspect='auto', origin='upper')
        else:
            self.logger.warning("No background image loaded. Plotting on empty axes.")

        self.ax.set_xlim(694970, 697782)
        self.ax.set_ylim(5328907, 5331570)
        self.ax.set_title(title)
        # Optional: Hide axis ticks/labels if the background provides context
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Initialize placeholder scatter plot for vehicles
        self.scatter = self.ax.scatter([], [], s=scatter_size, facecolors=[], edgecolors='none')

        # Initialize text for timestamp display
        self.time_text = self.ax.text(
            text_position[0], text_position[1], '', transform=self.ax.transAxes, ha=text_ha, va=text_va,
            fontsize=text_fontsize, color=text_color, backgroundcolor=text_backgroundcolor, zorder=text_zorder
            )

        plt.tight_layout() # Adjust layout to prevent labels overlapping

    def _update_frame(
        self, 
        frame_idx: int,
        fleet_trajectory: Dict[str, Dict[Any, Tuple[Any, float]]],
        timestamps: List[Any],
        speed: int,
        occupancy_color_map: Dict[int, str]
        ):
        """
        Updates the plot for a single animation frame.

        Args:
            timestamp: The current timestamp.
            fleet_trajectory: The main trajectory data structure.

        Returns:
            A tuple of mutable plot elements (scatter plot, time text).
        """
        time_step_index = frame_idx * speed
        timestamp = timestamps[time_step_index]

        if time_step_index >= len(timestamps):
            self.logger.warning(f"Timestamp index {time_step_index} is out of range for timestamps.")
            return self.scatter, self.time_text

        x_coords, y_coords, colors = [], [], []

        for vehicle_id, trajectory in fleet_trajectory.items():
            if timestamp in trajectory:
                node_id, occupancy = trajectory[timestamp]
                if node_id in self.nodes_dict:
                    x, y = self.nodes_dict[node_id]
                    x_coords.append(x)
                    y_coords.append(y)
                    color = occupancy_color_map[int(occupancy)]
                    colors.append(color)
                else:
                    self.logger.warning(f"Node ID '{node_id}' for vehicle '{vehicle_id}' at time {timestamp} not found in nodes data. Skipping vehicle for this frame.")

        # Efficiently update scatter plot data
        if x_coords: # Check if there are any vehicles to plot in this frame
            self.scatter.set_offsets(np.c_[x_coords, y_coords])

            self.scatter.set_facecolors(colors)
        else: # Clear scatter if no vehicles are present
            self.scatter.set_offsets(np.empty((0, 2)))
            self.scatter.set_facecolors(np.empty((0, 4)))

        # Update the time display
        self.time_text.set_text(f'Time: {timestamp}')

        # Return the updated plot elements (important for blit=True, harmless for blit=False)
        return self.scatter, self.time_text
    
    def draw_trajectory_animation(
        self,
        fleet_trajectory: Dict[str, Dict[Any, Tuple[Any, float]]],
        occupancy_color_map: Dict[int, str] = None,
        speed: int = 1,
        start_time: int = 0,
        end_time: int = None,
        time_step: int = 1,
        output_filepath: str = "./fleet_animation.gif",
        interval_ms: int = 100,
        ):
        if not fleet_trajectory:
            self.logger.error("Fleet trajectory data is empty. Cannot create animation.")
            return
        if not isinstance(speed, int) or speed <= 0:
             self.logger.error(f"Speed must be a positive integer, got {speed}.")
             raise ValueError("Speed must be a positive integer.")

        # Add a color bar for occupancy
        if occupancy_color_map is None:
            occupancy_color_map = {
                0: 'black',    # Example color for occupancy 0
                1: 'blue',    # Example color for occupancy 1
                2: 'lime',    # Example color for occupancy 2
                3: 'orange',  # Example color for occupancy 3
                4: 'red'      # Example color for occupancy 4
            }
            self.logger.info(f"Using default occupancy color map: {occupancy_color_map}")
        else:
            self.logger.info(f"Using provided occupancy color map: {occupancy_color_map}")

        self.logger.info(f"Preparing animation with speed={speed}, interval={interval_ms}ms...")

        # --- 1. Data Preparation ---
        # Get timestamps and sort them
        timestamps: np.ndarray = np.arange(start_time, end_time+1, time_step)
        
        num_time_steps = len(timestamps)
        num_frames = (num_time_steps + speed - 1) // speed # Ceiling division

        # --- 2. Setup Plot ---
        self._setup_plot()

        # --- 3. Create Animation ---
        self.logger.info(f"Creating animation with {num_frames} frames...")
        # Use functools.partial if needed to pass extra static args to _update_frame,
        # but fargs is standard for FuncAnimation.
        ani = animation.FuncAnimation(
            fig=self.fig,
            func=self._update_frame,
            frames=num_frames,
            fargs=(fleet_trajectory, timestamps, speed, occupancy_color_map), # Pass additional arguments
            interval=interval_ms,  # Delay between frames in milliseconds
            blit=False,           # blit=False is often more robust, especially with text/color changes
            repeat=False          # Do not repeat the animation
        )

        # --- 4. Save Animation ---
        try:
            self.logger.info(f"Saving animation to {output_filepath}...")
            # Determine writer based on extension (basic implementation)
            writer = None
            if output_filepath.lower().endswith('.gif'):
                writer = animation.PillowWriter(fps=1000 / interval_ms) # Pillow is common for GIFs
            else:
                raise NotImplementedError("Unsupported file format for animation.")

            ani.save(output_filepath, writer=writer)

            self.logger.info(f"Animation saved successfully to {output_filepath}")

        except Exception as e:
            self.logger.error(f"Error saving animation to {output_filepath}: {e}")

        # --- 6. Cleanup ---
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.scatter = None
        self.time_text = None
