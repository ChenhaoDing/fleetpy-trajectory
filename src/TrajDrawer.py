import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import logging
from typing import Any, Dict, Tuple, List
import rasterio
from rasterio.plot import show as rio_show


class TrajDrawer:
    def __init__(
            self, 
            background_image_path: str,
            bounding_box: tuple,  # Should be (x_min, y_min, x_max, y_max)
            logger: logging.Logger = None,
            pt_color: str = '#00796b',         
            pt_reversed_color: str = '#00796b',
            point_size: int = 30,
            occupancy_to_color_map: Dict[Any, Dict[str, Any]] = None,
            default_vehicle_color: str = 'grey',
            ):
        """
        Initializes the TrajDrawer.

        Args:
            background_image_path: Path to the TIF background image.
            bounding_box: Tuple defining the image extent (x_min, y_min, x_max, y_max).
            logger: Optional logger instance. If None, a default logger is created.
            pt_color: Color for PT trajectories.
            pt_reversed_color: Color for reversed PT trajectories.
            point_size: Size of the points in the scatter plot.
            occupancy_to_color_map: Dictionary mapping occupancy values to colors and labels.
            default_vehicle_color: Color to use if occupancy value is not found in the map.
        """
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            # Avoid adding handler if logger already has handlers (e.g., in Jupyter)
            if not self.logger.hasHandlers():
                self.logger.addHandler(logging.StreamHandler())
        else: 
            self.logger = logger

        self.background_image_path: str = background_image_path
        self.bounding_box: tuple = self._set_bounding_box(bounding_box)

        self.pt_color = pt_color
        self.pt_reversed_color = pt_reversed_color
        self.point_size = point_size

        self.fig = None
        self.ax_map = None
        self.ax_info = None
        self.sc_vehicle = None # Scatter for vehicles
        self.sc_pt = None      # Scatter for pt
        self.sc_pt_reversed = None # Scatter for pt_reversed
        self.time_text = None

        if occupancy_to_color_map is None:
            self.occupancy_to_color_map = {
                0: {'color': '#BDBDBD', 'label': 'Vehicle (Idle)'},
                1: {'color': "#3FC94D", 'label': 'Vehicle (Occ: 1)'},
                2: {'color': "#F7FF5FF2", 'label': 'Vehicle (Occ: 2)'},
                3: {'color': "#F09B1A", 'label': 'Vehicle (Occ: 3)'},
                4: {'color': "#F45525", 'label': 'Vehicle (Occ: 4)'},
            }
        else:
            self.occupancy_to_color_map = occupancy_to_color_map
        self.default_vehicle_color = default_vehicle_color
        
    def _set_bounding_box(self, bounding_box: tuple) -> tuple:
        """Validates and sets the bounding box."""
        if len(bounding_box) == 4 and all(isinstance(n, (int, float)) for n in bounding_box):
            self.logger.info(f"Bounding box set to: {bounding_box}")
            return bounding_box
        else:
            raise ValueError("Bounding box should be a tuple of four numbers: (x_min, y_min, x_max, y_max).")
        
    def _prepare_data_for_animation(self, trajectories_data: Dict[Any, Dict[str, List]]):
        """
        Prepares data by extracting all timestamps and determining occupancy range.
        """
        if not trajectories_data:
            self.logger.warning("Trajectories data is empty.")
            return []
        # We still need sorted timestamps
        return sorted(trajectories_data.keys())
        
    def _setup_plot(
        self,
        title: str = "Trajectory Animation",
        figsize: tuple = (7, 9), # Increased size for legend and colorbar
        text_position: tuple = (0.02, 0.95), text_ha: str = 'left', text_va: str = 'top',
        text_fontsize: int = 11, text_color: str = 'black', background_color: str = 'white', text_zorder: int = 10,
    ):
        """Sets up the Matplotlib figure and axes."""
        self.fig, self.ax_map = plt.subplots(figsize=figsize)

        self.ax_map.axis('off')

        if self.background_image_path is not None:
            with rasterio.open(self.background_image_path) as src:
                rio_show(src, ax=self.ax_map, transform=src.transform) 
                self.logger.info(f"Background image plotted using rasterio.plot.show. CRS: {src.crs}")
                self.logger.info(f"Rasterio set ax limits to: X={self.ax_map.get_xlim()}, Y={self.ax_map.get_ylim()}")
                self.ax_map.set_aspect('equal')
        else:
            self.logger.warning("No background image. Plotting on empty axes with bounding_box limits.")
            self.ax_map.set_xlim(self.bounding_box[0], self.bounding_box[2])
            self.ax_map.set_ylim(self.bounding_box[1], self.bounding_box[3])
            self.ax_map.set_aspect('auto')

            self.ax_map.set_title(title)

        # Initialize scatter plots for each type of point
        self.sc_vehicle = self.ax_map.scatter([], [], s=self.point_size, label='_nolegend_',
                                          alpha=0.8, linewidths=0.5, zorder=10,
                                          edgecolors='k')
        self.sc_pt = self.ax_map.scatter([], [], s=self.point_size-15, color=self.pt_color, marker='s', label='Subway Vehicle', alpha=0.8, zorder=9)
        self.sc_pt_reversed = self.ax_map.scatter([], [], s=self.point_size-15, color=self.pt_reversed_color, marker='s', label='Subway Vehicle (reversed)', alpha=0.8, zorder=9)

        self.time_text = self.ax_map.text(
            text_position[0], text_position[1], '', transform=self.ax_map.transAxes, ha=text_ha, va=text_va,
            fontsize=text_fontsize, color=text_color, backgroundcolor=background_color, zorder=text_zorder
        )

        # Create legend for vehicle occupancies
        legend_handles = []
        legend_labels = []

        sorted_occupancy_keys = sorted(self.occupancy_to_color_map.keys())

        for occ_key in sorted_occupancy_keys:
            info = self.occupancy_to_color_map[occ_key]
            color_str = info.get('color', self.default_vehicle_color)
            label_str = info.get('label', f'Vehicle (Occ: {occ_key})')

            vehicle_proxy = plt.Line2D(
                [0], [0],
                linestyle='None',
                marker='o',
                markersize=8,
                markerfacecolor=color_str,
                markeredgecolor='k',
                markeredgewidth=0.3,
            )
            legend_handles.append(vehicle_proxy)
            legend_labels.append(label_str)

        if self.sc_pt and self.sc_pt.get_label() and not self.sc_pt.get_label().startswith('_'):
            pt_proxy = plt.Line2D(
                [0], [0],
                linestyle='None',
                marker='s',
                markersize=8,
                markerfacecolor=self.pt_color,
                markeredgecolor='k',
                markeredgewidth=0.3,
            )
            legend_handles.append(pt_proxy)
            legend_labels.append(self.sc_pt.get_label())

        # if self.sc_pt_reversed and self.sc_pt_reversed.get_label() and not self.sc_pt_reversed.get_label().startswith('_'):
        #     pt_reversed_proxy = plt.Line2D(
        #         [0], [0],
        #         linestyle='None',
        #         marker='s',
        #         markersize=8,
        #         markerfacecolor=self.pt_reversed_color,
        #         markeredgewidth=0,
        #     )
        #     legend_handles.append(pt_reversed_proxy)
        #     legend_labels.append(self.sc_pt_reversed.get_label())

        pt_line_proxy = plt.Line2D(
            [0], [0],
            linestyle='-', 
            linewidth=2,
            color=self.pt_color,  
        )
        legend_handles.append(pt_line_proxy)
        legend_labels.append('Subway Line')

        pt_station_proxy = plt.Line2D(
            [0], [0],
            linestyle='None',
            marker='^',
            markersize=8,
            markerfacecolor=self.pt_color,
            markeredgecolor='k',
            markeredgewidth=0.3, 
        )
        legend_handles.append(pt_station_proxy)
        legend_labels.append('Mobility Hub')

        if legend_handles:
            legend_object = self.ax_map.legend(legend_handles, legend_labels, loc='lower left', 
                                fontsize=11, frameon=True, facecolor=background_color)
            legend_object.set_zorder(100)
        else:
            self.logger.warning("No items to show in custom legend.")

        # plt.tight_layout()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    def _update_frame(
        self,
        frame_idx: int, # This is the direct index for timestamps_for_animation
        trajectories_data: Dict[Any, Dict[str, List]],
        timestamps_for_animation: List[Any]
    ):
        """
        Updates the plot for a single animation frame using the new data structure.
        """
        current_time = timestamps_for_animation[frame_idx]
        data_at_time = trajectories_data.get(current_time, {})

        if not data_at_time:
            print(f"No data for time {current_time}. Skipping frame.")

        # Update Vehicle points
        vehicle_points = data_at_time.get('vehicle', [])
        if vehicle_points:
            coords, occupancies = zip(*vehicle_points)
            x_veh, y_veh = zip(*coords) if coords else ([], [])
            self.sc_vehicle.set_offsets(np.c_[x_veh, y_veh])
            
            vehicle_colors = []
            for occ_val in occupancies:
                key_to_check = int(occ_val)
                info_entry = self.occupancy_to_color_map.get(key_to_check)
                if info_entry:
                    color = info_entry.get('color', self.default_vehicle_color)
                else:
                    color = self.default_vehicle_color
                    self.logger.warning(
                        f"Time {current_time}: Occupancy value '{key_to_check}' "
                        f"not found in vehicle_occupancy_legend_info. Using default color '{self.default_vehicle_color}'."
                    )
                vehicle_colors.append(color)
            
            if vehicle_colors: # Should always be true if vehicle_points is true
                 self.sc_vehicle.set_facecolors(vehicle_colors)
            else: # Fallback if somehow vehicle_colors list ended up empty
                 self.sc_vehicle.set_facecolors([])
            self.sc_vehicle.set_alpha(0.8)
        else:
            self.sc_vehicle.set_offsets(np.empty((0, 2)))
            self.sc_vehicle.set_alpha(0.0) # Hide if no data

        # Update PT points
        pt_points = data_at_time.get('pt', [])
        if pt_points:
            x_pt, y_pt = zip(*pt_points) if pt_points else ([], [])
            self.sc_pt.set_offsets(np.c_[x_pt, y_pt])
            self.sc_pt.set_alpha(0.8)
        else:
            self.sc_pt.set_offsets(np.empty((0, 2)))
            self.sc_pt.set_alpha(0.0)

        # Update PT_reversed points
        pt_reversed_points = data_at_time.get('pt_reverse', [])
        if pt_reversed_points:
            x_ptr, y_ptr = zip(*pt_reversed_points) if pt_reversed_points else ([], [])
            self.sc_pt_reversed.set_offsets(np.c_[x_ptr, y_ptr])
            self.sc_pt_reversed.set_alpha(0.8)
        else:
            self.sc_pt_reversed.set_offsets(np.empty((0, 2)))
            self.sc_pt_reversed.set_alpha(0.0)

        self.time_text.set_text(f'Simulation Time: {current_time} s')

        return self.sc_vehicle, self.sc_pt, self.sc_pt_reversed, self.time_text

    def draw_trajectory_animation(
        self,
        trajectories_data: Dict[Any, Dict[str, List]], # New data structure
        # Removed: occupancy_color_map, start_time, end_time, time_step
        # speed parameter is now implicitly 1 frame per timestamp in trajectories_data
        output_filepath: str = "./trajectory_animation.gif",
        interval_ms: int = 100,
        repeat_animation: bool = False # Renamed from 'repeat' for clarity
    ):
        """
        Generates and saves the trajectory animation using the new data structure.
        """
        if not trajectories_data:
            self.logger.error("Trajectories data is empty. Cannot create animation.")
            return

        self.logger.info(f"Preparing animation with interval={interval_ms}ms...")

        # --- 1. Data Preparation & Normalization ---
        timestamps_for_animation = self._prepare_data_for_animation(trajectories_data)
        if not timestamps_for_animation:
            self.logger.error("No timestamps found in trajectories data. Cannot create animation.")
            return

        num_frames = len(timestamps_for_animation)

        # --- 2. Setup Plot ---
        self._setup_plot()

        # --- 3. Create Animation ---
        self.logger.info(f"Creating animation with {num_frames} frames...")
        ani = animation.FuncAnimation(
            fig=self.fig,
            func=self._update_frame,
            frames=num_frames,
            fargs=(trajectories_data, timestamps_for_animation),
            interval=interval_ms,
            blit=True,
            repeat=repeat_animation
        )

        # --- 4. Save Animation ---
        try:
            self.logger.info(f"Saving animation to {output_filepath}...")
            writer = None
            if output_filepath.lower().endswith('.gif'):
                # PillowWriter is generally good for GIFs. 'imagemagick' can also be used if installed.
                writer = animation.PillowWriter(fps=1000 / interval_ms)
            elif output_filepath.lower().endswith('.mp4'):
                # FFMpegWriter for MP4. Requires ffmpeg to be installed and on PATH.
                writer = animation.FFMpegWriter(fps=1000 / interval_ms)
            else:
                self.logger.error(f"Unsupported file format for animation: {output_filepath}. Please use .gif or .mp4.")
                # Fallback or raise error
                plt.close(self.fig) # Close the figure if not saving
                return

            ani.save(output_filepath, writer=writer)
            self.logger.info(f"Animation saved successfully to {output_filepath}")

        except Exception as e:
            self.logger.error(f"Error saving animation to {output_filepath}: {e}")
            self.logger.error("Ensure that the required writer (Pillow for GIF, FFMpeg for MP4) is installed.")
        finally:
            # --- 5. Cleanup ---
            # It's good practice to close the figure after saving the animation,
            # especially if generating many animations in a loop.
            if self.fig:
                plt.close(self.fig)
            self.fig = None
            self.ax_map = None
            self.ax_info = None
            self.sc_vehicle = None
            self.sc_pt = None
            self.sc_pt_reversed = None
            self.time_text = None
