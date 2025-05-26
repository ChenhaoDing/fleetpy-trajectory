import pandas as pd
import numpy as np
import os
import logging
from shapely.geometry import Point, LineString
from shapely.ops import substring

from src.NetworkRouter import NetworkRouter


class PTTrajBuilder:
    def __init__(self, network_router: NetworkRouter, logger: logging.Logger = None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            # Avoid adding handler if logger already has handlers (e.g., in Jupyter)
            if not self.logger.hasHandlers():
                self.logger.addHandler(logging.StreamHandler())
        else: 
            self.logger = logger

        self.network_router: NetworkRouter = network_router

    def get_line_points_with_distances(self, route: LineString) -> tuple:
        pass
