import os
import pandas as pd
import logging


class OpStatsLoader:
    def __init__(self, op_stats_path: str, logger: logging.Logger = None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            # Avoid adding handler if logger already has handlers (e.g., in Jupyter)
            if not self.logger.hasHandlers():
                self.logger.addHandler(logging.StreamHandler())
        else: 
            self.logger = logger

        self.op_stats_path: str = op_stats_path
        self.op_stats: pd.DataFrame = self._load_op_stats(op_stats_path)
        self.fleet_stats: dict[int, pd.DataFrame] = self._sort_fleet_stats()

    def _load_op_stats(self, op_stats_path: str) -> pd.DataFrame:
        op_stats = pd.read_csv(op_stats_path)
        return op_stats
    
    def get_op_stats(self) -> pd.DataFrame:
        return self.op_stats
    
    def _sort_fleet_stats(self) -> dict[int, pd.DataFrame]:
        op_stats = self.op_stats.copy()
        fleet_stats = {}

        # drop unnecessary columns
        op_stats = op_stats[['operator_id', 'vehicle_id', 'vehicle_type', 'status', 'start_time', 'end_time', 'occupancy', 'route']]

        uni_veh_ids = op_stats['vehicle_id'].unique()
        for veh_id in uni_veh_ids:
            fleet_stats[int(veh_id)] = op_stats[op_stats['vehicle_id'] == veh_id]
        
        self.logger.info(f"Loaded records for {len(fleet_stats.keys())} vehicles.")
        return fleet_stats

    def get_vehicle_stats(self, veh_id: int) -> dict[int, pd.DataFrame]:
        return self.fleet_stats[veh_id]
    
    def get_fleet_stats(self) -> dict[int, pd.DataFrame]:
        return self.fleet_stats
