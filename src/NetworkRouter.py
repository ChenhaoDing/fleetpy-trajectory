import os
import pandas as pd
import networkx as nx
import geopandas as gpd
import logging
import typing as tp


class NetworkRouter:
    def __init__(
            self,
            network_path: str,
            node_id_col: str = 'node_index', 
            origin_col: str = 'from_node', 
            destination_col: str = 'to_node',
            distance_col: str = 'distance', 
            time_col: str = 'travel_time',
            logger: logging.Logger = None,
            ):
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            # Avoid adding handler if logger already has handlers (e.g., in Jupyter)
            if not self.logger.hasHandlers():
                self.logger.addHandler(logging.StreamHandler())
        else: 
            self.logger = logger

        self.nodes_df: pd.DataFrame = None
        self.network: nx.Graph = self.build_network_from_csvs(network_path, node_id_col, origin_col, destination_col, distance_col, time_col)

    def build_network_from_csvs(
            self,
            network_path: str,
            node_id_col: str = 'node_index', 
            origin_col: str = 'from_node', 
            destination_col: str = 'to_node',
            distance_col: str = 'distance', 
            time_col: str = 'travel_time',
            ) -> tp.Optional[nx.Graph]:
        """
        Build a networkx network from CSV files.

        Args:
            network_path (str): Path to the network folder.
            node_id_col (str): Name of the column in the node file that represents the unique ID of the node.
            origin_col (str): Name of the column in the edge file that represents the starting node.
            destination_col (str): Name of the column in the edge file that represents the ending node.
            distance_col (str): Name of the column in the edge file that represents the distance.
            time_col (str): Name of the column in the edge file that represents the travel time.

        Returns:
            nx.Graph or None: NetworkX graph (if successfully created).
        """
        try:
            nodes_filepath = os.path.join(network_path, 'nodes.csv')
            edges_filepath = os.path.join(network_path, 'edges.csv')

            self.logger.info(f"Reading node file: {nodes_filepath}")
            self.nodes_df = pd.read_csv(nodes_filepath)
            self.logger.info(f"Read {len(self.nodes_df)} nodes.")

            self.logger.info(f"Reading edge file: {edges_filepath}")
            edges_df = pd.read_csv(edges_filepath)
            self.logger.info(f"Read {len(edges_df)} edges.")

            required_node_cols = [node_id_col]

            required_edge_cols = [origin_col, destination_col, distance_col, time_col]

            if not all(col in self.nodes_df.columns for col in required_node_cols):
                missing_cols = [col for col in required_node_cols if col not in self.nodes_df.columns]
                self.logger.error(f"Error: Node file is missing required columns: {missing_cols}")
                return None


            if not all(col in edges_df.columns for col in required_edge_cols):
                missing_cols = [col for col in required_edge_cols if col not in edges_df.columns]
                self.logger.error(f"Error: Edge file is missing required columns: {missing_cols}")
                return None


            # create a networkx graph
            G = nx.DiGraph()
            self.logger.info("Building network...")

            # add nodes (including all attributes)
            for _, row in self.nodes_df.iterrows():
                node_data = row.to_dict() # get all node information
                node_id = node_data[node_id_col]
                G.add_node(node_id, **node_data)

            # add edges (including all attributes)
            # rename columns to match expected names
            edges_df_renamed = edges_df.rename(columns={
                distance_col: 'distance',
                time_col: 'travel_time'
            })

            # check if travel_time contains non-numeric or negative values
            if not pd.api.types.is_numeric_dtype(edges_df_renamed['travel_time']):
                self.logger.warning(f"Warning: Column '{time_col}' contains non-numeric data, trying to convert to numeric...")
                edges_df_renamed['travel_time'] = pd.to_numeric(edges_df_renamed['travel_time'], errors='coerce')
                if edges_df_renamed['travel_time'].isnull().any():
                    self.logger.warning(f"Warning: After conversion, '{time_col}' still contains NaN values. These edge weights will be invalid.")

            if (edges_df_renamed['travel_time'] < 0).any():
                self.logger.warning(f"Warning: Column '{time_col}' contains negative values. This may cause shortest path algorithms (such as Dijkstra) to fail.")

            # check if distance contains non-numeric or negative values
            if not pd.api.types.is_numeric_dtype(edges_df_renamed['distance']):
                self.logger.warning(f"Warning: Column '{distance_col}' contains non-numeric data, trying to convert to numeric...")
                edges_df_renamed['distance'] = pd.to_numeric(edges_df_renamed['distance'], errors='coerce')
                if edges_df_renamed['distance'].isnull().any():
                    self.logger.warning(f"Warning: After conversion, '{distance_col}' still contains NaN values. These edge weights will be invalid.")

            if (edges_df_renamed['distance'] < 0).any():
                self.logger.warning(f"Warning: Column '{distance_col}' contains negative values. Shortest path calculations may be inaccurate.")

            for _, row in edges_df_renamed.iterrows():
                u = row[origin_col]
                v = row[destination_col]
                # ensure distance and travel_time are valid numeric values
                dist = row.get('distance')
                time = row.get('travel_time')

                # add edges with valid distance and time weights
                # (Dijkstra cannot handle NaN weights)
                if pd.notna(dist) and pd.notna(time) and dist >= 0 and time >= 0:
                    # store both weights when adding edges
                    G.add_edge(u, v, distance=dist, travel_time=time)
                else:
                    self.logger.warning(f"Warning: Skipping edge ({u}, {v}) because distance ({dist}) or travel_time ({time}) is invalid or negative.")

            self.logger.info(f"Network graph built. Contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            return G

        except FileNotFoundError as e:
            self.logger.error(f"Error: File not found - {e}")
            return None
        except KeyError as e:
            self.logger.error(f"Error: Missing required column names in CSV file - {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error building network: {e}")
            return None
        
    def get_shortest_path_details(self, start_node_id: int, end_node_id: int, weight_attribute: str) -> tp.Tuple[tp.List[int], float, float]:
        """
        Calculate the shortest path between two nodes in a network and return details.

        Args:
            start_node_id: ID of the starting node.
            end_node_id: ID of the ending node.
            weight_attribute (str): Weight to use for shortest path calculation ('distance' or 'travel_time').

        Returns:
            tuple: (path, total_distance, total_time)
                If no path is found, returns (None, None, None).
                If input is invalid, raises ValueError or KeyError.
        """
        if weight_attribute not in ['distance', 'travel_time']:
            raise ValueError("Weight attribute must be 'distance' or 'travel_time'")

        if start_node_id not in self.network:
            raise KeyError(f"Starting node ID '{start_node_id}' is not in the network.")
        if end_node_id not in self.network:
            raise KeyError(f"Ending node ID '{end_node_id}' is not in the network.")

        try:
            # use Dijkstra algorithm to find shortest path based on specified weight
            # NOTE: networkx's dijkstra_path only returns node list
            # shortest_path_length can directly get the total weight, but we need to calculate both totals
            self.logger.debug(f"Calculating shortest path from {start_node_id} to {end_node_id} based on '{weight_attribute}'...")
            path = nx.dijkstra_path(self.network, start_node_id, end_node_id, weight=weight_attribute)
            self.logger.debug("Path found.")

            # calculate the total distance and total time for this path
            total_distance = 0.0
            total_time = 0.0
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                edge_data = self.network.get_edge_data(u, v)
                if edge_data:
                    total_distance += edge_data.get('distance', 0)
                    total_time += edge_data.get('travel_time', 0)
                else:
                    raise Exception(f"Error: Edge data not found for edge ({u}, {v}).")

            # use shortest_path_length to verify the total length based on the selected weight
            calculated_weight_sum = nx.dijkstra_path_length(self.network, start_node_id, end_node_id, weight=weight_attribute)
            self.logger.debug(f"Verification: Total weight calculated using '{weight_attribute}': {calculated_weight_sum:.4f}")

            return path, round(total_distance, 3), round(total_time, 3)

        except nx.NetworkXNoPath:
            self.logger.error(f"Error: No path found between nodes {start_node_id} and {end_node_id}.")
            return None, None, None
        except Exception as e:
            self.logger.error(f"Error calculating shortest path: {e}")
            # may be due to invalid weight values (e.g., NaN or negative, though we try to filter these)
            return None, None, None
        
    def get_nodes_df(self) -> pd.DataFrame:
        return self.nodes_df
