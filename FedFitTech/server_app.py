"""FedFitTech: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, ConfigsRecord, RecordSet
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from FedFitTech.task import  get_weights
import sys
import os
from typing import List, Tuple
import torch
from omegaconf import DictConfig, OmegaConf
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from FedFitTech.flwr_utils.client_utils import *
from FedFitTech.flwr_utils.server_plotting_function import *
from FedFitTech.my_strategy import CustomFedAvg


def fit_config(server_round: int):
    """Returns the configuration dictionary for each round."""
    
    return {"server_round": server_round}

def evaluate_config(server_round: int):
    """Returns the configuration dictionary for each round."""
    
    return {"server_round": server_round}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """A function that will aggregate the metrics from all clients after evaluation"""
    
    #metrics_df = weighted_average_plottinng(metrics, plt_path)

    return {}



def server_fn(context: Context):
        
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cuda is available on Server = {torch.cuda.is_available()}\n")    
  
    net, cfg = get_net_and_config()
    print(OmegaConf.to_yaml(cfg))

    # Read from config
    num_rounds = cfg.GLOBAL_ROUND
    fraction_fit = cfg.fraction_fit

    # Initialize model parameters
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    cfg.plt_path, cfg.model_path, cfg.csv_files_path, cfg.root_log_path = get_model_plot_directory(plt_dir=True, model_dir=True, csv_dir=True, config=cfg) 
    
    # Create F1 result table for all Global rounds
    f1_result_table = pd.DataFrame(np.nan, index=range(cfg.GLOBAL_ROUND), columns=['Server_Round'] + [f'Client_Id_{i}' for i in range(0, cfg.Total_Clients)])
    f1_result_table["Server_Round"] = range(1, cfg.GLOBAL_ROUND + 1)
    
    # F1 scores for all clients and labels
    f1_client_vs_labels_table = pd.DataFrame(np.nan,  index=range(cfg.Total_Clients), columns=list(cfg.labels_set.keys()))
    f1_client_vs_labels_table.index = [f'Client_Id_{i}' for i in range(cfg.Total_Clients)]
    
    # Define strategy
    strategy = CustomFedAvg(
        cfg=cfg,
        model_path = cfg.model_path,
        plot_path = cfg.plt_path,
        csv_dir_path = cfg.csv_files_path,
        val_f1_df = f1_result_table,
        val_f1_client_vs_labels_table = f1_client_vs_labels_table,
        fraction_fit = fraction_fit,
        fraction_evaluate = cfg.fraction_evaluate,
        min_available_clients = int(cfg.min_available_clients),
        initial_parameters = parameters,  
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn = fit_config,
        on_evaluate_config_fn=evaluate_config,
            
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
