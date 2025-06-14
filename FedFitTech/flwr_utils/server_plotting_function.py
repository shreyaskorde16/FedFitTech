#----------------------------------------------------------------------------------------
# Implementation of Federated learning using Flower Framework with WEAR dataset 
# Functions needed only for server_app
#----------------------------------------------------------------------------------------
# Adaption by: Shreyas Korde
# E-Mail: shreyas.korde@student.uni-siegen.de
#----------------------------------------------------------------------------------------

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json



def ploting_averaage_metrics(df, plt_path, server_round="Final"):
    server_round = str(server_round)
    # Set Seaborn style for better visuals
    sns.set(style="whitegrid")
   
    # Create the plot using Seaborn barplot
    plt.figure(figsize=(8, 6))  # Optional: Set figure size
    
    palette = ["blue"]
    sns.barplot(x='Client_Id', y='Validation F1 score', data=df, palette=palette)

    # Calculate the mean of the Validation F1 score
    mean_f1_score = df['Validation F1 score'].mean()

    # Add a horizontal line at the mean value of Validation F1 score
    plt.axhline(y=mean_f1_score, color='red', linestyle='--', label=f'Mean F1 Score: {mean_f1_score:.2f}')

    # Add labels and title
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Validation F1 Score', fontsize=12)
    plt.title(f'Validation F1 Score vs Client ID for {server_round} Round', fontsize=14)

    # Display the mean value in the plot legend
    plt.legend()
    plt.grid(True)
    # Save the plot to a folder (replace 'path/to/folder' with your desired path)
    save_path = os.path.join(plt_path, f'distributed_f1score_{server_round}_round.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to free up memory and avoid errors
      
    
    
def weighted_average_plottinng(metrics, plt_path, server_round="final"):
    """
    A funtion to convert aggrigrasted metrics to a dataframe and save it as a CSV file
    metrics: List[Tuple[int, Metrics]] - A list of metrics from all clients
    plt_path: str - The path to save the CSV file
    """
    server_round = str(server_round)
    data = {}
    data["Client_Id"] = []
    data["Validation loss"] = []
    data["Validation Accuracy"] = []
    data["Validation Precision"] = []
    data["Validation Recall"] = []
    data["Validation F1 score"] = []
    
 
    
    for _, m in metrics:
        data["Client_Id"].append(int(m["Client_id"]))
        print(f"validaion loss in plotting {m['Validation loss']}")
        data["Validation loss"].append(float(m["Validation loss"]))
        data["Validation Accuracy"].append(float(m["Validation Accuracy"]))
        data["Validation Precision"].append(float(m["Validation Precision"]))
        data["Validation Recall"].append(float(m["Validation Recall"]))
        data["Validation F1 score"].append(float(m["Validation F1 score"]))
        
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='Client_Id')
    print(df_sorted)
    
    ploting_averaage_metrics(df_sorted, plt_path) # plot the average f1 score

    csv_path = os.path.join(plt_path, f'client_distributed_metrics_{server_round}_round.csv')
    df_sorted.to_csv(csv_path, index=False, sep=';', quoting=csv.QUOTE_MINIMAL)
    
    return df_sorted

  
    
def weighted_eval_average_plottinng(plt_path: str, csv_path:str, best_eval_f1: pd.DataFrame, 
                                    val_f1_df: pd.DataFrame, f1_labels_df: pd.DataFrame, cfg, results, server_round):
    """
    A funtion to convert aggrigrasted metrics to a dataframe and save it as a CSV file
    metrics: List[Tuple[int, Metrics]] - A list of metrics from all clients
    plt_path: str - The path to save the plots
    """
    
    #best_f1 = float(0)
    best_df = None
    best_server_round = None
    data = {}
    data["Client_Id"] = []
    data["Validation loss"] = []
    data["Validation Accuracy"] = []
    data["Validation Precision"] = []
    data["Validation Recall"] = []
    data["Validation F1 score"] = []
    data["Number of Val Examples(Btach)"] = []
    data["Training stop round"]	= []
    data["Number of Training Examples"] = []
 
    
    for _, m in results:
        data["Client_Id"].append(int(m.metrics["Client_id"]))
        print(f"validaion loss in plotting {m.metrics['Validation loss']}")
        data["Validation loss"].append(float(m.metrics["Validation loss"]))
        data["Validation Accuracy"].append(float(m.metrics["Validation Accuracy"]))
        data["Validation Precision"].append(float(m.metrics["Validation Precision"]))
        data["Validation Recall"].append(float(m.metrics["Validation Recall"]))
        data["Validation F1 score"].append(float(m.metrics["Validation F1 score"]))
        data["Number of Training Examples"].append(m.metrics["Number of Training Examples"])
        data["Number of Val Examples(Btach)"].append(m.num_examples)
        data["Training stop round"].append(m.metrics["Training_stop_round"])
        
    if server_round == cfg.GLOBAL_ROUND:
        f1_labels_scores = {} #dict containing clients: metrics
        for _, m in results:
            f1_labels_str = m.metrics["F1_labels_result"]   # str of results
            f1_labels_dict = json.loads(f1_labels_str)      # Converts str to dict
            f1_labels_scores[f"Client_Id_{m.metrics['Client_id']}"] = f1_labels_dict
            for client in f1_labels_scores.keys():
                for label in f1_labels_scores[client].keys():
                    if label in cfg.labels_set.keys():
                        f1_labels_df.loc[client, label] = round(f1_labels_scores[client][label]['f1-score'], 2)
            #print(f1_labels_df)
            #f1_labels_csv_path = os.path.join(csv_path, f'Client_vs_label_F1scores_{server_round}_round.csv')
            #f1_labels_df.to_csv(f1_labels_csv_path, index=True, sep=';', quoting=csv.QUOTE_MINIMAL)
    
    if server_round == cfg.GLOBAL_ROUND:
        # Save F1_scores_vs_labels_clientwise_dict into jasonn format        
        f1_labels_scores_json_path = os.path.join(cfg.root_log_path, 'F1_scores_vs_label_clientwise.json')
        with open(f1_labels_scores_json_path, "w") as json_file:
            json.dump(f1_labels_scores, json_file, indent=4) 
        f1_labels_csv_path = os.path.join(csv_path, f'Client_vs_label_F1scores_{server_round}_round.csv')
        f1_labels_df.to_csv(f1_labels_csv_path, index=True, sep=';', quoting=csv.QUOTE_MINIMAL)
        print(f1_labels_scores)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)  
      
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='Client_Id')
    print(df_sorted)
    
    current_mean_f1_score = df['Validation F1 score'].mean()
    if best_eval_f1 is None:
        best_eval_f1 = current_mean_f1_score
    elif best_eval_f1 is not None:
        if current_mean_f1_score > best_eval_f1: 
            best_eval_f1 = current_mean_f1_score
            best_df = df_sorted
            best_server_round = server_round
            best_log_path = os.path.join(plt_path, "Best_eval_f1")
            os.makedirs(best_log_path, exist_ok=True) 
            ploting_averaage_metrics(best_df, best_log_path, server_round=best_server_round)
    
    ploting_averaage_metrics(df_sorted, plt_path, server_round=server_round) # plot the average f1 score
    
    # save the metrics to a CSV file
    metrics_csv_path = os.path.join(csv_path, f'Client_Distributed_Metrics_{server_round}_round.csv')
    df_sorted.to_csv(metrics_csv_path, index=False, sep=';', quoting=csv.QUOTE_MINIMAL)
    
    # Update the val_f1_df with the current round's F1 scores
    for _, row in df_sorted.iterrows():
        val_f1_df.iloc[server_round - 1, int(row['Client_Id']+1)] = row["Validation F1 score"]
        
    f1_score_csv_path = os.path.join(csv_path, f'Validation_F1_Scores.csv')
    #result_table.loc[result_table["Server_Round"] <= server_round]
    transform_val_f1_Df = val_f1_df.loc[val_f1_df["Server_Round"] <= server_round]
    transform_val_f1_Df.to_csv(f1_score_csv_path, index=False, sep=';', quoting=csv.QUOTE_MINIMAL)
    
    
    return df_sorted, best_eval_f1, val_f1_df