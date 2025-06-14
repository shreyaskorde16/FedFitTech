

import matplotlib.pyplot as plt 
import seaborn as sns
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D 

import matplotlib.pyplot as plt
import seaborn as sns

def plot_f1_scores_baseline(df):
    print(df)
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # Calculate mean F1 score
    mean_f1_score_baseline = df['F1_score_Normal'].mean()
    mean_f1_score_baseline = round(mean_f1_score_baseline, 2)

    # Create the bar plot using seaborn
    ax = sns.barplot(
        x="Client_Id", 
        y="F1_score_Normal", 
        data=df, 
        color='blue', 
        width=0.6  # smaller width means thinner bars
    )  #edgecolor='black',

    # Add horizontal line for mean
    plt.axhline(y=mean_f1_score_baseline, color='black', linestyle='--', label="Mean F1 score for FedFitTech = {}".format(mean_f1_score_baseline))

    # Add labels and title
    plt.xlabel('Clients', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
    plt.title('F1 Score vs Clients for FedFitTech', fontsize=16, fontweight='bold')
    plt.xticks(rotation=0, fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold")
    plt.ylim(0, 1)
    plt.legend( fontsize=14, prop={'size': 16,'weight': 'bold'})
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig("F1_scores_baseline.eps", format="eps", dpi=1200, bbox_inches='tight')
    plt.savefig("F1_scores_baseline.svg", format="svg", dpi=1200, bbox_inches='tight')
    plt.savefig("F1_scores_baseline.pdf", format="pdf", dpi=1200, bbox_inches='tight')
    





def plot_f1_scores_comparison(df_distributed):
    
    # Convert to long format
    df_melted = df_distributed.melt(id_vars="Client_Id", var_name="Model", value_name="F1 Score")
    mean_f1_score_es = df_distributed['F1_score_ES'].mean()
    mean_f1_score_normal = df_distributed['F1_score_Normal'].mean()


    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    S_FL = "blue"
    ES_FL= "orange"
    

    # Plot using Seaborn
    ax = sns.barplot(data=df_melted, x="Client_Id", y="F1 Score", hue="Model", palette=[S_FL, "orange"], saturation=0.8)

    # Formatting
    plt.xlabel("", fontsize=14)
    plt.ylabel("", fontsize=14)
    plt.xticks(rotation=0, fontsize=18,  fontweight="bold")
    plt.yticks(fontsize=18, fontweight="bold")
    #plt.title("Comparison of F1 Scores", fontsize=16)

    # Add a horizontal line at the mean value of Validation F1 score
    mean_f1_score_es = round(mean_f1_score_es, 2)
    mean_f1_score_normal = round(mean_f1_score_normal, 2)
    line1 = plt.axhline(y=mean_f1_score_normal, color=S_FL, linestyle='--', label="Mean F1 score for Standard FL")
    line2 = plt.axhline(y=mean_f1_score_es, color="orange", linestyle='--', label="Mean F1 for Early stopping Employed FL")
    
    # Extract existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    print(handles)
    print(labels)
    
    # All legends to show
    #show_legend = [handles[0], handles[1], line1, line2]  # Show first two bars + both lines
    #show_labels = ["Clients F1 Score for FedFitTech","Clients F1 Score for ES-FedFitTech",
    #               f"Mean F1 Score for FedFitTech = {mean_f1_score_normal}", 
     #              f"Mean F1 Score for ES-FedFitTech = {mean_f1_score_es}"]

    show_legend = [line1, line2]  # Show first two bars + both lines
    show_labels = [f"Mean F1 Score for FedFitTech = {mean_f1_score_normal}", 
                   f"Mean F1 Score for Case Study = {mean_f1_score_es}"]



    # Add legend with selected handles
    plt.legend(show_legend, show_labels, fontsize=10, prop={'size': 16,'weight': 'bold'})
    
    #plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    #plt.legend(fontsize=14, )
    plt.tight_layout()  
    #plt.gca().legend_.remove() 
    # Save High-Quality Figure
    
    plt.savefig("F1_scores_comparison_double_bar_plot.eps", format="eps", dpi=1200, bbox_inches='tight')
    plt.savefig("F1_scores_comparison_double_bar_plot.svg", format="svg", dpi=1200, bbox_inches='tight')
    plt.savefig("F1_scores_comparison_double_bar_plot.pdf", format="pdf", dpi=1200, bbox_inches="tight")
    plt.clf()
    plt.close()
    print("F1 scores comparison double bar plot saved successfully")

    
    
    
    
def plot_f1_convergence(df_f1_scores):
  
    print(df_f1_scores.head())
    df_f1_scores.columns = [col.replace("Client_Id_", "") for col in df_f1_scores.columns]
    print(df_f1_scores.head())
    # Set the figure size
    plt.figure(figsize=(12, 8))
    
    # Step 1: Melt the DataFrame to long format
    df_melted = df_f1_scores.melt(id_vars=["Server_Round"], var_name="Client_Id", value_name="F1_Score")

    # Optional: Plot the mean F1 score across all clients
    mean_f1_scores = df_f1_scores.iloc[:, 1:].mean(axis=1)
    sns.set_context("talk", font_scale=1.2)  # Adjust context and font scale
    plt.rcParams['font.family'] = 'Times New Roman'  # Set font style
    sns.set_style("ticks")
    # Step 2: Create the line plot
    plt.figure(figsize=(12, 8)) # Set the figure size
   
    sns.lineplot(data=df_melted, x="Server_Round", y="F1_Score", hue="Client_Id", marker="o", markersize=3, linewidth=0.7,
                 alpha=0.9, palette="Paired")  # Create the line plot   style="Client_Id", palette="RdYlBu",  palette="Paired"
    
    # Plot the mean F1 score (in black) as a dashed line
    sns.lineplot(x=df_f1_scores["Server_Round"], y=mean_f1_scores, color="black", marker="^", markersize=5,
             linestyle="--", label="Mean F1", linewidth=1.4)
    
    # Customize the plot
    plt.xlabel('Server Round', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
    plt.title('F1 Score Convergence for 24 Clients Across 100 Global Rounds', fontsize=14, fontweight='bold')
    plt.legend(title="Clients", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # Place legend outside the plot

    plt.tight_layout()  
    #plt.gca().legend_.remove() 
    
    # Save High-Quality Figure
    plt.savefig("F1_scores_convergence_linegraph_baseline.eps", format="eps", dpi=1200, bbox_inches='tight')
    plt.savefig("F1_scores_convergence_linegraph_baseline.svg", format="svg", dpi=1200, bbox_inches='tight')
    plt.savefig("F1_scores_convergence_linegraph_baseline.pdf", format="pdf", dpi=1200, bbox_inches="tight")#
    plt.clf()
    plt.close()
    print("F1 scores convergence linegraph saved successfully")

    



def plot_heat_map_of_table(label_based_result_table: pd.DataFrame,
                           directory_name: os.path,
                           save_fig=True,
                           learning_type_name=None,
                           type=None,) -> plt:


    #any nan values will be 0.0
    label_based_result_table=label_based_result_table.fillna(0.0)
    # Set the figure size
    plt.figure(figsize=(12, 8))
    print(label_based_result_table.columns)
    
    labels_alpha = [chr(65 + i) for i in range(len(label_based_result_table.columns))]
    print(labels_alpha)
    
    printable_labels = {col:label for col,label in zip(label_based_result_table.columns,labels_alpha)}
    print(printable_labels)
    
    label_based_result_table.index = label_based_result_table.index.str.replace("Client_Id_", "")   
    print(label_based_result_table)
    
    label_based_result_table.columns = labels_alpha
    # Create the heatmap
    ax = sns.heatmap(label_based_result_table,
                annot=True,         # Show the F1-scores in each cell
                fmt=".2f",          # Format the numbers with two decimal places
                cmap="coolwarm",    # Use a diverging colormap to highlight low and high values
                annot_kws={"size": 11} ,
                cbar_kws={"shrink": 1}
               )  # Add a color bar label: cbar_kws={'label': 'F1 Score'}
    
    title = f"Federated Learning - F1-Score Heatmap {type}"
    #ax.figure.colorbar(ax.collections[0]).ax.yaxis.set_tick_params(labelsize=14)
    # Add titles and labels
    #plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("", fontsize=12, fontweight='bold')
    plt.ylabel("", fontsize=12, fontweight='bold')
    plt.xticks(rotation=0, fontsize=17, fontweight='bold')
    plt.yticks(fontsize=17, rotation=0, fontweight='bold')
    # Show the plot
    plt.tight_layout()
    plt.savefig(f"clients_vs_label_F1_scores_heatmaps_{type}.eps", format="eps", dpi=1200, bbox_inches='tight')
    plt.savefig(f"clients_vs_label_F1_scores_heatmaps_{type}.svg", format="svg", dpi=1200, bbox_inches='tight')
    plt.savefig(f"clients_vs_label_F1_scores_heatmaps_{type}.pdf", format="pdf", dpi=1200, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"clients_vs_label_F1_scores_heatmaps_{type} saved successfully")


def plot_f1_convergence_with_stop_round(df_ES_f1_vs_round, df_distributed_metrics_for_plot3):
    """
    df_ES_f1_vs_round (pd.Dataframe): contains f1 scoress for all rounds
    df_distributed_metrics_for_plot3: pd.Dataframe having clients ids with early stopping round
    
    """
    pd.set_option("display.max_rows", None)  # Show all rows

    pd.set_option("display.max_columns", None)  # Show all columns

    print(df_ES_f1_vs_round.head())
    df_ES_f1_vs_round.columns = [col.replace("Client_Id_", "") for col in df_ES_f1_vs_round.columns]
    print(df_ES_f1_vs_round.head())
    mean_f1_scores = df_ES_f1_vs_round.iloc[:, 1:].mean(axis=1)
    #print(df_distributed_metrics_for_plot3)     # columns as = Client_Id, F1_score_ES, F1_score_Normal, Training stop round

    for _, row in df_distributed_metrics_for_plot3.iterrows():
        if pd.notna(row["Training stop round"]):
            client_id = f"{int(row['Client_Id'])}"
            df_ES_f1_vs_round.loc[df_ES_f1_vs_round["Server_Round"] > row["Training stop round"], client_id] = np.nan
    
    df_melted_es_f1 =  df_ES_f1_vs_round.melt(id_vars=["Server_Round"], var_name="Client_Id", value_name="F1_Score")
    print(df_ES_f1_vs_round)   
    sns.set_context("talk", font_scale=1.2)  # Adjust context and font scale
    plt.rcParams['font.family'] = 'Times New Roman'  # Set font style
    sns.set_style("white")  
    
    
    # Get unique clients and assign colors using Seaborn's palette
    unique_clients = df_melted_es_f1["Client_Id"].unique()
    palette = sns.color_palette("Paired", len(unique_clients))
    client_colors = {client: color for client, color in zip(unique_clients, palette)}
 
    # Step 2: Create the line plot
    plt.figure(figsize=(12, 8)) # Set the figure size
   
    sns.lineplot(data=df_melted_es_f1, x="Server_Round", y="F1_Score", hue="Client_Id", 
                 linewidth=0.8, alpha=0.99, palette=client_colors)  # Create the line plot   style="Client_Id", palette="RdYlBu",  palette="Paired"
    
    # Plot the mean F1 score (in black) as a dashed line
    sns.lineplot(x=df_ES_f1_vs_round["Server_Round"], y=mean_f1_scores, color="black",
             linestyle="--",  linewidth=1)
    
    for _, row in df_distributed_metrics_for_plot3.iterrows():
        if pd.notna(row["Training stop round"]):
            stop_round = row["Training stop round"]
            client_id = int(row['Client_Id'])
            for _, row2 in df_ES_f1_vs_round.iterrows():
                x = stop_round
                y = df_ES_f1_vs_round.iloc[int(stop_round-1), client_id+1] 
                break
            line_color = client_colors[f"{int(client_id)}"]
            plt.plot(x, y, marker="^", linestyle="None", color=line_color, alpha=0.8, markersize=7)
            #plt.plot( x, y, linestyle="dashdot", color=line_color, alpha=0.8, linewidth=0.9) 
    #
    #early_stopping_legend = Line2D([0], [0], linestyle="dashdot", color="black", lw=1.5, label="Early Stopping Point")
        
    # Customize the plot
    plt.xlabel('', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.xticks(rotation=0, fontsize=18,  fontweight="bold")
    plt.yticks(fontsize=18, fontweight="bold")
    #plt.title('F1 Score Convergence for 24 Clients Across 100 Global Rounds with Early Stopping', fontsize=14, fontweight='bold')
    plt.legend(title="Clients", bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 13,'weight': 'bold'})  # Place legend outside the plot
    # Add legend with extra entry for early stopping
    #plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, handles=[early_stopping_legend] + plt.gca().get_legend().legend_handles)

    #plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlim(0)  # Start x-axis from 0
    plt.ylim(0)  # Start y-axis from 0
    plt.tight_layout() 
    #plt.gca().legend_.remove() 
    
    # Save High-Quality Figure
    plt.savefig("F1_scores_convergence_with_early_stopping_linegraph.eps", format="eps", dpi=1200, bbox_inches='tight')
    plt.savefig("F1_scores_convergence_with_early_stopping_linegraph.svg", format="svg", dpi=1200, bbox_inches='tight')
    plt.savefig("F1_scores_convergence_with_early_stopping_linegraph.pdf", format="pdf", dpi=1200, bbox_inches="tight")#
    plt.clf()
    plt.close()
    print("F1 scores convergence with early stopping linegraph saved successfully")
    
    
    
def plot_global_rounds(EA_dist_metric: pd.DataFrame, Global_rounds=100):
    
    EA_dist_metric["Training stop round"] = EA_dist_metric["Training stop round"].fillna(Global_rounds)
    plt.figure(figsize=(12, 8)) # Set the figure size
    df = EA_dist_metric[["Client_Id", "Training stop round"]]
    print(EA_dist_metric)
    
    sns.set_theme(style="whitegrid")
    # Plot a bar chart
    #sns.barplot(data=df, x="Client_Id", y="Training stop round", color= "red", saturation=0.5, width=0.6)
    
    
    
    EA_dist_metric["Remaining Rounds"] = Global_rounds - EA_dist_metric["Training stop round"]

    print(EA_dist_metric)
    
    # Plot stacked bars
    for i, row in EA_dist_metric.iterrows():
        # Plot the "Training stop round" part (bottom)
        plt.bar(i, row["Training stop round"], color="indianred", width=0.6)
        # Plot the "Remaining Rounds" part (top)
        plt.bar(i, row["Remaining Rounds"], bottom=row["Training stop round"], color="forestgreen", width=0.6)
    
    

    # Plot using Seaborn
    mean_global_round_value = int(EA_dist_metric["Training stop round"].mean())
    
    for _, row in EA_dist_metric.iterrows():
        if pd.notna(row["Training stop round"]):
                x = [row["Client_Id"], row["Client_Id"]]
                y_co = row["Training stop round"]
                y = [y_co, Global_rounds]
                comp_saved=int(Global_rounds-y_co)
                #print(f"x = {x} , y = {y_co}")
                #plt.plot( x, y, linestyle="dashdot", color="black", alpha=1, linewidth=1.5)
                #plt.text(x[0]+0.2, y[0]+3, f"CS = {comp_saved}", fontsize=10, color="black", rotation=90)
        else:
            x = [row["Client_Id"], row["Client_Id"]]
            y = [0, Global_rounds]
            #plt.plot( x, y, linestyle="dashdot", color="gray", alpha=0.7, linewidth=0.7)
            #plt.text(x[0]+0.05, y[0]+10, f"No Computation Saved", fontsize=10, color="gray", rotation=90)
                
    #plt.plot([], [], color="black", linestyle="dashdot", label="CS = Computation Saved (Server Rounds)") 
    # Formatting
    plt.xlabel("", fontsize=14)
    plt.ylabel("", fontsize=14)
    plt.xticks(ticks=EA_dist_metric["Client_Id"],rotation=0, fontsize=17, fontweight="bold")
    plt.yticks(fontsize=17, fontweight="bold")
    #plt.title("Global Round vs. Client ID with Computation Saved through Early Stopping", fontsize=16)

    # Add a horizontal line at the mean value of Validation F1 score
    plt.axhline(y=mean_global_round_value, color='black', linestyle='--', alpha=0.9,
                label=f"Mean Global Round = {mean_global_round_value}", zorder=10)
    
    # Add the mean value as text
    #plt.text(x=21, y=mean_global_round_value +1, s=f"Mean Global Round {mean_global_round_value}", color='black', fontsize=17, ha='center')
    #plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    #plt.legend(loc="upper left",fontsize= 12)
    plt.tight_layout()  
    # Save High-Quality Figure
    
    plt.savefig("Global_rounds_vs_clients_single_bar_plot.eps", format="eps", dpi=1200, bbox_inches='tight')
    plt.savefig("Global_rounds_vs_clients_single_bar_plot.svg", format="svg", dpi=1200, bbox_inches='tight')
    plt.savefig("Global_rounds_vs_clients_single_bar_plot.pdf", format="pdf", dpi=1200, bbox_inches="tight")
    plt.clf()
    plt.close()
    print("Global rounds vs clients single bar plot saved successfully")
