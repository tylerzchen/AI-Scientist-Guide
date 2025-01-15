import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Helper to ensure the results directory exists
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# Function to plot training vs test accuracy over epochs
def plot_accuracy_curves(results_file, output_dir, model_name):
    """
    Plots training and test accuracy over epochs.
    Args:
        results_file (str): Path to the results JSON file.
        output_dir (str): Path to save the plots.
        model_name (str): Descriptive model name for the title and file name.
    """
    ensure_dir(output_dir)
    
    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)

    train_acc = results["train_accuracy"]
    test_acc = results["test_accuracy"]
    epochs = range(1, len(train_acc) + 1)

    # Plot
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Training Accuracy", lw=1.8)
    plt.plot(epochs, test_acc, label="Test Accuracy", lw=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy.png"))
    plt.close()
    print(f"Saved accuracy plot to {output_dir}/{model_name}_accuracy.png")


# Function to plot accuracy vs SPD readouts (K)
def plot_accuracy_vs_shots(results_df_file, output_dir, title="Accuracy vs Shots"):
    """
    Plots accuracy vs SPD readouts (K) with error bars.
    Args:
        results_df_file (str): Path to the results DataFrame saved as a pickle.
        output_dir (str): Path to save the plots.
        title (str): Title for the plot.
    """
    ensure_dir(output_dir)

    results_df = pd.read_pickle(results_df_file)

    # Extract unique K values and model configurations
    Ks = results_df["K"].unique()
    Ns = results_df["N"].unique()
    common_kwg = {"lw": 1.2, "capsize": 0, "alpha": 0.9}

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    for N in Ns:
        means = results_df.loc[results_df["N"] == N]["mean_accuracy"]
        stds = results_df.loc[results_df["N"] == N]["std_dev"]
        plt.errorbar(Ks, means, yerr=stds, fmt="o-", ms=5, label=f"Hidden Neurons: {N}", **common_kwg)

    plt.xlabel("Shots of SPD Readouts per Activation (K)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_shots.png"))
    plt.close()
    print(f"Saved accuracy vs shots plot to {output_dir}/accuracy_vs_shots.png")


# Function to plot accuracy vs hidden neurons (N) for various K
def plot_accuracy_vs_hidden_neurons(results_df_file, output_dir, title="Accuracy vs Hidden Neurons"):
    """
    Plots accuracy vs number of hidden neurons (N) for different SPD readout shots (K).
    Args:
        results_df_file (str): Path to the results DataFrame saved as a pickle.
        output_dir (str): Path to save the plots.
        title (str): Title for the plot.
    """
    ensure_dir(output_dir)

    results_df = pd.read_pickle(results_df_file)

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    common_kwg = {"lw": 1.2, "alpha": 0.9, "capsize": 0}

    Ks = [0, 10, 5, 3, 2, 1]  # Specific K values to plot
    Ns = results_df["N"].unique()

    for K in Ks:
        means = results_df.loc[results_df["K"] == K]["mean_accuracy"]
        plt.plot(Ns, means, label=f"K={K}", **common_kwg)

    plt.xlabel("Number of Hidden Neurons (N)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_hidden_neurons.png"))
    plt.close()
    print(f"Saved accuracy vs hidden neurons plot to {output_dir}/accuracy_vs_hidden_neurons.png")

# Main function to call plot functions
def main():
    # Paths to data
    output_dir = "./plots"  # Directory to save plots
    results_dir = "./run_0/results"  # Directory where final_info.json files are saved

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Model name (adjust as per your experiment)
    model_type = "MLP"  # Set "ConvNN" for convolutional network
    model_name = f"MNIST_28x28_{model_type}_N400_SGD_lr0.01_mom0.9"

    # File paths
    results_json = os.path.join(results_dir, f"{model_name}_results.json")
    results_df_mnist = os.path.join(results_dir, "MNIST_coh_simacc")  # Path to MNIST pickle file

    # Check file existence for results.json
    if not os.path.exists(results_json):
        raise FileNotFoundError(f"Results file not found: {results_json}")

    # Plot accuracy curves
    plot_accuracy_curves(results_json, output_dir, model_name)

    # Check file existence for results_df_mnist
    if os.path.exists(results_df_mnist):
        # Plot accuracy vs shots (K)
        plot_accuracy_vs_shots(results_df_mnist, output_dir, title="MNIST Test Accuracy vs SPD Readouts (K)")

        # Plot accuracy vs hidden neurons (N)
        plot_accuracy_vs_hidden_neurons(results_df_mnist, output_dir, title="MNIST Test Accuracy vs Hidden Neurons (N)")
    else:
        print(f"Warning: {results_df_mnist} not found. Skipping accuracy vs shots/neurons plots.")

if __name__ == "__main__":
    main()
