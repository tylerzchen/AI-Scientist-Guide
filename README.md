# AI-Scientist-Guide
## Introduction:
The AI Scientist is a multi-LLM agent framework capable of writing code, running experiments, visualizing plots, and composing academic papers. It was proposed in the 2024 paper by Lu et al., *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*. Due to its compatibility with differentiable physics simulations, the AI Scientist has significant potential for applications across numerous projects in the Logan Wright Lab. This repository serves as a basic guide to using the AI Scientist for research purposes. It is a work in progress and will be updated as we refine our use of the AI Scientist. The AI Scientist has been tested for its capability to generate research papers using the McMahon lab’s framework for Single-Photon-Detection Neural Networks (SPDNNs). A summary of our results will be added soon under the **Results from SPDNN Testing section**.

## Requirements and Installation
The AI Scientist codebase requires the use of NVIDIA GPUs which can utilize CUDA and PyTorch. While it is possible to run AI Scientist templates on CPU machines, runtime for the generation becomes extremely lengthy. For that reason, we recommended using the Grace computing clusters to run the AI Scientist. Please note that we ran the AI scientist by running commands directly on the Grace cluster terminal while editing code through a Jupyter notebook session. While these instructions should still work, the computing clusters can be funny at times and steps may differ if you are using the Grace cluster in an alternative manner (e.g. sshing through VSCode, etc.). Additionally, please note that the following instructions do not utilize Slurm, so if you run the commands that follow it will require that you keep your computer/Grace cluster terminal open (it can be in the background) while running the AI Scientist. Paper generation can be fairly time intensive even with GPUs, so if you would like to try and run the AI scientist with Slurm please see the following [Grace cluster documentation on Slurm](https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/)for tips.

### 1. Uploading the Code
To start, upload the AI Scientist code to the Grace cluster. You can do this by cloning the GitHub repository directly:
```bash
git clone https://github.com/SakanaAI/AI-Scientist.git
```
Alternatively, download the ZIP file from GitHub. If you are using the SPDNN template, clone/download files from the corresponding repository instead.

### 2. Allocating GPU resources
After the code is uploaded, allocate a GPU for running the code:
```bash
salloc -G 1 -p gpu_devel -t 6:00:00
```
You can verify that the GPU allocated is an NVIDIA GPU with the following command:
```bash
watch -n 1 nvidia-smi
```
For additional options for GPU allocation, please refer to this [Grace Cluster documentation on salloc](https://docs.ycrc.yale.edu/clusters/grace/).

### 3. Setting up the Environment
Next, create and activate a Conda environment specifically for the AI Scientist dependencies:
```bash
ml miniconda
conda create -yn <environment-name>
conda activate <environment name>
```

After creating the environment, navigate to the directory containing the AI Scientist code and install dependencies:
```bash
pip install -r requirements.txt
```

While we were succesfull using `pip install` if you encounter errors with the command you may need to use `conda install` instead. For more detailon how to use conda install or creating a conda environment with predownloaded packages, please refer to [Haoyu's guide on using grace cluster](https://app.slack.com/client/T04J7N1E4G7/C04JHSMA807).

### 4. Exporting the API Key
After installing the requirements, export the API key for the appropriate LLM. The AI Scientist supports multiple models, including OpenAI’s GPT-4o and GPT-4o-mini, as well as Anthropic’s Claude Sonnet 3.5. For testing the SPDNN template, we used OpenAI's GPT-4o, exporting the key with the following command:

```bash
export OPENAI_API_KEY = "insert_your_key"
```

### 5. Optional: Texlive for LaTeX to PDF Conversion
If you would like to use Texlive for automatic LaTeX to PDF conversion:
```bash
ml teklive
```
If this fails, an alternative command may be:
```bash
ml texlive/20220321-GCC-12.2.0
```
Please note that we were unable to get Teklive to work on the Gracecluster. Instead, we manually uploaded the generated LaTeX code to Overleaf to produce the final PDF.

## Template Creation

The AI Scientist requires five key files to generate a paper:

- **`experiment.py`**: Core experiment code.
- **`plot.py`**: Generates plots from experimental results.
- **`prompt.json`**: Contains the LLM prompt with experiment details.
- **`seed_ideas.json`**: Includes hypotheses for the LLM to test.
- **`latex/template.tex`**: The LaTeX template for the manuscript.

Each file must conform to the AI Scientist framework. Detailed instructions for each file are provided below.

### 1. `experiment.py`
This file contains the core experiment code and should include all necessary functions to run the experiment. Contrary to (generally) good coding practice, the experiment.py file shouldn't rely on abstraction. The SPDNN GitHub repository initially contained three files with code which was used for training, but to make it compatible with the AI scientist, the code from all three files was put into the experiment.py file. ![SPDNN code to experiment.py](https://ibb.co/3YR0x0n) Additionally, it is important to make sure that the file can accept arguments to specify where outputs are saved. This can be achieved by using the Python argparse module. This is done with the following code in the SPDNN template:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPDNN Experiment")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="run_0", help="Output directory")
    parser.add_argument("--image_size", type=int, default=28, help="Image size for MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--n_linear", type=int, default=400, help="Number of neurons in linear layer")
    parser.add_argument("--n_channels", type=int, nargs='+', default=[16], help="Channels for conv layers")
    parser.add_argument("--kernel_sizes", type=int, nargs='+', default=[5], help="Kernel sizes for conv layers")
    parser.add_argument("--strides", type=int, nargs='+', default=[1], help="Strides for conv layers")
    parser.add_argument("--model_type", type=str, choices=["conv", "mlp"], default="conv",
                        help="Model type: 'conv' for ConvNN, 'mlp' for MLP.")

    args = parser.parse_args()
    config = vars(args)
    print("Experiment Configuration:", config)
    main(config)
```

This code block is useful as it allows the AI Scientist to pass runtime configuration parameters to the experiment programmatically. The key lines to pay attention to are the following:
```python
parser.add_argument("--output_dir", type=str, default="run_0", help="Output directory")
```
and
```python
main(config)
```
The first line is important as it will allow the outputs of the training/testing to be stored in a run_0 folder, which will later be accessed and utilized by the AI Scientist.

The second line, main(config) is important because it calls the main function that performs the experimentation. In our case, this is loading. It will set up the output directories, load the MNIST dataset, perform training and then also testing. The key block of code in the main function which is also required for compability is the following:
```python
    final_info = {
    "training_time": {"means": round(time.time() - start_time, 2)},
    "train_accuracy": {"means": train_acc_track},
    "test_accuracy": {"means": test_acc_track},
    "best_accuracy": {"means": round(best_acc, 4)},
    "eval_loss": {"means": round(test_loss, 4)}
    }

    # Use the correct output directory
    final_info_path = os.path.join(config["output_dir"], "final_info.json")
    with open(final_info_path, "w") as f:
        json.dump(final_info, f, indent=4)

    # Save model checkpoint
    model_path = os.path.join(config["output_dir"], f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
```
After performing training and testing, the key results of the experiment must be saved in a `final_info.json` file in the specific output directory. This is represented in python as a dictionary, where the keys are strings describing metrics (e.g. `“training_time”`, `“train_accruacy”`), and the values are nested dictionaries containing a `“means”` key. The use of `{“means”: value}` in the nested dictionary is necessary and the AI Scientist won’t function without it. This nested dictionary translates to a structured way which lets the AI Scientist access the metric-related data as it lets the `final_info` dictionary be saved as a JSON file. 

### 2. `plot.py`
This file generates visualizations from experimental results. It reads `final_info.json` and generates plots that will be included in the final paper. While actual data and types of plots will vary from template to template, a general procedure can still be assigned to creating the `plot.py` file. Mainly, the file will first load the results outputted from `experiment.py`. For example, the SPDNN template `plot.py` file has the following function:

```python
def plot_accuracy_curves(results_file, output_dir, model_name):
    ensure_dir(output_dir)
    
    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)

    train_acc = results["train_accuracy"]
    test_acc = results["test_accuracy"]
    epochs = range(1, len(train_acc) + 1)
```

This data (in the same function) is then used to programmatically generate a plot using matplotlib, and the output is saved to a specified directory:
```python
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
```
Similarly to the `experiment.py` file, a main function must be used which the AI Scientist can use to call the plot functions:
```python
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

if __name__ == "__main__":
    main()
```
This function defines the appropriate paths, and then calls the plotting function. Taken in sum, these functions can serve as a general outline for how to write the `plot.py` file.

### 3. `prompt.json`
This file provides structured information to the LLM. and has two main components, the system and task_description. The system is meant to define the role of the LLM, while the task description describes information about the experiment. This can be represented in the following manner in the `prompt.json` file:

```json
{
    "system": "You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.",
    "task_description": "You are given the following experiment.py file to work with, that trains a Single-Photon-Detection Neural Network (SPDNN). In essence, SPDNNs are uniquely designed to detect only a few photons with each dot product or neuron activation during inference. The models trained here can be integrated on various Optical Neural Network (ONN) platforms equipped with single-photon detectors (SPDs). In order to include the stochastic single-photon detection process during training, the program employs physics-aware stochastic training to incorporate the actual stochastic physical process forward pass, while eliminating the stochasticity in the backward pass. Interesting ideas would involve benchmarking the SPDNNs on different tasks, and also trying out new reparametrization of activation functions (e.g. using a Gumbel-Softmax reparam)."
}
```

### 4. `seed_ideas.json`
The seed_ideas.json file contains the initial hypothesis for the LLM to test. It is made up of six components – the name of the idea, the title of the generated paper, a brief experiment title, the interestingness of the experiment, the feasibility of the experiment, and then the novelty of the experiment. It is structured in the following manner:
```
[
    {
        "Name": "photon_activation_function",
        "Title": "Training SPDNN Models with Photon Activation Functions on MNIST Dataset.",
        "Experiment": "In this experiment, we will compare the performance of SPDNN MLP and SPDNN Convolutional Neural Networks and evaluate them based on training loss, test accuracy, and training time.",
        "Interestingness": 4,
        "Feasibility": 10,
        "Novelty": 3
    }
]
```

### 5. `latex/template.tex`
The last component of the file is the `latex/template.tex` file. The latex folder contains a template.tex file which serves as the LaTeX template for the paper generation. If you would like to generate a paper with a specific journal template, you can alter the files in the latex folder. However, for our SPDNN template we just used the provided latex folder and `template.tex` file.

## Template Use
Once you have prepared the appropriate template files, you can run the AI Scientist. The first step in this is running the baseline experiment. First navigate to the folder containing the template:

```bash
cd templates/spdnn
```
Then run your `experiment.py` file. In the case of the SPDNN template, that could be done with the following command (note there are different configurations but this is just one example):

```bash
python experiment.py --output_dir run_0 --model_type mlp --epochs 10 --batch_size 128 --lr 0.01 --momentum 0.9
```

Now that you have your baseline run, cd to the main directory of the AI Scientist:

```
cd ../..
```

You can now run the AI scientist with the following command, where the experiment label is the name of your template folder!

```
python launch_scientist.py --model “gpt-4o-2024-08-06” --experiment spdnn --num-ideas 1
```

Note your model, experiment name, and number of ideas may differ depending on your OpenAI API keys and template. Also, if you would like a shorter runtime you can also use the `--skip-novelty-check` flag to skip the novelty check the AI Scientist automatically performs. 

After running the AI Scientist, the results will be stored in the results/spdnn directory. For each evaluated hypothesis, you will find a `final_info.json file`, and then a `latex/` folder which contains the LaTeX file and figures for the generated paper. We had issues running TekLive on the Grace cluster, so we simply uploaded the generated LaTeX folder onto Overleaf with the appropriate figures and were able to produce a paper!

## Improving AI Scientist Output
in progress!

## Results from SPDNN Testing
in progress!

## Contact
For clarification, complaints, or issues, with this GitHub repository please reach out to tyler.chen.tzc2@yale.edu or jinchen.zhao@yale.edu.

## References
C. Lu, C. Lu, R. Tjarko, J. Foerster, J. Clune, D. Ha, "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." 
arXiv:2408.06292 (2024)
S.-Y. Ma, T. Wang, J. Laydevant, L. G. Wright and P. L. McMahon. "Quantum-noise-limited optical neural networks operating at a few quanta per activation." arXiv:2307.15712 (2023)

