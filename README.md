# AI-Scientist-Guide
## Introduction:
The AI Scientist is a multi-LLM agent framework capable of writing code, running experiments, visualizing plots, and composing academic papers. It was proposed in the 2024 paper by Lu et al., *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*. Due to its compatibility with differentiable physics simulations, the AI Scientist has significant potential for applications across numerous projects in the Logan Wright Lab. This repository serves as a basic guide to using the AI Scientist for research purposes. It is a work in progress and will be updated as we refine our use of the AI Scientist. The AI Scientist has been tested for its capability to generate research papers using the McMahon lab’s framework for Single-Photon-Detection Neural Networks (SPDNNs).

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
This file contains the core experiment code and should include all necessary functions to run the experiment. Contrary to (generally) good coding practice, the experiment.py file shouldn't rely on abstraction. Additionally, it is important to make sure that the file can accept arguments to specify where outputs are saved. After performing training and testing, the key results of the experiment must be saved in a `final_info.json` file in the specific output directory. This is represented in python as a dictionary, where the keys are strings describing metrics (e.g. `“training_time”`, `“train_accruacy”`), and the values are nested dictionaries containing a `“means”` key. The use of `{“means”: value}` in the nested dictionary is necessary and the AI Scientist won’t function without it. This nested dictionary translates to a structured way which lets the AI Scientist access the metric-related data as it lets the `final_info` dictionary be saved as a JSON file. 

### 2. `plot.py`
This file generates visualizations from experimental results. It reads `final_info.json` and generates plots that will be included in the final paper. While actual data and types of plots will vary from template to template, a general procedure can still be assigned to creating the `plot.py` file. Mainly, the file will first load the results outputted from `experiment.py`. 
sults_json, output_dir, model_name)

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
cd templates/your_template
```
Then run your `experiment.py` file. 

```
python experiment.py [insert extra argument commands if you need]
```

Now that you have your baseline run, cd to the main directory of the AI Scientist:

```
cd ../..
```

You can now run the AI scientist with the following command, where the experiment label is the name of your template folder!

```
python launch_scientist.py --model “gpt-4o-2024-08-06” --experiment [your experiment] --num-ideas [number of ideas]
```

Note your model, experiment name, and number of ideas may differ depending on your OpenAI API keys and template. Also, if you would like a shorter runtime you can also use the `--skip-novelty-check` flag to skip the novelty check the AI Scientist automatically performs. 

After running the AI Scientist, the results will be stored in the results/spdnn directory. For each evaluated hypothesis, you will find a `final_info.json file`, and then a `latex/` folder which contains the LaTeX file and figures for the generated paper. We had issues running TekLive on the Grace cluster, so we simply uploaded the generated LaTeX folder onto Overleaf with the appropriate figures and were able to produce a paper!

## Improving AI Scientist Output
in progress!

## Contact
For clarification, complaints, or issues, with this GitHub repository please reach out to tyler.chen.tzc2@yale.edu or jinchen.zhao@yale.edu.

## References
C. Lu, C. Lu, R. Tjarko, J. Foerster, J. Clune, D. Ha, "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." 
arXiv:2408.06292 (2024)\
\
S.-Y. Ma, T. Wang, J. Laydevant, L. G. Wright and P. L. McMahon. "Quantum-noise-limited optical neural networks operating at a few quanta per activation." arXiv:2307.15712 (2023)

