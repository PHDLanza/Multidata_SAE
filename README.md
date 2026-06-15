<!-- <!-- # Multidata_SAE

A project for analyzing multimodal concepts using SAE (Self-Attention Encoder). Right now we focus only on the VLMs injected in `LlaVA 8B-OV` and the dataset `LlaVA-Next`, however the framework can be easly adapted for being active with different VLMs and dataset.

## Highlights
- Modular pipeline: embedding extraction, hypothesis generation, and evaluation.
- Configuration via gin files (configs/).
- Outputs are JSON files written to the configured PATH_EMBEDDING.
- Recommended GPU: A100 or H100

## Usage

For module of the framework you need to execute a different 
`runnners`:


Here the list:
 1. Extraction of embedding  
 2. Generate visual and textual concepts hypotheses
 3. Evaluation with CLIP and ALIGN scores

Each of these modules will produce one or more json files as output, they will all be saved in the  `PATH_EMBEDDING` parameter 
 

Notes:
- Recommended GPU: A100 or H100

## Prepare the enviroment

1. Create a virtual enviroment and install all the requirements:
```bash
    ./setup.sh
```

### Gin Configuration Files

This project uses [gin-config](https://github.com/google/gin-config) files to manage hyperparameters and experiment settings. Each module has an associated `.gin` file in the `configs/` directory, specifying model paths, dataset locations, and other options.

**To run the framework end-to-end:**
1. Review and edit the relevant `.gin` files to set correct paths for your data, model, and output directories.
2. Make sure the dataset paths in the gin files match where you have placed your data.
3. Adjust any hyperparameters as needed for your experiments.

**Example:**  
To change the dataset location, open `configs/config_general.gin` and update the `DATASET_LLAVA_PATH` parameter:
```gin
DATASET_LLAVA_PATH = "/path/to/your/llava-next"
```

Repeat this for each module's gin file before running the corresponding script.

  

## Run the code

To execute each module, run the corresponding Python script from the project root:

1. **Extract embeddings**
```bash
python run_extract_llava.py --layer <layer> --id <0|1|2|3|4|-1>
```
- --layer: layer name or index to extract (e.g. "select-layer" or a numeric index)
- --id: dataset split to analyze. Valid values: 0,1,2,3,4. Use -1 to process all sections (default).


2. **Generate visual and textual concept hypotheses**
```bash
python run_generate_hypotheses.py --id <0|1|2|3|4> --mode <textual|visual|both>
```
- --id: dataset split to analyze
- --mode: modality for hypothesis generation

3. **Evaluation**
    ```bash
python run_evaluation.py --mode <textual|visual>
```
- --mode: modality to evaluate (textual or visual)
- Evaluation supports CLIP and ALIGN scoring
This repository framework for analyzing multimodal concepts using SAEs, here the [paper](). The current focus is on VLMs injected in LLaVA 8B-OV and the LLaVA-Next dataset, but the framework is designed to be adaptable to other VLMs and datasets.  

 -->

# Extraction of multimodal concepts through SAE

This repository provides a modular pipeline to extract and analyze multimodal concepts using Sparse AutoEncoder (SAEs). All the details and results will presented at ICANN 2026 [paper](). The current reference setup targets VLMs injected into LLaVA 8B-OV and the LLaVA-Next dataset, but the codebase is adaptable to other vision-language models and datasets.



## Requirements


Installation

1. Create a Python virtual environment and install dependencies:
```bash
./setup.sh
```
2. Recommended GPU: 1 or 2 A100 or H100
## Configuration (gin)
This project uses [gin-config](https://github.com/google/gin-config) files to manage hyperparameters and experiment settings. Each module has an associated `.gin` file in the `configs/` directory,
After cloning the repo, update paths and other relevant values in `configs/config_general.gin` 

```gin
DATASET_LLAVA_PATH = "/path/to/your/llava-next"
EMBEDDING_PATH = "/path/to/output/embeddings"
```
Repeat for other config files as needed before running modules.

Pipeline modules and usage
All modules write one or more JSON files to EMBEDDING_PATH.


### 1) Extract embeddings
```bash
python run_extract_llava.py 
```

### 2) Generate visual and textual concept hypotheses
```bash
python run_generate_hypotheses_llava.py 
```

### 3) Evaluation is divided in two main files for both metrics 

```bash
python evaluation/run_concept_evaluation.py
```
and

```bash
python evaluation/run_autointerpretability.py
```








