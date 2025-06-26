# MAARTA

## Table of Contents

- [Dataset](#dataset)
- [Usage](#usage)

# Simulated Error Dataset: <a name="dataset"></a>

Due to unavailability of Real world error dataset, this study was conducted on the simulated error dataset:

https://drive.google.com/drive/folders/1RzlGzvJ9Dl01dgrNlhNedY7JWhQ3pmJE?usp=sharing

It contains two files:

1. Error dataset with missing or masked fixations
2. Labels for the the cases

## Dataset Generation

This dataset was derived from the [EGD-CXR](https://physionet.org/content/egd-cxr/1.0.0/) dataset. The EGD-CXR dataset was created using an eye-tracking system to monitor a radiologist's gaze while interpreting and reading 1,083 publicly available chest X-ray (CXR) images. We provide a Python file within the `dataset_generation` directory with the code used to generate the simulated error dataset. To reproduce the synthesized error dataset, you will need to download the original EGD-CXR dataset from PhysioNet and extract the audio_segmentation_transcripts and fixation folders as displayed within the `dataset_generation` directory. Then run the `egd_cxr_processing.py` file with the following command:

```bash
python3 egd_cxr_processing.py
```

For convenience we have provided the respective folders in the `dataset_generation` directory and simply running the `egd_cxr_processing.py` file will generate the synthesized error dataset.

# Usage <a name="usage"></a>

Within this repository, we provide python files that utilizes the synthesized perceptual error dataset and evaluates the SCoT framework against standard chain of thought (CoT) prompting in zero-shot and few-shot settings on the synthesized error dataset highlighting its effectiveness in improving multimodal reasoning across different LLM/LMM models. In this study the models used are Mistral-7B-Instruct-v0.3, LLAMA-3.2-11B-Vision-Instruct and GPT-4o-Mini. The Mistral and Llama models are accessed using [together.ai's](https://www.together.ai/) API. GPT-4o-Mini is accessed using [OpenAI's](https://openai.com/api/) API.

- These python files are located in the `./zero_shot`, `./few_shot` directories of this repository.
- To run these programs you will need a together.ai API key and OpenAI API key to perform requests to these models.
- These API keys can be obtained by signing up for an account on the respective platforms and following their instructions for generating API keys.
- The API keys will need to be inserted into the code in the respective files located near the top of the file.
- These scripts are designed to be run from the command line and requires Python 3.8 or higher.

```bash
Usage:
  python3 ./file - [flags]

Flags:
  --data (required)                File path to our synthesized error dataset file with missing or masked fixations
  --metadata (required)            File path to our synthesized error dataset metadata file containing labels for corresponding cases
  --results (optional)             File path to preexisting results output file generated to continue appending results from the LLM/LMM models
```
