# Visual Attention Sink in Large Multimodal Models

This repository contains the code for [See What You Are Told: Visual Attention Sink in Large Multimodal Models](https://openreview.net/forum?id=7uDI7w5RQA&nesting=2&sort=date-desc), an approach designed to improve the object hallucination in large multimodal models. This experimental repository builds upon the foundation of [LLaVA](https://github.com/haotian-liu/LLaVA) by introducing custom modifications and enhancements.

---

## Table of Contents

- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Dataset Structure](#dataset-structure)
- [Running Experiments](#running-experiments)
- [Acknowledgements](#acknowledgements)

---

## Installation

Begin by cloning the repository and navigating into the project directory:

```bash
$ git clone 
$ cd VisAttnSink
```

## Environment Setup

To reproduce the exact development environment, follow these steps:

1. Create the Conda Environment:

```bash
$ conda create -n VisAttnSink -f env_conda.txt
```

2. Install the Python Dependencies:

```bash
$ pip install -r env_pip.txt
```

These commands ensure that your environment matches the dependencies required for the project.

## Dataset Structure

The repository expects a specific directory structure for the datasets. Ensure your datasets follow the format below:

```text
./D_datasets
└── {{ DATASET_NAME }}
    ├── Questions
    │   └── {{ CATEGORY_NAME }}-questions.jsonl
    └── Images
        ├── {{ IMAGE_NAME_1 }}.jpg
        ├── {{ IMAGE_NAME_2 }}.jpg
        └── ...
```

Replace the placeholder values (e.g., {{ DATASET_NAME }}, {{ CATEGORY_NAME }}) with the actual names corresponding to your dataset.

## Running Experiments

Before running an experiment, configure your experiment settings using a YAML configuration file. Below is an example configuration:

```yaml
name_exp: {{ EXPERIMENT_NAME }}
name_dataset: {{ DATASET_NAME }}
name_category: {{ CATEGORY_NAME }}

path_image_dir: {{ PATH_IMAGE_DIR }}
path_question_dir: D_datasets/{{ DATASET_NAME }}/Questions
path_model: {{ PATH_MODEL }}

conv_mode: vicuna_v1

logic: 1
tau: 20
rho: 0.5
beta: 0.6
summ: 0.2

max_new_tokens: 128
except_last_layer: 1

<COMMENTS>
Make sure to update the placeholder values with your specific settings:
    - name_exp: Name of your experiment.
    - name_dataset: The dataset name.
    - name_category: The category for the questions.
    - path_image_dir: Directory path containing images.
    - path_question_dir: Directory path for question files.
    - path_model: Path to the model checkpoint.
</COMMENTS>
```

## Evaluation

We follow the LLaVA evaluation methodology. For detailed evaluation instructions, please refer to this [link](https://github.com/haotian-liu/LLaVA/tree/main/llava/eval).

## License

All code and algorithm logic in this repository are licensed under the Apache 2.0 license.

## Acknowledgements

This repository is built on the foundation provided by LLaVA and integrates several experimental enhancements aimed at optimizing visual attention in large multimodal models.

For any questions or issues, please open an issue on the repository.

