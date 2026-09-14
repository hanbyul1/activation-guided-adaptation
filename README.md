# Permanent Activation-Guided Representation Adaptation in Transformer Language Models

## An Empirical Study of Software Maintainability Properties

This repository contains the implementation and experimental artifacts for the study:

**Permanent Activation-Guided Representation Adaptation in Transformer Language Models: An Empirical Study of Software Maintainability Properties**

The study investigates whether activation representations associated with software maintainability properties can be permanently internalized into a pretrained code language model through activation-guided parameter adaptation. The evaluated properties are:

- Complexity
- Coupling
- Modularity

The experiments use **StarCoderBase-1B** and instances derived from **SWE-bench**. Permanent adaptation is implemented using LoRA and compared with inference-time activation steering under the same experimental setting.

## Repository Structure

    activation-guided-adaptation/
    ├── README.md
    ├── requirements.txt
    ├── scripts/
    │   ├── prepare_dataset_and_labels.py
    │   ├── discover_activation_centroids.py
    │   ├── train_activation_guided_lora_forward_hooks_v4_fixed.py
    │   ├── evaluate_activation_guided_model.py
    │   ├── train_activation_steering.py
    │   ├── evaluate_activation_steering.py
    │   ├── summarize_results.py
    │   └── visualize_comparative_results.py
    ├── data/
    │   ├── prepared_swebench/
    │   └── activation_centroids/
    ├── results/
    │   ├── permanent_adaptation/
    │   ├── activation_steering/
    │   └── training/
    │       ├── complexity/
    │       ├── coupling/
    │       ├── modularity/
    │       └── steering/
    │           ├── complexity/
    │           ├── coupling/
    │           └── modularity/
    └── figures/

## Experimental Pipeline

The experimental workflow consists of the following stages.

### 1. Dataset Preparation and Property Labeling

    python scripts/prepare_dataset_and_labels.py

This stage prepares the SWE-bench instances and assigns deterministic structural-property labels for complexity, coupling, and modularity.

Prepared dataset artifacts are provided in `data/prepared_swebench/`.

### 2. Activation-Centroid Discovery

    python scripts/discover_activation_centroids.py

This stage analyzes internal model activations and identifies property-specific layers and activation centroids.

The resulting artifacts are provided in `data/activation_centroids/`.

### 3. Permanent Activation-Guided Adaptation

    python scripts/train_activation_guided_lora_forward_hooks_v4_fixed.py

The script performs property-specific LoRA adaptation using the joint language-modeling and activation-guidance objective.

Training artifacts for complexity, coupling, and modularity are provided in `results/training/`.

### 4. Permanent-Adaptation Evaluation

    python scripts/evaluate_activation_guided_model.py

Evaluation artifacts are provided in `results/permanent_adaptation/`. They include activation-level results, output-level results, capability-preservation measurements, per-instance results, and generated predictions.

### 5. Activation-Steering Training

    python scripts/train_activation_steering.py

Property-specific steering coefficients and training artifacts are provided in `results/training/steering/`.

### 6. Activation-Steering Evaluation

    python scripts/evaluate_activation_steering.py

Evaluation artifacts are provided in `results/activation_steering/`. These results provide the inference-time steering baseline used for comparison with permanent adaptation.

### 7. Result Summarization

    python scripts/summarize_results.py

### 8. Comparative Visualization

    python scripts/visualize_comparative_results.py

The resulting comparative figures are available in `figures/`.

## Environment

The experiments were conducted using:

    Python       3.11.14
    PyTorch      2.9.1
    Transformers 4.57.3
    PEFT         0.19.1
    Datasets     4.4.1
    NumPy        2.2.0
    Matplotlib   3.10.6

Install the required Python packages using:

    pip install -r requirements.txt

## Large Generated Artifacts

Several large intermediate PyTorch artifacts are intentionally excluded from the repository because they can be regenerated using the provided scripts:

    data/activation_centroids/all_layer_activations.pt
    data/activation_centroids/selected_layer_activations.pt
    data/prepared_swebench/prepared_dataset.pt

The property centroids required for subsequent experiments are included in:

    data/activation_centroids/property_centroids.pt

## Results

The repository provides aggregate and per-instance results for permanent activation-guided adaptation and inference-time activation steering across complexity, coupling, and modularity.

The artifacts include activation-level behavior, output-level property behavior, capability-preservation measurements, and generated predictions.

## Reproducibility

Configuration files, training histories, validation summaries, property-specific steering coefficients, evaluation summaries, per-instance results, and generated predictions are provided.

Large intermediate activation tensors and merged model checkpoints are not included because they can be regenerated using the supplied scripts and configurations.

## Data and Models

The experiments use the publicly available SWE-bench dataset and the StarCoderBase-1B pretrained model. These external resources remain subject to their respective licenses and terms of use.

## Citation

If you use this repository, please cite:

    Dae-Kyoo Kim.
    "Permanent Activation-Guided Representation Adaptation in Transformer Language Models:
    An Empirical Study of Software Maintainability Properties."

Publication information will be added after publication.

## License

The source code in this repository is provided for research and reproducibility purposes. Third-party datasets and pretrained models are subject to their respective licenses.