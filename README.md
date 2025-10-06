# FlexiDepth Official Implementation

This repository contains the official implementation for the paper: **[Adaptive Layer-skipping in Pre-trained LLMs](https://arxiv.org/abs/2503.23798)**.

Our final trained model checkpoint is available on the Hugging Face Hub:

  * **[xuan-luo/FlexiDepth-Llama-3-8B-Instruct](https://huggingface.co/xuan-luo/FlexiDepth-Llama-3-8B-Instruct)**

## 🚀 Updated Training Strategy

This repository uses an updated two-stage training method that differs slightly from the direct training approach described in the paper. We found this new strategy yields improved results. The process is as follows:

1.  **Alignment Stage:** We first train the model for one epoch on a diverse, general-purpose instruction tuning dataset (`tulu-v3-sft-mixture`) using a high layer-skipping penalty. This teaches the model to learn effective and efficient layer-skipping patterns on a wide range of tasks.
2.  **Annealing Stage:** Next, we continue training for one more epoch on a dataset rich in reasoning tasks (math and code, using `open-perfectblend`) but with a lowered penalty. Surprisingly, we found that even after drastically reducing the penalty in this stage, the model does not significantly increase its average layer usage. However, this process leads to a substantial improvement in performance on reasoning tasks.

## 🛠️ How to Train

Follow these steps to set up the environment and run the training process.

### 1\. Setup Environment

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/luoxuan-cs/Flexidepth.git
cd Flexidepth
```

Next, install the required dependencies. You can either use the `requirements.txt` file or install the specific version of `trl` we used:

```bash
# Option 1: Using requirements.txt
pip install -r requirements.txt

# Option 2: Installing trl directly
pip install trl==0.23.0
```

### 2\. Prepare the Model

Create a `models/` directory and download our pre-trained FlexiDepth model.

```bash
mkdir models
cd models

# Clone our pre-trained model
hf download xuan-luo/FlexiDepth-Llama-3-8B-Instruct --local_dir ./FlexiDepth-Llama-3-8B-Instruct

cd ..
```

> **Note on Training from Scratch:** If you want to train a FlexiDepth model from a standard LLM (e.g., the original `Llama-3-8B-Instruct`), you must first manually initialize the router and adapter parameters with random values before beginning the Alignment Stage.

### 3\. Prepare Datasets

The tokenization scripts will automatically download the source datasets from Hugging Face and prepare them for training.

Navigate to the `datasets` subfolder and run the scripts:

```bash
cd datasets
python tokenize_tulu-v3.py
python tokenize_perfectblend.py
cd ..
```

This will create the tokenized data required for the next steps.

### 4\. Run Training

The training logic is split into two stages as described above. You will need to run the training script twice with different configurations.

#### Stage 1: Alignment Stage

In this stage, we train on the `tulu-v3` dataset for one epoch.

  * **Dataset**: `allenai/tulu-v3-sft-mixture`
  * **Penalty**: `1e-4`
  * **Global Batch Size**: 32 (assuming 8 GPUs)
  * **Epochs**: 1

Navigate to the `train` directory and execute the script:

```bash
cd train
python sft.py
```

*(Note: Ensure the parameters within `sft.py` are set for Stage 1, including the dataset path and the `1e-4` penalty.)*

> Note: If using gradient accumulation, you must manually normalize the layer-skipping loss because the trainer automatically normalizes the main prediction loss, but will not automatically normalize the custom layer-skipping loss.

#### Stage 2: Annealing Stage

After Stage 1 is complete, we continue training on the `open-perfectblend` dataset with a reduced penalty.

  * **Dataset**: `mlabonne/open-perfectblend`
  * **Penalty**: `1e-5` (reduced from `1e-4`)
  * **Epochs**: 1
  * All other hyperparameters remain the same.

You will need to modify the training script (`sft.py`) to point to the new dataset and update the penalty value to `1e-5`. Then, run the script again from the `train` directory:

```bash
# (Ensure you are still in the train/ directory)
python sft.py
```

## 📜 Citation

If you find our work useful, please consider citing the original paper:

```bibtex
@inproceedings{
luo2025adaptive,
title={Adaptive Layer-skipping in Pre-trained {LLM}s},
author={Xuan Luo and Weizhi Wang and Xifeng Yan},
booktitle={Second Conference on Language Modeling},
year={2025},
}
```
