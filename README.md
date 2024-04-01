# CoIN 
This repository is the official implementation of [Contrastive Instruction Tuning](https://arxiv.org/abs/2402.11138).

## Installation
### Dependency
Experiments are run in the following environment:

| Package        | Version   |
|----------------|-----------|
| conda          |   22.9.0  |
| Python         |   3.8     |
| CUDA           |   11.8    |

### Install via Conda and Pip

```bash
conda create -n coin python=3.8
conda activate coin
pip install -r requirements.txt
```

## Data
The original data source of our new dataset is the FLAN collection, specifically from [Muennighoff/flan](https://huggingface.co/datasets/Muennighoff/flan) on Huggingface.
We follow the steps discussed in section 3.2 of the paper to curate dataset for CoIN, which is available [here](dataset/contrastive_flan_data.csv).
- Each entry contains:
  - The original instruction-input pair (```original_instruction```)
  - The paraphrased instruction-input pair (```paraphrased_instruction```)
  - Label (```targets```)
  - Task name
  - Keyword data (a dictionary that contains KV pairs that will be parsed into the instruction templates to get the full input).
- Instruction templates are available [here](utils/flan_preprocessing_constants.py).
- Every entry at the odd index is the hard negative for the entry above it.

## Training
Parameters are defined in [run_contrastive.sh](scripts/run_contrastive.sh). Check ```ContrastiveLlamaTrainingArgument``` in [run_contrastive_llama.py](run_contrastive_llama.py) for more details regarding default values of all parameters.
- To start training the CoIN model, please run the following:
  ```
  bash scripts/run_contrastive.sh
  ```
- To run the continually instruction-tuned model (training with data augmentation only), change ```do_contrastive``` to FALSE.

## Evaluation
### Run Evaluation on Unseen Instructions
In this project, we follow [PromptBench](https://github.com/microsoft/promptbench) to add perturbations to instructions. All perturbed instructions for 10 GLUE tasks are available [here](promptbench/config.py).
To evaluate a model, please:
- Go to [eval_contrastive.sh](scripts/eval_contrastive.sh)
- Change ```checkpoint_dir``` to the path of your checkpoint/output directory
- Run:
  ```
  bash scripts/eval_contrastive.sh
  ```
- You can change ```perturb_method``` and ```promptbench_eval_task``` to evaluate the model on different perturbation methods and evaluation tasks. Supported perturbation methods and tasks are available in the bash script and ```UnseenInstructionEvalArgs``` in [run_contrastive_llama.py](run_contrastive_llama.py).

### Postprocessing of Evaluation Results
To obtain average accuracy(exact match) and standard deviation of the model on the perturbed instructions for each task, please run:
```
python promptbench/postprocessing.py --output_dir "YOUR_OUTPUT_DIR"
```
- The evaluation script will store model's outputs to the directory named ```preds``` under your model's checkpoint directory.
- Substitute ```YOUR_OUTPUT_DIR``` with the path where the outputs are stored (e.g. ```output/CoIN/preds```).
- The script will produce a csv file named ```unseen_instruction_acc.csv``` under ```YOUR_OUTPUT_DIR```.

## Citation
```
@article{yan2024contrastive,
  title={Contrastive Instruction Tuning},
  author={Yan, Tianyi and Wang, Fei and Huang, James Y and Zhou, Wenxuan and Yin, Fan and Galstyan, Aram and Yin, Wenpeng and Chen, Muhao},
  journal={arXiv preprint arXiv:2402.11138},
  year={2024}
}
```
