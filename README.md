# MedREK


Source code for MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts.

## Setup
First enter the `medrek` directory: `cd medrek`

In `utils/global_attrs.py`, change  `ROOT_PATH` and `MODEL_PATH` to your paths.

## Train MedREK

Please run:
```
python train_medrek.py -mn 'meditron-7b' -dn 'medcf'  
```
Checkpoints will be saved in `train_records/recipe/meditron-7b/train_name/checkpoints/`.
You can view training information in `train_records/recipe/meditron-7b/train_name/logs/` through Tensorboard.

## Evaluate MedREK

Please run:
```
python test_medrek.py -en 'medrek' -mn 'meditron-7b' -et 'batch' -dvc 'cuda:0' -ckpt 'train_records/recipe/meditron-7b/train_name/checkpoints/a_checkpoint' -dn 'medcf' -edn 100
```
You can check results in `eval_results/medrek`.
