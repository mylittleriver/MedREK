# MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts.

Source code for **MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts**.


## 📰 News

* [2025/10/16] 🚀 Our paper is available on arXiv! 
* [2025/10/15] 🔥 We release the MedVersa dataset and code for MekREK!


## 🔧 Setup

```bash
conda create -n medrek python=3.10 -y
conda activate medrek
git clone https://github.com/mylittleriver/MedREK
pip install --upgrade medrek 
pip install -r requirements.txt
```

First enter the `medrek` directory: `cd medrek`

In `utils/global_attrs.py`, change  `ROOT_PATH` and `MODEL_PATH` to your paths.

## 📊 MedVersa

Please check the `/medversa` floder to see the `train\valid\test` split of our enhanced **Medical LLM Model Editing** Benchmark.


## ⚙️ Train MedREK

Please run:
``` bash
python train_medrek.py -mn 'meditron-7b' -dn 'medcf'  
```
Checkpoints will be saved in `train_records/recipe/meditron-7b/train_name/checkpoints/`.

You can view training information in `train_records/recipe/meditron-7b/train_name/logs/` through Tensorboard.

We also provide the implementation of [RECIPE](https://arxiv.org/abs/2405.03279) for medical LLM model editing. Please check the `RECIPE` floder.

## 💫 Evaluate MedREK

Please run:
```bash
python test_medrek.py -en 'medrek' -mn 'meditron-7b' -et 'batch' -dvc 'cuda:0' \
    -ckpt 'train_records/recipe/meditron-7b/train_name/checkpoints/a_checkpoint' \
    -dn 'medcf' -edn 100 \
```
You can check results in `eval_results/medrek`.


## 🙏 Acknowledgement
This repo is built upon the following projects: 

* [RECIPE](https://github.com/qizhou000/RECIPE)
* [MedLaSA](https://github.com/quqxui/MedLaSA)

We thank the authors for their codes.
