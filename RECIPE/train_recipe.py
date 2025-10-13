#%%
# nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_recipe.py -mn 'meditron-7b' -sample_n 2534 -dn 'medfe' -bs 8 -epo 200 -tn 'medrek_medfe_500sample' --device 'auto' --model_parallel > medrek_train_meditron_medfe_500_prompt_token_know_proto_use_mlp_qk.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python train_recipe.py -mn 'meditron-7b' -dn 'medmcqa' -bs 8 -epo 160 -tn 'medrek_medmcqa_bs8_160ep' --device 'cuda:0' --model_parallel > medrek_medmcqa_bs8_160ep.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 python train_recipe.py -mn 'meditron-7b' -dn 'medmcqa' -bs 8 -epo 160 -tn 'recipe_medmcqa_bs8_160ep' --device 'cuda:0' --model_parallel > recipe_medmcqa_bs8_160ep.log 2>&1 &

from editors.recipe.data import RECIPETrainData
from utils.utils import get_model_editor_config_path, model_path_map
from transformers import  AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from editors.recipe.recipe import RECIPE, RECIPEConfig
from utils.global_attrs import ROOT_PATH
import os
import torch
import time


def train_recipe(model_name:str, data_name, n_epoch, batch_size, train_name, device, load_ckpt_path, model_parallel, sample_n):
    # model_path, config_path = get_model_editor_config_path(model_name, 'recipe')
    batch_size=int(batch_size)
    # model_name=f"/mnt/sj/LLM_checkpoint/{model_name}-llama"
    model_name=f"/home/xiashujun/LLM_checkpoint/{model_name}"
    if args.device == 'auto':
        device = None
    device_map = 'auto' if model_parallel else device
    dtype = torch.float32
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device) 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True
    ) 
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if 'meditron' in model_name:
        config_path = "configs/recipe/RECIPE-meditron.yaml"
    elif 'huatuo' in model_name:
        config_path = "configs/recipe/RECIPE-huatuo.yaml"
    config = RECIPEConfig.from_yaml(config_path)
    recipe = RECIPE(model, tokenizer, config, device, 'models/roberta-base')
    rtd = RECIPETrainData(tokenizer, sample_n, data_name, None, False, device)  
    start_time=time.time()
    train_name=f'epoch{n_epoch}_'+train_name
    recipe.train_init(rtd.sample_count, rtd.get_data_by_ids, 
        batch_size = batch_size, 
        records_dir = os.path.join(ROOT_PATH, 'train_records'), 
        train_name_prefix = None, 
        train_name = train_name, 
        load_ckpt_path = load_ckpt_path, 
        save_ckpt_per_i = 3000, 
        log_per_i = 10, 
        random_seed = 1)

    n_epoch=int(n_epoch)
    recipe.train(n_epoch) 
    cost_t=(time.time()-start_time)/60/60
    print(f'train for {n_epoch} epochs takes time: {cost_t} hours')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str)
    parser.add_argument('-dn', '--data_name', type=str)
    parser.add_argument('-sample_n', '--sample_number', type=int)
    parser.add_argument('-tn', '--train_name', type=str)
    parser.add_argument('-bs', '--batch_size', type=str)
    parser.add_argument('-epo', '--epoch', type=int)
    parser.add_argument('-ckpt', '--checkpoint', type=str, default = None)
    parser.add_argument('--device', type=str, default = None)
    parser.add_argument("--model_parallel", action="store_true")
    args = parser.parse_args()
    print(args)
    train_recipe(args.model_name, args.data_name,args.epoch,  args.batch_size, args.train_name, args.device, args.checkpoint, args.model_parallel, args.sample_number)