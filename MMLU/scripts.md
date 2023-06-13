# Ours
python run_mmlu_open_source.py --ckpt_dir /mnt/checkpoints/ours_6b7_DistDataV2_CCv3_300B_tohf/ --param_size 7 --model_type ours --ntrain 0

# Falcon-1.5T-7B (but need torch.v.2.0.1)
python run_mmlu_open_source.py --ckpt_dir tiiuae/falcon-7b --param_size 7 --model_type falcon --ntrain 0

# openlm-research/open_llama_7b_400bt_preview
python run_mmlu_open_source.py --ckpt_dir openlm-research/open_llama_7b_400bt_preview --param_size 7 --model_type gptj --ntrain 0

# pythia and redpajama
tokenizer -> use_fast=True

# llama
python run_mmlu_open_source.py --ckpt_dir pretrained=huggyllama/llama-7b --param_size 7 --model_type llama --ntrain 0




## 5 shot
python run_mmlu_open_source.py --ckpt_dir openlm-research/open_llama_7b_400bt_preview --param_size 7 --model_type gptj --ntrain 5 > ollma_400b.txt; python run_mmlu_open_source.py --ckpt_dir openlm-research/open_llama_7b_700bt_preview --param_size 7 --model_type gptj --ntrain 5 > ollma_700b; python run_mmlu_open_source.py --ckpt_dir openlm-research/open_llama_7b --param_size 7 --model_type gptj --ntrain 5 > ollma_1000b.txt; python run_mmlu_open_source.py --ckpt_dir stabilityai/stablelm-base-alpha-7b --param_size 7 --model_type gptj --ntrain 5 > s_lm.txt; python run_mmlu_open_source.py --ckpt_dir EleutherAI/gpt-j-6B --param_size 7 --model_type gptj --ntrain 5 > gptj.txt; python run_mmlu_open_source.py --ckpt_dir EleutherAI/pythia-6.9b-deduped --param_size 7 --model_type gptj --ntrain 5 > pythia.txt; 