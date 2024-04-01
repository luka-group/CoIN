
lora_target_modules=("q_proj" "k_proj" "v_proj" "o_proj")

CUDA_VISIBLE_DEVICES=0 python run_contrastive_llama.py \
    --lora_target_modules "${lora_target_modules[@]}" \
    --do_train \
    --batch_size 64 \
    --cutoff_len 256 \
    --base_model "yahma/llama-7b-hf" \
    --lora_weight "tloen/alpaca-lora-7b" \
    --data_path "dataset/contrastive_flan_data.csv" \
    --val_set_size 2000 \
    --use_contrastive_data TRUE \
    --do_contrastive TRUE \
    --wandb_project CoIN \
    --wandb_usr YOUR_USERNAME \
    --wandb_run_name "CoIN" \
    --output_dir "outputs/CoIN"