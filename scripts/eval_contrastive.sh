
device=0
checkpoint_dir="PATH_TO_CHECKPOINT_DIR"
perturb_method=all
# ["all", "clean", "deepwordbug", "textfooler", "checklist", "semantic"]
promptbench_eval_task=all
# ['all', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'wnli']
eval_sample_cnt=300

echo "device=${device}, checkpoint=${checkpoint_dir}, attack_method=${perturb_method}, promptbench_eval_task=${promptbench_eval_task}"
lora_target_modules=("q_proj" "v_proj" "k_proj" "o_proj")
CUDA_VISIBLE_DEVICES=${device} python run_contrastive_llama.py \
    --lora_target_modules "${lora_target_modules[@]}" \
    --do_predict \
    --perturb_method ${perturb_method} \
    --promptbench_eval_task ${promptbench_eval_task} \
    --eval_sample_cnt ${eval_sample_cnt} \
    --wandb_project CoIN \
    --wandb_usr YOUR_USERNAME \
    --wandb_run_name "CoIN_Eval" \
    --resume_from_checkpoint "${checkpoint_dir}" \
    --output_dir "${checkpoint_dir}"
