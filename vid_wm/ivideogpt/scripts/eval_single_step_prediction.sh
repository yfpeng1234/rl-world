# using one GPU

python eval_vgpt.py --per_device_eval_batch_size 4 \
    --dataset_path {path to preprocessed data} \
    --pretrained_model_name_or_path {path to pretrained perframe tokenizer} \
    --pretrained_transformer_path {path to pretrained single-step pred transformer} \
    --processor_type simple \
    --output_jsonl eval_jsonl/vgpt_small_multi1_head12_fulleval_release.jsonl \
    --use_eval_dataset \
    --max_eval_iters 400 \
    --exp_name vgpt_small_multi1_head12_fulleval