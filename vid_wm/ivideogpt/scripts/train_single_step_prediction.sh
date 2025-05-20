# using 8x40G A100 GPUs

accelerate launch train_vgpt.py \
    --per_device_train_batch_size 4 \
    --config_name configs/vgpt/llama_small.json \
    --dataset_path {path to preprocessed data} \
    --pretrained_model_name_or_path {path to pretrained perframe tokenizer} \
    --output_dir trm-output \
    --skip_first_val \
    --exp_name vgpt_small_multi1_head12