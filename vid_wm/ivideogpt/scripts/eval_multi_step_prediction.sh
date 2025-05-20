# using one GPU

python eval_vgpt_multiturn.py --per_device_eval_batch_size 4 \
    --config_name configs/vgpt/ctx_llama_small.json\
    --dataset_path {path to preprocessed data} \
    --pretrained_model_name_or_path {path to pretrained compressive tokenizer} \
    --pretrained_transformer_path {path to pretrained multi-step pred transformer} \
    --processor_type ctx_msp \
    --output_jsonl eval_jsonl/vgpt_small_ctx_msp8_head12_fulleval_release.jsonl \
    --max_decode_batchsize 1 \
    --segment_length 8 \
    --use_eval_dataset \
    --max_eval_iters 400 \
    --exp_name vgpt_small_ctx_msp8_head12_fulleval