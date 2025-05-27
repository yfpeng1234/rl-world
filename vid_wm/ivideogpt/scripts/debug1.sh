# using one GPU

python eval_vgpt_multiturn.py --per_device_eval_batch_size 4 \
    --config_name configs/vgpt/ctx_llama_small.json\
    --dataset_path $DATASET_PATH \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH/rt1-compressive-tokenizer \
    --pretrained_transformer_path $PRETRAINED_MODEL_PATH/rt1-world-model-multi-step-base \
    --processor_type ctx_msp \
    --output_jsonl eval_jsonl/vgpt_small_ctx_msp8_head12_fulleval_release_base_debug.jsonl \
    --max_decode_batchsize 1 \
    --segment_length 8 \
    --use_eval_dataset \
    --max_eval_iters 400 \
    --debug \
    --exp_name vgpt_small_ctx_msp8_head12_fulleval_base_debug $@