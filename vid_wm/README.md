# Video World Models with RLVR

## Method Overview

![video world model](assets/vid_wm.png)

<!-- ## Installation -->

## Data Preparation

Download the RT1 dataset from [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) and extract single episodes as `.npz` files:

```bash
python oxe_data_converter.py --dataset_name {dataset name, e.g. bridge} --input_path {path to downloaded OXE} --output_path {path to stored npz}
```

## Pre-training

Single-step prediction:

```bash
cd ivideogpt
bash scripts/train_perframe_tokenizer.sh
bash scripts/train_single_step_prediction.sh
```

Multi-step prediction:

```bash
cd ivideogpt
bash scripts/train_compressive_tokenizer.sh
bash scripts/train_multi_step_prediction.sh
```

## Post-training with RLVR

Single-step prediction:

```bash
cd verl
bash examples/grpo_trainer/run_vgpt.sh \
    trainer.experiment_name='vgpt'\
    processor.processor_type=simple \
    data.video.dataset_path={path to preprocessed data} \
    processor.tokenizer.path={path to pretrained perframe tokenizer} \
    actor_rollout_ref.model.path={path to pretrained single-step pred transformer} \
    data.max_response_length=321 \
    trainer.val_before_train=True trainer.test_freq=10 trainer.save_freq=10 \
    actor_rollout_ref.rollout.n=16
```

Multi-step prediction:

```bash
cd verl
bash examples/grpo_trainer/run_ctx_msp_vgpt.sh \
    trainer.experiment_name='ctx_vgpt_msp8' \
    trainer.reward_fn=mae \
    data.video.dataset_path={path to preprocessed data} \
    processor.tokenizer.path={path to pretrained compressive tokenizer} \
    actor_rollout_ref.model.path={path to pretrained multi-step pred transformer} \
    trainer.val_before_train=True trainer.test_freq=10 trainer.save_freq=10 \
    actor_rollout_ref.rollout.n=16
```

## Video Prediction Evaluation

```bash
cd ivideogpt
bash eval_single_step_prediction.sh
bash eval_multi_step_prediction.sh
```

## Application: Real2Sim Policy Evaluation

```bash
cd ivideogpt
bash scripts/eval_policy.sh
```

## Acknowledgement

Our `verl` directory is forked from commit [`15263cb`](https://github.com/volcengine/verl/tree/15263cb86a464264edb1e5462675e25ddf6ff9d8) of official repo.