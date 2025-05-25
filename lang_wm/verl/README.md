# Language World Model

## Install
```bash
# Python 3.10 or 3.11 recommended
conda create -n verl python=3.10
pip install -e .
```

## Web Page Prediction

### Data

The SFT and RLVR training datasets have been uploaded to Hugging Face.

The content of both datasets is essentially the same, with only the key names differing to match their respective training scripts.

Please download them and place them directly in this folder.

### SFT

To train the model, run the following command:

```
bash examples/sft/webagent/run_web_agent_sft.sh
```

After training, the model will be saved in `default_local_dir` specified in the script.

To merge the LoRA weights into the base model, run the following command:

```bash
python merge.py
```

You have to specify the directory for LoRA weights in the script.

### RLVR

run the following command to train:

```
bash examples/grpo_trainer/run_web_agnet_rl.sh
```

The trained model will be saved in `default_local_dir` specified in the script.

To merge weights, run the following command:

```bash
python scripts/model_merger.py --local_dir log/rl/DeepSeek-R1-Distill-Qwen-1.5B-merged-grpo-reward-v1-p-0.1-final-v2/checkpoints/global_step_xxxx/actor --output_dir <output_dir> --backend fsdp --hf_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
```

## Acknowledgements

Our code is heavily based off the <a href="https://github.com/volcengine/verl" target="_blank">verl codebase</a>.
