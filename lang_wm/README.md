# Language World Models with RLVR

## Method Overview

![language world model](assets/lang_wm.png)

# Language World Model

## Install
```bash
# Python 3.10 or 3.11 recommended
conda create -n verl python=3.10
cd verl
pip install -e .
```

## Data Preparation
### Text Game Simulator
All data has been uploaded to [TODO]. You can either download them directly or generate them by following the ``README.md`` in ``data_process_for_text_game_simulator``:

###  Web Page
The SFT and RLVR training datasets have been uploaded to Hugging Face.

The content of both datasets is essentially the same, with only the key names differing to match their respective training scripts.

Please download them and place them directly in this folder.

## Supervised Fine-Tuning (SFT)


### Text Game Simulator
Run the following command. Please modify the following paths: training set, validation set, model, and output. (Since we select the best epoch based on accuracy on the validation set rather than loss, we do not separately split a validation set for SFT. You can simply use the same path for both the training and validation sets.)

```
bash verl/examples/sft/text_game_simulator/run_text_game_simulator_sft.sh
```

### Web Page

To train the model, run the following command:

```
bash verl/examples/sft/webagent/run_web_agent_sft.sh
```

After training, the model will be saved in `default_local_dir` specified in the script.

To merge the LoRA weights into the base model, run the following command:

```bash
python verl/merge.py
```

You have to specify the directory for LoRA weights in the script.


## Post-training with RLVR


### Text Game Simulator
Run the following command. This command uses binary reward by default. If you want to use the task-specific reward described in the paper, simply modify the two parameters ``data.sample_no_gold_data_num`` and ``reward_model.text_game_reward_type`` as indicated in the comments.

```
bash examples/grpo_trainer/run_web_agnet_rl.sh
```

### Web Page
Run the following command to train:

```
bash examples/grpo_trainer/run_web_agnet_rl.sh
```


The trained model will be saved in `default_local_dir` specified in the script.

To merge weights, run the following command:

```bash
python verl/scripts/model_merger.py --local_dir log/rl/DeepSeek-R1-Distill-Qwen-1.5B-merged-grpo-reward-v1-p-0.1-final-v2/checkpoints/global_step_xxxx/actor --output_dir <output_dir> --backend fsdp --hf_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
```

## Web Agents with language world model
Code and instruction can be found in ``webagent/``

## Acknowledgements

Our code is heavily based off the <a href="https://github.com/volcengine/verl" target="_blank">verl codebase</a>.
