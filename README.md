# dgx_spark_summary_from_human_feedback

This repo contains the code for reproducing [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) in one DGX Spark. [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031) has already done this in an 8-H100 cluster. We will refer to this work for many details.

# Steps
1. Supervised Fine-tuning
2. Reward Model Training
3. Reinforcement learning from human feedback

# Supervised Fine-tuning
I will first use [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) as the base model and its tokenizer to preprocess the TLDR dataset. Checking the padding token and eos token of the tokenizer,

```python
from transformers import AutoTokenizer
model = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model)
print(f"padding token id: {tokenizer.pad_token_id}, eos token id: {tokenizer.eos_token_id}")
```

```
padding token id: 151643, eos token id: 151643
padding token: <|endoftext|>, eos token: <|endoftext|>
```

## SFT Dataset
We use [summarize_from_feedback_tldr_3_filtered](https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered) as the SFT dataset.

1. for each reference summary, add a prepending whitespace and a trailing <|endoftext|> token. The maximum length of tokens of the reference response after formatting is **63**

To generate and upload the SFT dataset to Hugging Face, run:
```bash
python3 -m dgx_spark_summary_from_human_feedback.process_sft_dataset
```

## Supervised Fine-tuning

### Training Configuration

> Note: each episode is one prompt-response pair.

> Note: Effective batch size is batch_size * gradient_accumulation_steps = 16 * 8 = 128.

<table>
<tr>
<th>Parameter</th>
<th>Value</th>
</tr>
<tr>
<td>num_train_epochs</td>
<td>1(116,722 episodes)</td>
</tr>
<tr>
<td>batch_size</td>
<td>16</td>
</tr>
<tr>
<td>gradient_accumulation_steps</td>
<td>8</td>
</tr>
<tr>
<td>learning_rate</td>
<td>3e-06</td>
</tr>
<tr>
<td>min_learning_rate</td>
<td>3e-07</td>
</tr>
<tr>
<td>warmup_steps</td>
<td>10</td>
</tr>
<tr>
<td>lr_decay_steps</td>
<td>800</td>
</tr>
<tr>
<td>grad_clip</td>
<td>1</td>
</tr>
<tr>
<td>seed</td>
<td>42</td>
</tr>
<tr>
<td>adamw_beta1</td>
<td>0.9</td>
</tr>
<tr>
<td>adamw_beta2</td>
<td>0.95</td>
</tr>
<tr>
<td>adamw_weight_decay</td>
<td>0.1</td>
</tr>
<tr>
<td>use_eight_bit_optimizer</td>
<td>true</td>
</tr>
<tr>
<td>dataset</td>
<td>seangogo/processed_tldr_sft_dataset_20251029_045736</td>
</tr>
</table>

### Training and Validation Loss
![training loss](https://raw.githubusercontent.com/liyuan24/dgx_spark_summary_from_human_feedback/refs/heads/main/assets/SUMMARY_TRAINING_LOSS.png)
![validation loss](https://raw.githubusercontent.com/liyuan24/dgx_spark_summary_from_human_feedback/refs/heads/main/assets/SUMMARY_VALIDATION_LOSS.png)

### Example Responses
It is clear that TLDR from SFT model is making much more sense than the base model.

<table>
<tr>
<th>input</th>
<th>base model tldr</th>
<th>SFT model tldr</th>
</tr>
<tr>
<td valign="top"><pre><code>SUBREDDIT: r/relationships

TITLE: Me 28 F with guy I'm
dating 30 M - 1 month, Is it
okay to ask if everything is
okay or am I being pushy?

POST: I have been dating this
guy for 1 month now and
everything was great until
last Sunday when I noticed he
wasn't responding to my text
with the same flirty, cute,
enthusiastic text that he
usually would. He used to
call me baby, babe but since
last Sunday he has said none
of those words and texting
has been less and
communication has been less.

He invited me over last night
to watch a movie so I went
over and I think we had one
conversation the whole night,
it was how my day was. He
never even tried to start a
conversation after the movie
and he has never asked if he
could get high in front of me
until today which I was
totally fine with but I
thought it was weird that the
whole time we hung out before
he never smoked in front of
me. I am not sure if he's
getting comfortable or he
just doesn't care anymore?

Anyways, as he was walking me
out to my car I asked him if
everything was okay because
he has been acting different.
His only response was, "Yes,
I'm fine" and then it got
awkward and I left in my car.

Do you think I am being pushy
or too clingy asking that
question? I regret asking it
right after I asked it
because it makes me feel like
I have low self-esteem for
this relationship. Maybe I
just worry too much but it
has been eating at me.

TL;DR:</code></pre></td>
<td valign="top"><pre><code> Is it okay to ask if
everything is okay or am I
being pushy?

Title: Me 28 F with guy I'm
dating 30 M - 1 month, Is it
okay to ask if everything is
okay or am I being pushy?</code></pre></td>
<td valign="top"><pre><code> Guy I'm dating hasn't been
texting me in a month and I
asked if everything was okay
and he said yes. Am I being
pushy or too clingy asking if
everything is okay?<|endoftext|></code></pre></td>
</tr>
</table>

### Steps
#### 1. Build the Docker Container
```bash
sudo docker build --build-arg HF_TOKEN=$HF_TOKEN -t summary_from_human_feedback .
```

#### 2. Run the Docker Container
```bash
sudo sh lauch_docker.sh
cd /dgx_spark_summary_from_human_feedback
```

#### 3. Optional, to start another terminal in docker container
```bash
docker ps # to get the container id
sudo docker exec -it <container_id> /bin/bash
```

#### 4. Train the SFT model
```bash
python sft.py
```

#### 5. Generate text from the SFT model checkpoint

```bash
cd ..
python3 -m dgx_spark_summary_from_human_feedback.generation --checkpoint_path your_checkpoint_path --use_hf_checkpoint
```

## Reward Model
The dataset for reward model is [openai/summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback). But we cannot use `datasets.load_dataset` to download it because it is using a builder class which is not supported by new version of `datasets` library.

### Download the comparison dataset
I will use [summarize_from_feedback.py](https://huggingface.co/datasets/openai/summarize_from_feedback/blob/main/summarize_from_feedback.py) to download the comparison dataset. I copied it to the repo and name it to `summarize_from_feedback.py`.

Use this command to download the comparison dataset and get the number of maximum length of tokens for both chosen and rejected responses. Note the tokenizer is [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
```bash
python3 -m dgx_spark_summary_from_human_feedback.get_comparison_data_stats
```

```
max length of training data: 133
max length of validation data: 133
```

### Reward Model Training
[The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031) suggested using SFT model to initialize the reward model. But in my experiments, the training and validation performance for SFT model as the initial reward model is not good. One possible cause is the SFT model is of size 0.5B.

I then tried [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) and [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) to initialize the reward model. Qwen 2.5 1.5b is better than Qwen 2.5 0.5b in my experiments.

#### Qwen 2.5 0.5B vs Qwen 2.5 1.5B
The training and validation performance for Qwen 2.5 0.5B and Qwen 2.5 1.5B are shown in the following charts.

![qwen comparison](https://raw.githubusercontent.com/liyuan24/dgx_spark_summary_from_human_feedback/refs/heads/main/assets/qwen_comparison.png)

#### Hyperparameters for Qwen 2.5 1.5B

<table>
<tr>
<th>Parameter</th>
<th>Value</th>
</tr>
<tr>
<td>grad_clip</td>
<td>1</td>
</tr>
<tr>
<td>batch_size</td>
<td>8</td>
</tr>
<tr>
<td>adamw_beta1</td>
<td>0.9</td>
</tr>
<tr>
<td>adamw_beta2</td>
<td>0.95</td>
</tr>
<tr>
<td>warmup_ratio</td>
<td>0.03</td>
</tr>
<tr>
<td>learning_rate</td>
<td>0.00005</td>
</tr>
<tr>
<td>num_train_epochs</td>
<td>1</td>
</tr>
<tr>
<td>adamw_weight_decay</td>
<td>0.1</td>
</tr>
<tr>
<td>use_eight_bit_optimizer</td>
<td>true</td>
</tr>
<tr>
<td>gradient_accumulation_steps</td>
<td>16</td>
</tr>
</table>

### Upload the trained reward model to Hugging Face
1. Copy the `reward.py` to the saved checkpoint folder
2. Update the `config.json` file to add the `auto_map` section. It will map the `AutoConfig` and `AutoModel` to the `ScalarModelConfig` and `RewardModel` classes.
3. Test the reward model by running `python3 -m dgx_spark_summary_from_human_feedback.load_local_load_local_reward_model_and_push_to_hfreward_model --model_path your_local_checkpoint_path`
4. Upload the reward model to Hugging Face by running
```bash
python3 -m dgx_spark_summary_from_human_feedback.load_local_reward_model_and_push_to_hf --model_path /workspace/dgx_spark_summary_from_human_feedback/reward_output/checkpoint_basemodel_qwen_2.5_1.5b_final_step --push_to_hf --hf_repo_id seangogo/Qwen2.5-1.5B_reward_model_v2
```

### Reward Normalization
Following reward normalization in [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031), use the trained reward model to compute the reward for the [SFT dataset](https://huggingface.co/datasets/seangogo/processed_tldr_sft_dataset_20251029_045736) and use the average reward to normalize the reward model output.

```bash
python3 -m dgx_spark_summary_from_human_feedback.reward_normalization --model_path seangogo/Qwen2.5-1.5B_reward_model_v2 --sft_dataset seangogo/processed_tldr_sft_dataset_20251029_045736 --push_model_to_hf --hf_model_repo_id seangogo/Qwen2.5-1.5B_reward_model_v2_normalized --push_dataset_to_hf
```

### Agreement Rate with GPT5

## DPO
[DPO](https://arxiv.org/abs/2305.18290) is a method for aligning language models without reward model. It only needs a comparison dataset to fine-tune the policy model. Compared to RLHF, e.g. PPO, GRPO, DPO is much cheaper and faster to train.

Although there is no reward model in DPO, the loss function of it is similar to the reward model.

The DPO loss function is defined as:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}_{\text{PREF}}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_c | x)}{\pi^{\text{SFT}}(y_c | x)} - \beta \log \frac{\pi_\theta(y_r | x)}{\pi^{\text{SFT}}(y_r | x)} \right) \right]$$

where:
- $\pi_\theta$ is the current policy model being trained
- $\pi^{\text{SFT}}$ is the supervised fine-tuned (SFT) reference model
- $(x, y_c, y_r)$ are triplets sampled from the preference dataset $\mathcal{D}_{\text{PREF}}$
  - $x$ is the input prompt
  - $y_c$ is the chosen (preferred) response
  - $y_r$ is the rejected response
- $\beta$ is a hyperparameter that controls how confident you are about the comparison dataset quality. 
- $\sigma$ is the sigmoid function

The loss aims to maximize the log-probability of the chosen response relative to the SFT model, while minimizing the log-probability of the rejected response relative to the SFT model.

To train the DPO model, run:
```bash
python3 dpo.py --output_dir dpo_output --output_checkpoint_prefix checkpoint --wandb_run_name <wandb_run_name> --sft_model_path <local_or_hf_path_to_sft_model>
```

The trained model was pushed to [Hugging Face](https://huggingface.co/seangogo/dpo_summary_from_human_feedback).

**Training curve of DPO:**

![DPO training curve](https://github.com/liyuan24/dgx_spark_summary_from_human_feedback/blob/main/assets/dpo.png)

**Hyperparameters for DPO:**

<table>
<tr>
<th>Parameter</th>
<th>Value</th>
</tr>
<tr>
<td>grad_clip</td>
<td>1</td>
</tr>
<tr>
<td>batch_size</td>
<td>8</td>
</tr>
<tr>
<td>adamw_beta1</td>
<td>0.9</td>
</tr>
<tr>
<td>adamw_beta2</td>
<td>0.95</td>
</tr>
<tr>
<td>warmup_ratio</td>
<td>0.03</td>
</tr>
<tr>
<td>learning_rate</td>
<td>0.00005</td>
</tr>
<tr>
<td>num_train_epochs</td>
<td>1</td>
</tr>
<tr>
<td>adamw_weight_decay</td>
<td>0.1</td>
</tr>
<tr>
<td>use_eight_bit_optimizer</td>
<td>true</td>
</tr>
<tr>
<td>gradient_accumulation_steps</td>
<td>16</td>
</tr>
<tr>
<td>label_smoothing</td>
<td>0.1</td>
</tr>
<tr>
<td>beta</td>
<td>0.1</td>
</tr>
</table>

### Example Responses

<table>
<tr>
<th>input</th>
<th>base model tldr</th>
<th>SFT model tldr</th>
<th>DPO model tldr</th>
</tr>
<tr>
<td valign="top"><pre><code>SUBREDDIT: r/relationships

TITLE: Me 28 F with guy I'm
dating 30 M - 1 month, Is it
okay to ask if everything is
okay or am I being pushy?

POST: I have been dating this
guy for 1 month now and
everything was great until
last Sunday when I noticed he
wasn't responding to my text
with the same flirty, cute,
enthusiastic text that he
usually would. He used to
call me baby, babe but since
last Sunday he has said none
of those words and texting
has been less and
communication has been less.

He invited me over last night
to watch a movie so I went
over and I think we had one
conversation the whole night,
it was how my day was. He
never even tried to start a
conversation after the movie
and he has never asked if he
could get high in front of me
until today which I was
totally fine with but I
thought it was weird that the
whole time we hung out before
he never smoked in front of
me. I am not sure if he's
getting comfortable or he
just doesn't care anymore?

Anyways, as he was walking me
out to my car I asked him if
everything was okay because
he has been acting different.
His only response was, "Yes,
I'm fine" and then it got
awkward and I left in my car.

Do you think I am being pushy
or too clingy asking that
question? I regret asking it
right after I asked it
because it makes me feel like
I have low self-esteem for
this relationship. Maybe I
just worry too much but it
has been eating at me.

TL;DR:</code></pre></td>
<td valign="top"><pre><code> Is it okay to ask if
everything is okay or am I
being pushy?

Title: Me 28 F with guy I'm
dating 30 M - 1 month, Is it
okay to ask if everything is
okay or am I being pushy?</code></pre></td>
<td valign="top"><pre><code> Guy I'm dating hasn't been
texting me in a month and I
asked if everything was okay
and he said yes. Am I being
pushy or too clingy asking if
everything is okay?<|endoftext|></code></pre></td>
<td valign="top"><pre><code> I have been dating this guy 
for 1 month and he hasn't responded to my text with the 
same flirty, cute, enthusiastic text that he usually would. 
I was really happy when we had our movie and I was really 
excited about it. I was wondering if I should ask him if 
everything is okay or am I being pushy.<|endoftext|></code></pre></td>
</tr>
</table>

## GRPO
The trained model was pushed to [Hugging Face](https://huggingface.co/seangogo/summary_from_human_feedback_grpo_100).

### Rollout Generation
A few notes on generation configuration
```
self.generation_config = GenerationConfig(
    max_new_tokens=config.response_length,
    temperature=config.temperature + 1e-7,
    top_k=0,
    top_p=1.0,
    do_sample=True,
    eos_token_id=self.tokenizer.eos_token_id,
    pad_token_id=self.tokenizer.pad_token_id,
)
```
1. We need to explicitly set the `eos_token_id` in the generation configuration. By default it is `None` and the generation will continue until the `max_new_tokens` is reached.
2. In the [PPO code](https://github.com/vwxyzjn/summarize_from_feedback_details/blob/main/summarize_from_feedback_details/ppo.py#L542) of [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031), `min_new_tokens` and `max_new_tokens` are set to the same value as `response_length`. In my experiments, this will not work because the generation will continue until the `max_new_tokens` is reached and the EOS token will not be generated. And we need to get the reward from the EOS token. So I removed the `min_new_tokens` setting.
3. The input prompt is left padded with pad token so that prompt tokens are with the same length.
4. Note that when generating the responses for a batch of prompts, with the config above, when one sequence reaches the EOS token, pad token will be added to pad the response to the same length as the longest response in the batch.

### Optimization Process

Multiple epochs of mini-batches gradient updates. See more details in [PPO paper](https://arxiv.org/pdf/1707.06347).
1. The whole rollout data will be split into multiple mini-batches.
2. In case the GPU memory cannot fit one mini-batch, we will split the mini-batch into multiple micro-batches and do gradient accumulation for each micro-batch.
3. After all micro-batches are processed, the weights of the policy model will be updated.

So the total number of weight update steps is 

$$\frac{num\_epochs * total\_samples * num\_responses\_per\_prompt}{batch\_size * mini\_batch\_size}$$

### Vanilla PPO loss vs [Dual Clip loss](https://arxiv.org/pdf/1912.09729)
![PPO loss](https://raw.githubusercontent.com/liyuan24/dgx_spark_summary_from_human_feedback/refs/heads/main/assets/ppo_loss.png)
![Dual Clip loss](https://raw.githubusercontent.com/liyuan24/dgx_spark_summary_from_human_feedback/refs/heads/main/assets/dual_clip_loss.png)

When advantage is positive, the magnitude of the loss is bounded in vanilla PPO loss. But when advantage is negative, the magnitude of the loss is unbounded in vanilla PPO loss. Dual Clip loss is proposed to tackle this. When advantage is negative and ratio is greater than `c`, the loss will be clipped to `c * advantage` to stabilize the training.

### Training Curve

Run the following command to train the GRPO model:
```bash
sh run_grpo.sh
```

![GRPO training curve](https://raw.githubusercontent.com/liyuan24/dgx_spark_summary_from_human_feedback/refs/heads/main/assets/grpo_reward_climbing.png)

### Hyperparameters

| Category | Hyperparameter | Value |
|----------|---------------|-------|
| **Training** | `num_train_epochs` | 1 |
| | `learning_rate` | 5e-5 |
| | `warmup_ratio` | 0.03 |
| | `grad_clip` | 1.0 |
| **Batch Sizes** | `batch_size` | 16 |
| | `mini_batch_size` | 64 |
| | `micro_batch_size` | 8 |
| | `num_responses_per_group` | 8 |
| **Optimizer** | `use_adamw_fused` | true |
| | `adamw_beta1` | 0.9 |
| | `adamw_beta2` | 0.95 |
| | `adamw_weight_decay` | 0.1 |
| | `use_eight_bit_optimizer` | false |
| **GRPO** | `update_per_rollout` | 4 |
| | `clip_ratio` | 0.2 |
| | `kl_coeff` | 0.05 |
| | `kl_penalty_mode` | "k3" |
| | `loss_agg_mode` | "seq-mean-token-mean" |
| | `normalize_adv_by_std_of_group` | true |
| | `no_eos_penalty` | -1.0 |
| **Generation** | `response_length` | 63 |
| | `temperature` | 0.7 |

### Example Responses

### Example Responses

<table>
<tr>
<th>input</th>
<th>base model tldr</th>
<th>SFT tldr</th>
<th>DPO tldr</th>
<th>GRPO tldr</th>
</tr>
<tr>
<td valign="top"><pre><code>SUBREDDIT: r/relationships

TITLE: Me 28 F with guy I'm
dating 30 M - 1 month, Is it
okay to ask if everything is
okay or am I being pushy?

POST: I have been dating this
guy for 1 month now and
everything was great until
last Sunday when I noticed he
wasn't responding to my text
with the same flirty, cute,
enthusiastic text that he
usually would. He used to
call me baby, babe but since
last Sunday he has said none
of those words and texting
has been less and
communication has been less.

He invited me over last night
to watch a movie so I went
over and I think we had one
conversation the whole night,
it was how my day was. He
never even tried to start a
conversation after the movie
and he has never asked if he
could get high in front of me
until today which I was
totally fine with but I
thought it was weird that the
whole time we hung out before
he never smoked in front of
me. I am not sure if he's
getting comfortable or he
just doesn't care anymore?

Anyways, as he was walking me
out to my car I asked him if
everything was okay because
he has been acting different.
His only response was, "Yes,
I'm fine" and then it got
awkward and I left in my car.

Do you think I am being pushy
or too clingy asking that
question? I regret asking it
right after I asked it
because it makes me feel like
I have low self-esteem for
this relationship. Maybe I
just worry too much but it
has been eating at me.

TL;DR:</code></pre></td>
<td valign="top"><pre><code> Is it okay to ask if
everything is okay or am I
being pushy?

Title: Me 28 F with guy I'm
dating 30 M - 1 month, Is it
okay to ask if everything is
okay or am I being pushy?</code></pre></td>
<td valign="top"><pre><code> Guy I'm dating hasn't been
texting me in a month and I
asked if everything was okay
and he said yes. Am I being
pushy or too clingy asking if
everything is okay?<|endoftext|></code></pre></td>
<td valign="top"><pre><code> I have been dating this guy 
for 1 month and he hasn't responded to my text with the 
same flirty, cute, enthusiastic text that he usually would. 
I was really happy when we had our movie and I was really 
excited about it. I was wondering if I should ask him if 
everything is okay or am I being pushy.<|endoftext|></code></pre></td>
<td valign="top"><pre><code> Dating guy is acting different and never responds to anything I ask. Is it okay to ask if everything is okay or am I being pushy? Worried about low self-esteem for this relationship.<|endoftext|></code></pre></td>
</tr>
</table>