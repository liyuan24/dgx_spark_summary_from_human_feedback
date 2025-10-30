# dgx_spark_summary_from_human_feedback

This repo contains the code for reproducing [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) in one DGX Spark. [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031) has already done this in an 8-H100 cluster. We will refer to this work for many details.

# Steps

1. Preprocessing the data
2. Supervised Fine-tuning
3. Reward Model Training
4. Reinforcement learning from human feedback

# Preprocessing the TLDR dataset
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

## Supervised Fine-tuning

<table>
<tr>
<th>input</th>
<th>base model tldr</th>
<th>SFT model tldr</th>
</tr>
<tr>
<td valign="top"><pre><code>SUBREDDIT: r/tifu

TITLE: TIFU by
forgetting my
lube in the
shower

POST: So I'm
sitting in the
living room
with my then
GF not long
ago and my
roommate (Carl
with the
slicked back
hair) comes
in from the
bathroom to
ask me where
the little
black bottle
in the shower
went.
Confused, I
looked back at
him and told
him I didn't
know what he
was talking
about. This is
about what
happened next:

\>Carl: Are you
sure it wasn't
yours? It
appeared in the
bathroom the
other day and
then today it's
gone.

\>Me: Still not
sure. Why do
you ask?

\>Carl:
Aww...damn.
Whatever that
shit was it
was awesome. It
was like this
conditioner or
something that
kept my hair
super slick
alll day long.
It was crazy,
not even water
could get it
out. It worked
so much better
than the hair
stuff I use
right now.
Maybe Jenny
(his GF) knows
where it went.
Are you sure
you don't know?

At this point
I have figured
out that the
little black
bottle to which
he kept
referring was
my bottle of
lube. I glanced
over at my GF
and all the
color had
drained from
her face and
she was trying
not to give it
away that she
knew. With the
straightest face
I could muster
I told him that
I still had no
idea. I haven't
told him to
this day.

TL;DR:</code></pre></td>
<td valign="top"><pre><code>I'm not sure if
I should tell
my GF that I
forgot to put
my lube in the
shower. I'm not
sure if I
should tell my
GF that I
forgot to put
my lube in the
shower. I'm not
sure if I
should tell my
GF that I
forgot to put
my lube in</code></pre></td>
<td valign="top"></td>
</tr>
</table>

### 1. Build the Docker Container
```bash
sudo docker build --build-arg HF_TOKEN=$HF_TOKEN -t summary_from_human_feedback .
```

### 2. Run the Docker Container
```bash
sudo sh lauch_docker.sh
cd /summary_from_human_feedback
```