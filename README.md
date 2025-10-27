# dgx_spark_summary_from_human_feedback

This repo contains the code for reproducing [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) in one DGX Spark. [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031) has already done this in an 8-H100 cluster. We will refer to this work for many details.

# Steps

1. Preprocessing the data
2. Supervised Fine-tuning
3. Reward Model Training
4. Reinforcement learning from human feedback