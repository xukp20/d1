<div  align="center">
    <h1>d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning</h1>
        <p>A two-stage recipe with SFT followed by a novel variant of GRPO, <i>diffu</i>-GRPO, to convert pre-trained dLLMs into strong reasoning models.</p>
</div>

****************************************************************

**Updates:**

* 04-11-2025: We released [our paper]() announced via [this tweet](). Additionally, the SFT code was open-sourced.

****************************************************************

## d1

### SFT

We open-source our code to perform completion-only masked SFT for dLLMs. We implement the algorithm proposed in [LLaDA](https://github.com/ML-GSAI/LLaDA), and also provide it below for completeness.

![SFT Algorithm](media/algorithm_sft.png)

The API is similar to 🤗 Transformers; `dLLMTrainer` inherits from `Trainer` and modifies the loss function, and `dLLMDataCollator` inherits from `DefaultDataCollator` which applies the forward noising process to the batch. Additionally, we provide a torch dataset, `dLLMSFTDataset`.

To preprocess and tokenize your dataset, you will need to modify `preprocess_dataset`. Presently, it works with the s1K dataset.

SFT results can be reproduced with the command,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file ddp_config.yaml --main_process_port 29500 --num_processes 4 sft_train.py
```

### _diffu_-GRPO
Code coming soon!

### Eval
Code coming soon!

## Citation
```
@inproceedings {zhaoandgupta2023d1,
    title={d1: Scaling Reasoning in Diffusion Large Language Models via Test-Time Inference},
    author={Zhao, Siyan and Gupta, Devaansh and Zheng, Qinqing and Grover, Aditya},
    booktitle={preprint},
    year={2025}
}
```