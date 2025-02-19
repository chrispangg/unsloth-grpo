import os
from typing import List

from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)  # needed for GRPO

import re  # noqa: E402
from dataclasses import (
    dataclass,  # noqa: E402
    field,  # noqa: E402
)

import huggingface_hub  # noqa: E402
import hydra  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from unsloth import is_bfloat16_supported  # noqa: E402
from vllm import SamplingParams  # noqa: E402

import wandb  # noqa: E402

max_seq_length = 2048  # Can increase for longer reasoning traces
lora_rank = 64  # Larger rank = smarter, but slower


@dataclass
class LoraConfig:
    rank: int = 64
    target_modules: List = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407


@dataclass
class ModelConfig:
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    fast_inference: bool = True
    lora: LoraConfig = field(default_factory=lambda: LoraConfig())

    gpu_memory_utilization: float = 0.5


def prepare_model(cfg: DictConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,  # False for LoRA 16bit
        fast_inference=cfg.model.fast_inference,  # Enable vLLM fast inference
        max_lora_rank=cfg.model.lora.rank,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.model.lora.rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=cfg.model.lora.rank,
        use_gradient_checkpointing=cfg.model.lora.use_gradient_checkpointing,  # Enable long context finetuning
        random_state=cfg.model.lora.random_state,
    )
    return model, tokenizer


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<|begin_of_thought|>
...
<|end_of_thought|>
<|begin_of_solution|>
...
\\boxed{answer}
<|end_of_solution|>
"""
XML_COT_FORMAT = """\
<|begin_of_thought|>
{reasoning}
<|end_of_thought|>
<|begin_of_solution|>
\\boxed{{{answer}}}
<|end_of_solution|>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<|begin_of_solution|>")[-1]
    answer = answer.split("<|end_of_solution|>")[0]
    answer = answer.strip()
    if "\\boxed{" in answer:
        answer = answer.split("\\boxed{")[1].split("}")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


dataset = get_gsm8k_questions()


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# def strict_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<\|begin_of_thought\|>\n.*?\n<\|end_of_thought\|>\n<\|begin_of_solution\|>\n.*?\n<\|end_of_solution\|>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>\s*<\|begin_of_solution\|>.*?<\|end_of_solution\|>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<|begin_of_thought|>\n") == 1:
        count += 0.125
    if text.count("\n<|end_of_thought|>\n") == 1:
        count += 0.125
    if text.count("\n<|begin_of_solution|>\n") == 1:
        count += 0.125
        count -= len(text.split("\n<|end_of_solution|>\n")[-1]) * 0.001
    if text.count("\n<|end_of_solution|>") == 1:
        count += 0.125
        count -= (len(text.split("\n<|end_of_solution|>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def strawberry_example(tokenizer, model):
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "How many r's are in strawberry?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    output = (
        model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )[0]
        .outputs[0]
        .text
    )

    print(output)


# output


def strawberry_example_lora(tokenizer, model):
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How many r's are in strawberry?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    output = (
        model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora("grpo_saved_lora"),
        )[0]
        .outputs[0]
        .text
    )

    print(output)


def save(cfg, model, tokenizer):
    if cfg.saving.save_gguf.enabled:
        for quant_method in cfg.saving.save_gguf.quantization_methods:
            model.save_pretrained_gguf(
                cfg.saving.model_dir, tokenizer, quantization_method=quant_method
            )
            if cfg.saving.token:  # Only push if token is provided
                model.push_to_hub_gguf(
                    cfg.saving.hub_model_id,
                    tokenizer,
                    quantization_method=quant_method,
                    token=os.getenv("HF_TOKEN"),
                )

    if cfg.saving.save_merged.enabled:
        for save_method in cfg.saving.save_merged.methods:
            model.save_pretrained_merged(
                cfg.saving.model_dir, tokenizer, save_method=save_method
            )
            if cfg.saving.token:  # Only push if token is provided
                model.push_to_hub_merged(
                    cfg.saving.hub_model_id,
                    tokenizer,
                    save_method=save_method,
                    token=cfg.saving.token,
                )


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    model, tokenizer = prepare_model(cfg)
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=cfg.training.learning_rate,
        adam_beta1=cfg.training.adam_beta1,
        adam_beta2=cfg.training.adam_beta2,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=cfg.training.num_generations,
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=cfg.training.max_completion_length,
        max_steps=cfg.training.max_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=2,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=cfg.training.report_to,
        output_dir=cfg.training.output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            # strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    strawberry_example(tokenizer=tokenizer, model=model)
    strawberry_example_lora(tokenizer=tokenizer, model=model)
    trainer.save_model(
        "/workspace/grpo_demo/chriswhpang/Llama-3.2-1B-Instruct-OpenThought-SFT-GRPO-GGUF"
    )

    if cfg.saving is not None:
        save(cfg, model, tokenizer)


if __name__ == "__main__":
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    huggingface_hub.login(token=os.getenv("HF_TOKEN"))

    main()
