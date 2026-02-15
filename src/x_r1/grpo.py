import json
import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import pandas as pd
import torch
import transformers
from configs import GRPOConfig
from datasets import Dataset, DatasetDict, load_dataset
from rewards import (accuracy_reward, format_reward,
                     get_molecular_structure_reward,
                     get_smile_optimization_reward,
                     get_smile_similarity_reward, get_smile_validity_reward,
                     len_reward)
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from utils.callbacks import get_callbacks
from x_grpo_trainer import XGRPOTrainer

from math_reward import compute_score

logger = logging.getLogger(__name__)

# System prompt for all tasks
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project



@dataclass
class GRPOScriptArguments(ScriptArguments):
    variant: str = field(
        default="default",
        metadata={"help": "Entry variant for consolidated `grpo.py` (default|math|mumo|pure|noisy_demo|random_mask)"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    
    # Length reward parameters
    length_min_tokens: int = field(
        default=100,
        metadata={"help": "Minimum number of tokens required to start receiving a reward"},
    )
    length_target_tokens: int = field(
        default=500,
        metadata={"help": "Target number of tokens that will receive the maximum reward"},
    )
    length_max_reward: float = field(
        default=0.8,
        metadata={"help": "Maximum reward value for reaching the target length"},
    )
    length_reward_curve: str = field(
        default="linear",
        metadata={"help": "How the reward scales between min and target lengths (linear, quadratic, logarithmic)"},
    )
    
    # Add data scale and subtask selection parameters
    data_scale: str = field(
        default="light",
        metadata={"help": "Data scale to use. Options: small, medium, large, xlarge, light"},
    )
    
    subtask_selection: list[str] = field(
        default_factory=lambda: ["LogP", "MR", "QED"],
        metadata={"help": "Subtask to use for training. Options: all, AtomNum, BondNum, FunctionalGroup, AddComponent, SubComponent, DelComponent, LogP, MR, QED"},
    )



def main(script_args, training_args, model_args, variant: str = "default"):
    variant = (variant or "default").strip().lower()
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Variant-specific system prompt (math needs boxed answers)
    system_prompt = SYSTEM_PROMPT
    if variant == "math":
        system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> \\\\boxed{{answer here}} </answer>"
        )

    # Dataset selection
    if variant == "math":
        # Match grpo_math.py: load train/test splits directly from HF dataset name
        train_dataset = load_dataset(script_args.dataset_name, split="train")
        eval_dataset = load_dataset(script_args.dataset_name, split="test")
        dataset = None
    elif variant == "mumo":
        # Match grpo_mumo.py: support structural opt / property opt / multi-prop datasets
        if script_args.subtask_selection == ["AddComponent", "SubComponent", "DelComponent"]:
            data = json.load(open("data/structural_opt_light.json", "r"))
            df = pd.DataFrame(data)
            ds = Dataset.from_pandas(df)
            dataset = DatasetDict({"train": ds})
            dataset = dataset.rename_column("instruction", "problem")
            dataset = dataset.rename_column("output", "solution")
        elif script_args.subtask_selection == ["LogP", "MR", "QED"]:
            dataset_path = f"data/OpenMolIns/{script_args.data_scale}/train.csv"
            df = pd.read_csv(dataset_path)
            if script_args.subtask_selection != "all":
                if isinstance(script_args.subtask_selection, list):
                    df = df[df["SubTask"].isin(script_args.subtask_selection)]
                else:
                    df = df[df["SubTask"] == script_args.subtask_selection]
            train_dataset = Dataset.from_pandas(df)
            dataset = DatasetDict({"train": train_dataset})
            dataset = dataset.rename_column("Instruction", "problem")
            dataset = dataset.rename_column("molecule", "solution")
        elif script_args.subtask_selection in ["bbbp+drd2+plogp", "bbbp+drd2+qed", "bbbp+plogp+qed"]:
            data = json.load(
                open(f"data/TRAIN_multi_prop/IND_sft_train_data_{script_args.subtask_selection}.json", "r")
            )
            df = pd.DataFrame(data)
            ds = Dataset.from_pandas(df)
            dataset = DatasetDict({"train": ds})
            dataset = dataset.rename_column("instruction", "problem")
            dataset = dataset.rename_column("output", "solution")
        else:
            dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        train_dataset = None
        eval_dataset = None
    else:
        # Default (existing grpo.py behavior)
        # OpenMolIns Property Optimization
        if script_args.subtask_selection == ["LogP", "MR", "QED"]:
            dataset_path = f"data/OpenMolIns/{script_args.data_scale}/train.csv"

            if dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)

                # Filter by subtask if specified
                if script_args.subtask_selection != "all":
                    if isinstance(script_args.subtask_selection, list):
                        df = df[df["SubTask"].isin(script_args.subtask_selection)]
                        logger.info(f"Filtered dataset to subtasks: {script_args.subtask_selection}")
                    else:
                        df = df[df["SubTask"] == script_args.subtask_selection]
                        logger.info(f"Filtered dataset to subtask: {script_args.subtask_selection}")
                    logger.info(f"Filtered dataset size: {len(df)} examples")

                train_dataset = Dataset.from_pandas(df)
                dataset = DatasetDict({"train": train_dataset})
                dataset = dataset.rename_column("Instruction", "problem")
                dataset = dataset.rename_column("molecule", "solution")

                logger.info(f"Loaded OpenMolIns dataset with {len(dataset['train'])} examples")
                logger.info(f"Sample example: {dataset['train'][0]}")
        else:
            dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

        train_dataset = None
        eval_dataset = None

    # align the dataset (dataset-dict only)
    if dataset is not None and script_args.dataset_name == "FreedomIntelligence/medical-o1-verifiable-problem":
        dataset = dataset.rename_columns(
            {"Open-ended Verifiable Question": "problem", "Ground-True Answer": "solution"}
        )

    # Harmonize common column names across datasets (dataset-dict only)
    if dataset is not None:
        try:
            train_split = script_args.dataset_train_split if hasattr(script_args, "dataset_train_split") else "train"
            if train_split in dataset:
                column_names = set(dataset[train_split].column_names)
                if "Instruction" in column_names and "problem" not in column_names:
                    dataset = dataset.rename_column("Instruction", "problem")
                if "molecule" in column_names and "solution" not in column_names:
                    dataset = dataset.rename_column("molecule", "solution")
        except Exception:
            pass

    # Get reward functions
    if variant == "mumo":
        from rewards_mumo import (  # local import to keep default path untouched
            accuracy_reward as mumo_accuracy_reward,
            format_reward as mumo_format_reward,
            get_cosine_scaled_reward as mumo_get_cosine_scaled_reward,
            get_multi_prop_optimization_reward,
            get_repetition_penalty_reward as mumo_get_repetition_penalty_reward,
            get_smile_optimization_reward as mumo_get_smile_optimization_reward,
            get_smile_similarity_reward as mumo_get_smile_similarity_reward,
            get_smile_validity_reward as mumo_get_smile_validity_reward,
            len_reward as mumo_len_reward,
            reasoning_steps_reward as mumo_reasoning_steps_reward,
        )

        REWARD_FUNCS_REGISTRY = {
            "accuracy": lambda prompts, completions, solutions=None, **kwargs: mumo_accuracy_reward(
                completions=completions, solution=solutions
            )
            if solutions is not None
            else [0.0] * len(completions),
            "format": lambda prompts, completions, **kwargs: mumo_format_reward(completions=completions),
            "reasoning_steps": lambda prompts, completions, **kwargs: mumo_reasoning_steps_reward(completions=completions),
            "cosine": lambda prompts, completions, solutions=None, **kwargs: mumo_get_cosine_scaled_reward(
                min_value_wrong=script_args.cosine_min_value_wrong,
                max_value_wrong=script_args.cosine_max_value_wrong,
                min_value_correct=script_args.cosine_min_value_correct,
                max_value_correct=script_args.cosine_max_value_correct,
                max_len=script_args.cosine_max_len,
            )(completions=completions, solution=solutions if solutions is not None else [""] * len(completions)),
            "repetition_penalty": lambda prompts, completions, **kwargs: mumo_get_repetition_penalty_reward(
                ngram_size=script_args.repetition_n_grams, max_penalty=script_args.repetition_max_penalty
            )(completions=completions),
            "length": lambda prompts, completions, solutions=None, **kwargs: mumo_len_reward(
                prompts=prompts,
                completions=completions,
                solutions=solutions,
                min_tokens=script_args.length_min_tokens if hasattr(script_args, "length_min_tokens") else 100,
                target_tokens=script_args.length_target_tokens if hasattr(script_args, "length_target_tokens") else 500,
                max_reward=script_args.length_max_reward if hasattr(script_args, "length_max_reward") else 0.8,
                reward_curve=script_args.length_reward_curve if hasattr(script_args, "length_reward_curve") else "linear",
            ),
            "smile_validity": lambda prompts, completions, **kwargs: mumo_get_smile_validity_reward(
                extract_pattern=r"<answer>(.*?)</answer>", validity_weight=1.0
            )(completions=completions),
            "smile_similarity": lambda prompts, completions, **kwargs: mumo_get_smile_similarity_reward(
                extract_pattern=r"<answer>(.*?)</answer>"
            )(completions=completions),
            "smile_optimization": lambda prompts, completions, **kwargs: mumo_get_smile_optimization_reward(
                property_name="logP",
                target_direction=None,
                reference_smiles=None,
                extract_pattern=r"<answer>(.*?)</answer>",
            )(completions=completions, prompts=prompts),
            "structure_optimization": lambda prompts, completions, **kwargs: mumo_get_molecular_structure_reward(
                extract_pattern=r"<answer>(.*?)</answer>"
            )(completions=completions, prompts=prompts, **kwargs),
            "multi_prop_optimization": lambda prompts, completions, **kwargs: get_multi_prop_optimization_reward(
                task=script_args.subtask_selection, extract_pattern=r"<answer>(.*?)</answer>"
            )(completions=completions, prompts=prompts, **kwargs),
        }
    else:
        REWARD_FUNCS_REGISTRY = {
            "accuracy": lambda prompts, completions, solutions=None, **kwargs: accuracy_reward(
                completions=completions, solution=solutions
            )
            if solutions is not None
            else [0.0] * len(completions),
            "math_accuracy": lambda prompts, completions, solutions=None, **kwargs: compute_score(
                completions=completions, solution=solutions
            )
            if solutions is not None
            else [0.0] * len(completions),
            "format": lambda prompts, completions, **kwargs: format_reward(completions=completions),
            "length": lambda prompts, completions, solutions=None, **kwargs: len_reward(
                prompts=prompts,
                completions=completions,
                solutions=solutions,
                min_tokens=script_args.length_min_tokens if hasattr(script_args, "length_min_tokens") else 100,
                target_tokens=script_args.length_target_tokens if hasattr(script_args, "length_target_tokens") else 500,
                max_reward=script_args.length_max_reward if hasattr(script_args, "length_max_reward") else 0.8,
                reward_curve=script_args.length_reward_curve if hasattr(script_args, "length_reward_curve") else "linear",
            ),
            "smile_validity": lambda prompts, completions, **kwargs: get_smile_validity_reward(
                extract_pattern=r"<answer>(.*?)</answer>", validity_weight=1.0
            )(completions=completions),
            "smile_similarity": lambda prompts, completions, **kwargs: get_smile_similarity_reward(
                extract_pattern=r"<answer>(.*?)</answer>"
            )(completions=completions),
            "smile_optimization": lambda prompts, completions, **kwargs: get_smile_optimization_reward(
                property_name="logP",
                target_direction=None,
                reference_smiles=None,
                extract_pattern=r"<answer>(.*?)</answer>",
            )(completions=completions, prompts=prompts),
            "structure_optimization": lambda prompts, completions, **kwargs: get_molecular_structure_reward(
                extract_pattern=r"<answer>(.*?)</answer>"
            )(completions=completions, prompts=prompts),
        }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    reward_func_names = script_args.reward_funcs

    # Convert reward weights dict to list matching reward_funcs order
    if hasattr(training_args, 'reward_weights') and isinstance(training_args.reward_weights, dict):
        reward_weights = [training_args.reward_weights[func] for func in script_args.reward_funcs]
        training_args.reward_weights = reward_weights

    # Format into conversation
    def make_conversation(example):
        # Use the unified system prompt for all tasks
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["problem"]},
            ],
            "solution": example["solution"] if "solution" in example else None,
        }

    if variant == "math":
        train_dataset = train_dataset.map(make_conversation)
        eval_dataset = eval_dataset.map(make_conversation)
    else:
        dataset = dataset.map(make_conversation)

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    training_args.gradient_checkpointing = False
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    training_args.model_init_kwargs = model_kwargs


    #############################
    # Initialize the XGRPO trainer
    #############################
    trainer_cls = XGRPOTrainer

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        # model = model,
        reward_funcs=reward_funcs,
        reward_func_names=reward_func_names,  
        args=training_args,
        variant=variant,
        train_dataset=train_dataset if variant == "math" else dataset[script_args.dataset_train_split],
        eval_dataset=(eval_dataset if variant == "math" else dataset[script_args.dataset_test_split])
        if training_args.eval_strategy != "no"
        else None,
        peft_config=get_peft_config(model_args), # LoRA parameter
        callbacks=get_callbacks(training_args, model_args),
    )

    print(trainer)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset) if variant == "math" else len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["X-R1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

def run(variant: str = "default"):
    """
    Minimal consolidation point for GRPO variants.
    - default: existing behavior of `grpo.py`
    - math: behavior of `grpo_math.py` (boxed prompt, math_accuracy default, x_grpo_trainer_math)
    - mumo: behavior of `grpo_mumo.py` (MuMo dataset + rewards, x_grpo_trainer_mumo)
    - pure: behavior of `grpo_pure.py` (x_grpo_trainer_pure)
    """
    variant = (variant or "default").strip().lower()

    # Variant-specific parser choices (keep the CLI stable; wrappers just call run(variant=...))
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if variant == "default":
        variant = getattr(script_args, "variant", "default")

    # Variant-specific defaults / wiring (keep changes minimal and opt-in)
    if variant == "math":
        # Match grpo_math.py defaults unless user explicitly changed reward_funcs
        if script_args.reward_funcs == ["accuracy", "format"]:
            script_args.reward_funcs = ["math_accuracy", "format"]
    main(script_args, training_args, model_args, variant=variant)


if __name__ == "__main__":
    run("default")
