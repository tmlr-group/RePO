import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.utils.data
from accelerate.utils import (broadcast_object_list, gather, gather_object,
                              is_peft_model, set_seed)
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from rdkit import Chem
from torch import nn
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          GenerationConfig, PreTrainedModel,
                          PreTrainedTokenizerBase, Trainer, TrainerCallback,
                          is_wandb_available)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from trl.data_utils import (apply_chat_template, is_conversational,
                            maybe_apply_chat_template)
from trl.import_utils import is_vllm_available
from trl.models import (create_reference_model, prepare_deepspeed,
                        unwrap_model_for_generation)
from trl.trainer import GRPOTrainer
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import pad, selective_log_softmax

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


def replace_smile(text, solution):
    
    answer_pattern = r"(<answer>)(.*?)(</answer>)"
    match = re.search(answer_pattern, text, re.DOTALL)
    if match:
        smile = match.group(2).strip()
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                return re.sub(answer_pattern, rf"\1{solution}\3", text, count=1, flags=re.DOTALL)
        except:
            pass

    smile_tag_pattern = r"(<smile>)(.*?)(</smile>)"
    match = re.search(smile_tag_pattern, text, re.DOTALL)
    if match:
        smile = match.group(2).strip()
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                return re.sub(smile_tag_pattern, rf"\1{solution}\3", text, count=1, flags=re.DOTALL)
        except:
            pass

    smiles_prefix_pattern = r"(SMILES?:\s*)([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)"
    match = re.search(smiles_prefix_pattern, text, re.IGNORECASE)
    if match:
        smile = match.group(2).strip()
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                return re.sub(smiles_prefix_pattern, rf"\1{solution}", text, count=1, flags=re.IGNORECASE)
        except:
            pass

    common_words = {"think", "answer", "molecule", "modify", "optimize", "increase", "decrease", 
                    "value", "please", "smiles", "structure", "similar", "drug-like", "maintaining"}
        
    smiles_pattern = r'([A-Za-z0-9@\[\]\(\)\\.=#\-\+]{5,})'
    matches = re.finditer(smiles_pattern, text)
        
    for match in matches:
        potential_smile = match.group(1)
        if potential_smile.lower() in common_words:
            continue
        
        if not re.search(r'[CNOPS]|Cl|Br|Fe|Si|Al|Ca|Mg|Na|Li|He|Ne|Ar|Kr|Xe|Rn', potential_smile):
            continue

        try:
            mol = Chem.MolFromSmiles(potential_smile)
            if mol is not None and Chem.MolToSmiles(mol) == potential_smile:
                return re.sub(re.escape(potential_smile), solution, text, count=1)
        except:
            continue

    return text # + f"<answer>{solution}</answer>"

def replace_solution(text: str, solution: str) -> str:
    """Math variant: replace the last boxed segment with the gold `solution`."""
    try:
        from math_reward import last_boxed_only_string
    except Exception:
        return text

    boxed_segment = last_boxed_only_string(text)
    if boxed_segment is None:
        return text

    if boxed_segment.startswith("\\boxed "):
        replacement = "\\boxed " + solution
    elif boxed_segment.startswith("\\boxed{"):
        replacement = "\\boxed{" + solution + "}"
    elif boxed_segment.startswith("\\fbox{"):
        replacement = "\\fbox{" + solution + "}"
    else:
        replacement = solution

    return text.replace(boxed_segment, replacement, 1)


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

@contextmanager
def _disable_peft_adapters(model):
    """
    PEFT changed the API name across versions (`disable_adapter` vs `disable_adapters`).
    Use whichever exists; no-op if neither exists.
    """
    fn = getattr(model, "disable_adapter", None)
    if fn is None:
        fn = getattr(model, "disable_adapters", None)
    if fn is None:
        yield
        return
    with fn():
        yield


class XGRPOTrainer(GRPOTrainer):
    # base trl GRPO_trainer


    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        reward_func_names: Optional[list[str]] = None,
        variant: str = "default",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.variant = (variant or "default").strip().lower()
        # random-mask knobs (used when variant == "random_mask")
        self.thinking_mask_keep_prob = float(getattr(args, "thinking_mask_keep_prob", 1.0))
        self.keep_final_answer = bool(getattr(args, "keep_final_answer", True))

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        model.config.pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
            if reward_func_names is not None:
                reward_func_names = [reward_func_names]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        
        self.reward_func_names = reward_func_names

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        # Multi-step
        # self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        # self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        Trainer.__init__(self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    # n=args.num_generations,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _move_model_to_vllm(self):
        # https://github.com/huggingface/trl/issues/2840#issuecomment-2662747485
        for param in self.model.parameters():
            param.ds_active_sub_modules.clear()
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # unwrapped_model.unmerge_adapter()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = Trainer._prepare_inputs(self, inputs = prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm() 
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                for output in outputs:
                    print('-'*100)
                    print('\n\n\n')
                    prompt = output.prompt
                    for output_t in  output.outputs:
                        # print(completion_ids)
                        print('='*100)
                        generated_text = output_t.text
                        print("[USER]: ", prompt )
                        print("\n[ASSISTANT]:", generated_text)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]


            prompt_string = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
            output_string = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            for prompt, completion in zip(prompt_string, output_string):
                print('='*100)
                print("[USER]: ", prompt )
                print("\n[ASSISTANT]:", completion)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                print('is not peft')
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                print('is peft')
                with _disable_peft_adapters(self.accelerator.unwrap_model(self.model)):
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Always read solutions from the batch (do not depend on reward-loop side effects)
        solutions = [str(example.get("solution", "")) for example in inputs]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = Trainer._prepare_inputs(self, inputs = reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                if "solution" in reward_kwargs:
                    solutions_for_reward = reward_kwargs.pop("solution")
                    if "molecule" in reward_kwargs: 
                        original_mol = reward_kwargs.pop("molecule")
                        removed_group = reward_kwargs.pop("removed_group")
                        added_group = reward_kwargs.pop("added_group")

                        output_reward_func = reward_func(
                            prompts=prompts,
                            completions=completions, 
                            target_mol=solutions_for_reward, original_mol=original_mol, 
                            removed_group=removed_group, added_group=added_group,
                            **reward_kwargs)
                    else:
                        output_reward_func = reward_func(
                            prompts=prompts,
                            completions=completions,
                            solution=solutions_for_reward,
                            **reward_kwargs,
                        )
                else:
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        # print('x_grpo_rewars output:',rewards)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)


        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        print('advantage:', advantages)

        '''DePO'''

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if self.variant == "math":
            solutions_text = [replace_solution(text, x["solution"]) for (text, x) in zip(completions_text, inputs)]
        else:
            solutions_text = [
                replace_smile(text, x["solution"]) for (text, x) in zip(completions_text, inputs)
            ]

        solution_positions = []
        for text, sol in zip(solutions_text, solutions):
            start_char = text.find(sol)
            if start_char == -1:
                solution_positions.append((0, 0))
            else:
                end_char = start_char + len(sol)
                solution_positions.append((start_char, end_char))

        solution_inputs = self.processing_class(
            solutions_text,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,
            return_offsets_mapping=True
        )

        offset_mappings = solution_inputs.pop("offset_mapping")
        batch_size, seq_len = offset_mappings.shape[:2]
        solution_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

        for i in range(batch_size):
            start_char, end_char = solution_positions[i]
            for j in range(seq_len):
                start, end = offset_mappings[i][j].tolist()
                if (start, end) == (0, 0):
                    solution_mask[i][j] = 0
                else:
                    if start < end_char and end > start_char:
                        solution_mask[i][j] = 1
                    else:
                        solution_mask[i][j] = 0

        solution_inputs["attention_mask"] = solution_mask
        solution_inputs = super(GRPOTrainer, self)._prepare_inputs(solution_inputs)
        solution_ids, solution_mask = solution_inputs["input_ids"], solution_inputs["attention_mask"]

        if self.variant == "random_mask":
            pad_token_id = self.processing_class.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processing_class.eos_token_id
            valid_tokens = solution_ids != pad_token_id
            thinking_tokens = valid_tokens & (solution_mask == 0)

            keep_prob = max(0.0, min(1.0, float(self.thinking_mask_keep_prob)))
            if keep_prob == 0.0:
                sampled_thinking = torch.zeros_like(solution_mask, dtype=torch.bool)
            elif keep_prob == 1.0:
                sampled_thinking = thinking_tokens
            else:
                random_vals = torch.rand_like(solution_mask, dtype=torch.float)
                sampled_thinking = (random_vals < keep_prob) & thinking_tokens

            base_mask = solution_mask.bool() if bool(self.keep_final_answer) else torch.zeros_like(solution_mask, dtype=torch.bool)
            solution_mask = (base_mask | sampled_thinking).long()

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if self.reward_func_names is not None and i < len(self.reward_func_names):
                reward_func_name = self.reward_func_names[i]
            elif isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                if hasattr(reward_func, "__name__") and reward_func.__name__ != "<lambda>":
                    reward_func_name = reward_func.__name__
                else:
                    reward_func_name = f"reward_func_{i}"
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "solution_ids": solution_ids,
            "solution_mask": solution_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)


        solution_ids, solution_mask = inputs["solution_ids"], inputs["solution_mask"]

        input_ids = torch.cat([prompt_ids, solution_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, solution_mask], dim=1)
        logits_to_keep = solution_ids.size(1)  # we only need to compute the logits for the completion tokens
        s_per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        s_loss = -(s_per_token_logps * solution_mask).sum() / (solution_mask.sum() + 1e-4)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # advantages = advantages + (1 / (s_loss + 1))
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        self._metrics["s_loss"].append(self.accelerator.gather_for_metrics(s_loss).mean().item())

        loss = loss + s_loss

        return loss
