import contextlib
import copy
import json
import logging
import signal
import time
from collections.abc import Mapping
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, List, NewType, Optional, Union
import argparse
import os, wandb
import random
from src.datatypes import NamedParametersToOptimize

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from metrics import calculate_metric
from dataclasses import dataclass, field, asdict
from trainer import OurTrainer

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn import CrossEntropyLoss
from transformers.data.data_collator import DataCollatorMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import PaddingStrategy

InputDataClass = NewType("InputDataClass", Any)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(
        default=None, metadata={"help": "the algorithm to use"}
    )
    global_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to use global evaluation for each round"},
    )
    g_result_file: Optional[str] = field(
        default=None,
        metadata={"help": "the file to save the global evaluation results"},
    )
    num_rounds: Optional[int] = field(
        default=500, metadata={"help": "the number of rounds"}
    )
    num_clients: Optional[int] = field(
        default=2, metadata={"help": "the number of clients"}
    )
    sample_clients: Optional[int] = field(
        default=2, metadata={"help": "the number of clients to sample"}
    )
    split_strategy: Optional[str] = field(
        default="iid", metadata={"help": "the split strategy"}
    )
    alpha: Optional[float] = field(
        default=1,
        metadata={"help": "the alpha parameter of the non-iid split strategy"},
    )
    min_partition_size: Optional[int] = field(
        default=0,
        metadata={"help": "the minimum partition size of the non-iid split strategy"},
    )
    prox_mu: Optional[float] = field(
        default=0.01, metadata={"help": "the mu parameter of FedProx"}
    )
    fedopt_tau: Optional[float] = field(
        default=1e-3,
        metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"},
    )
    fedopt_eta: Optional[float] = field(
        default=1e-3,
        metadata={
            "help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"
        },
    )
    fedopt_beta1: Optional[float] = field(
        default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"}
    )
    fedopt_beta2: Optional[float] = field(
        default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"}
    )
    save_model_freq: Optional[int] = field(
        default=50,
        metadata={
            "help": "the frequency to save the model. 50 means save every 50 rounds"
        },
    )


@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = (
        "SST2"  # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    )
    zo_max_grad_norm: float = None # max grad norm in zeroth-order optimization
    dp_eps: float = 0.1  # epsilon in differential privacy
    
    ## compression related
    compression: str = None  # compression method
    # support qsgd_b
    d : int = 0  # quantization level
    correction: bool = False  # whether to use correction for compression
    
     
    # Number of examples
    num_train: int = (
        0  # ICL mode: number of demonstrations; training mode: number of training samples
    )
    num_dev: int = None  # (only enabled with training) number of development samples
    num_eval: int = None  # number of evaluation samples
    num_train_sets: int = (
        None  # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    )
    train_set_seed: int = 0  # designated seed to sample training samples/demos
    result_file: str = (
        None  # file name for saving performance; if None, then use the task name, model name, and config
    )

    # Model loading
    model_name: str = "facebook/opt-125m"  # HuggingFace model name
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = False  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take
    no_auto_device: bool = (
        False  # do not load model by auto device; should turn this on when using FSDP
    )

    # Calibration
    sfc: bool = False  # whether to use SFC calibration
    icl_sfc: bool = False  # whether to use SFC calibration for ICL samples

    template_ver: int = (
        0  # template. For some tasks (SST2, RTE, Copa), we add template ver=1 as the empty template.
    )

    # Training
    trainer: str = "none"
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo_sgd: zeroth-order SGD (MeZO) training
    ## - zo_conserv: zeroth-order SGD conservative training
    ## - zo_adam: zeroth-order Adam training
    ## - zo_sign_opt: zeroth-order sign sgd training
    ## - forward_grad: forward gradient
    ## - zo_fl: zeroth-order DP training
    optimizer: str = "adamw"
    ## options
    ## - sgd
    ## - adam
    ## - adamw # this is huggingface default
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = (
        False  # take the log likelihood of all options and train as classification
    )
    momentum: float = 0.0  # only work for SGD optimizer

    # MeZO
    zo_eps: float = 1e-3  # eps in MeZO
    perturbation_mode: str = "one_side"
    q: int = 1  # number of Gaussian samples for zeroth-order trainers

    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    no_reparam: bool = True  # do not use reparameterization trick
    prefix_init_by_real_act: bool = (
        True  # initialize prefix by real activations of random words
    )

    # prompt tuning hyperparameters
    prompt_tuning: bool = False  # whether to use prompt tuning
    num_virtual_tokens: int = 10  # number of prompt tokens to use
    prompt_init_by_real_tokens: bool = (
        False  # whether to sample random tokens from Embedding layer
    )

    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Linear probing
    linear_probing: bool = False  # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing
    head_tuning: bool = False  # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False  # untie the embeddings and LM head

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    non_diff: bool = (
        False  # use non-differentiable objective (only support F1 for SQuAD for now)
    )

    # Auto saving when interrupted
    save_on_interrupt: bool = (
        False  # save model when interrupted (useful for long training)
    )

    clean_model_at_end: bool = True  # remove everthing at the end.


def parse_args():
    """
    Parse arguments:
    Returns:
        args: TrainingArguments
        fed_args: FedArguments
    """
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser((FedArguments, OurArguments))
    fed_args, args = parser.parse_args_into_dataclasses()
    # print(args)
    return args, fed_args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HFDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def convert_(framework, samples):
    """
    Convert samples to HF-compatible dataset
    """
    data = []
    for sample in samples:
        encoded_candidates, option_lens = encode_prompt(
            framework.task,
            framework.task.get_template(template_version=framework.args.template_ver),
            [],
            sample,
            framework.tokenizer,
            max_length=framework.args.max_length,
            generation=framework.task.generation,
            generation_with_gold=True,
            max_new_tokens=framework.args.max_new_tokens,
        )
        if framework.task.generation:
            correct_candidate_id = 0
        elif isinstance(sample.correct_candidate, list):
            correct_candidate_id = sample.candidates.index(
                sample.correct_candidate[0]
            )
        else:
            correct_candidate_id = sample.candidates.index(
                sample.correct_candidate
            )

        if framework.args.non_diff:
            # For non-differentiable objective, there is no teacher forcing thus the
            # current answer part is removed
            encoded_candidates[correct_candidate_id] = encoded_candidates[
                correct_candidate_id
            ][: -option_lens[correct_candidate_id]]

        if framework.args.train_as_classification:
            # For classification, we provide the label as the correct candidate id
            data.append(
                [
                    {
                        "input_ids": encoded_candidates[_i],
                        "labels": correct_candidate_id,
                        "option_len": option_lens[_i],
                        "num_options": len(sample.candidates),
                    }
                    for _i in range(len(encoded_candidates))
                ]
            )
        elif framework.args.only_train_option:
            # Otherwise, it is just LM-style teacher forcing
            if framework.args.non_diff:
                # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                data.append(
                    {
                        "input_ids": encoded_candidates[correct_candidate_id],
                        "labels": encoded_candidates[correct_candidate_id],
                        "option_len": option_lens[correct_candidate_id],
                        "gold": sample.correct_candidate,
                    }
                )
            else:
                data.append(
                    {
                        "input_ids": encoded_candidates[correct_candidate_id],
                        "labels": encoded_candidates[correct_candidate_id],
                        "option_len": option_lens[correct_candidate_id],
                    }
                )
        else:
            data.append(
                {
                    "input_ids": encoded_candidates[correct_candidate_id],
                    "labels": encoded_candidates[correct_candidate_id],
                }
            )
    return data

class Framework:

    def __init__(self, args, task, fed_args=None):
        self.args = args
        self.task = task
        self.fed_args = fed_args
        self.model, self.tokenizer = self.load_model()
        self.names_to_optm = self.get_names_to_optm()
    
    def get_names_to_optm(self):
        opt_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                opt_names.append(name)
        return opt_names


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time(
            "Loading model with FP%d" % (16 if self.args.load_float16 else 32)
        ):
            free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
            print(free_in_GB)
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                # Head tuning
                if "opt" in self.args.model_name.lower():
                    from modeling_opt import OPTForCausalLM

                    model = OPTForCausalLM.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                        max_memory={
                            i: f"{free_in_GB - 5}GB"
                            for i in range(torch.cuda.device_count())
                        },
                    )
                elif "llama" in self.args.model_name.lower():
                    from modeling_llama import LlamaForCausalLMWithHeadTuning

                    model = LlamaForCausalLMWithHeadTuning.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                        max_memory={
                            i: f"{free_in_GB - 5}GB"
                            for i in range(torch.cuda.device_count())
                        },
                    )
                elif "mistral" in self.args.model_name.lower():
                    from modeling_mistral import MistralForCausalLMWithHeadTuning

                    model = MistralForCausalLMWithHeadTuning.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                        max_memory={
                            i: f"{free_in_GB - 5}GB"
                            for i in range(torch.cuda.device_count())
                        },
                    )
                else:
                    raise NotImplementedError(
                        f"Head tuning is not supported for {self.args.model_name}"
                    )
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    max_memory={
                        i: f"{free_in_GB - 5}GB"
                        for i in range(torch.cuda.device_count())
                    },
                    load_in_8bit=self.args.load_int8,
                )
            model.eval()

        # Load tokenizer
        #  In mezo, use_fast is set to False. But TypeError will occur when running SQuaD. Setting to be True can fix.
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        if ("llama" in self.args.model_name) or (
            "mistral" in self.args.model_name.lower()
        ):
            # LLaMA padding token
            tokenizer.pad_token_id = 0  # technically <unk>

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix_tuning import PrefixTuning

            PrefixTuning(
                model,
                num_prefix=self.args.num_prefix,
                reparam=not self.args.no_reparam,
                float16=self.args.load_float16,
                init_by_real_act=self.args.prefix_init_by_real_act,
            )
        if self.args.lora:
            from lora import LoRA

            LoRA(
                model,
                r=self.args.lora_r,
                alpha=self.args.lora_alpha,
                float16=self.args.load_float16,
            )

        if self.args.prompt_tuning:
            from prompt_tuning import PromptTuning

            print("Adding Prompt Tuning to model...")
            PromptTuning(
                model,
                num_virtual_tokens=self.args.num_virtual_tokens,
                init_by_real_tokens=self.args.prompt_init_by_real_tokens,
                hide_virtual_token_logits=True,  # a workaround for the other loss/prediction functions
            )
            print(
                "Total/Trainable number of parameters: {}/{}".format(
                    sum(p.numel() for p in model.parameters()),
                    sum(p.numel() for p in model.parameters() if p.requires_grad),
                )
            )

        if self.args.head_tuning:
            if model.config.model_type in ["opt", "llama", "mistral"]:
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer

    def set_model_dict(self, named_parameters_to_optm):
        """
        Set the model parameters to the given named_parameters_to_optm
        """
        state_dict = self.model.state_dict()
        for name in self.names_to_optm:
            state_dict[name].copy_(named_parameters_to_optm[name])
        self.model.load_state_dict(state_dict)

    def get_named_parameters_to_optm(self) -> NamedParametersToOptimize:
        named_parameters_to_optm = NamedParametersToOptimize()
        for name in self.names_to_optm:
            param = self.model.state_dict()[name]
        
            named_parameters_to_optm[name] = param
            param.grad = None
        return named_parameters_to_optm
    
    def agg_model_parameters(self, local_updates):
        """
        Update/Aggregate the model parameters by adding local_updates
        """
        self.set_model_dict(local_updates)


    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids,
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(
                    args.max_new_tokens, args.max_length - input_ids.size(1)
                ),
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                    self.tokenizer.eos_token_id,
                ],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(
                outputs[0][input_ids.size(1) :], skip_special_tokens=True
            ).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[
                torch.arange(len(labels)).to(labels.device), labels
            ]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        # if verbose:
        #     logger.info("========= Example =========")
        #     logger.info(f"Candidate: {eval_sample.candidates}")
        #     logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task,
            self.task.get_template(template_version=self.args.template_ver),
            train_samples,
            eval_sample,
            self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens,
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task,
                self.task.get_template(template_version=self.args.template_ver),
                train_samples,
                eval_sample,
                self.tokenizer,
                max_length=self.args.max_length,
                sfc=self.args.sfc,
                icl_sfc=self.args.icl_sfc,
                generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens,
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            # if verbose:
            #     logger.info("=== Prompt ===")
            #     logger.info(self.tokenizer.decode(encoded_candidates[0]))
            #     logger.info(f"Output: {output_text}")
            return Prediction(
                correct_candidate=eval_sample.correct_candidate,
                predicted_candidate=output_text,
            )
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(
                    encoded_candidate, option_len=option_lens[candidate_id]
                )
                if verbose:
                    # if candidate_id == 0:
                    #     logger.info("=== Candidate %d ===" % candidate_id)
                    #     logger.info(self.tokenizer.decode(encoded_candidate))
                    # else:
                    #     logger.info("=== Candidate %d (without context)===" % candidate_id)
                    #     logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(
                        f"Log probabilities of the option tokens: {selected_log_probs}"
                    )

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(
                        sfc_encoded_candidates[candidate_id],
                        option_len=sfc_option_lens[candidate_id],
                    )  # if verbose:  #     logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)  #     logger.info(  #         self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])  #     logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append(
                    {
                        "log_probs": selected_log_probs,
                        "sfc_log_probs": (
                            sfc_selected_log_probs
                            if self.args.sfc or self.args.icl_sfc
                            else None
                        ),
                    }
                )

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [
                    x["log_probs"].sum().item() - x["sfc_log_probs"].sum().item()
                    for x in outputs
                ]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x["log_probs"].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [
                    eval_sample.candidates.index(c)
                    for c in eval_sample.correct_candidate
                ]
            else:
                correct_candidate_id = eval_sample.candidates.index(
                    eval_sample.correct_candidate
                )

            return Prediction(
                correct_candidate=correct_candidate_id,
                predicted_candidate=int(np.argmax(scores)),
            )

    def evaluate(
        self,
        train_samples,
        eval_samples,
        one_train_set_per_eval_sample=False,
        description=None,
    ):
        """
        Evaluate function.
        Here, train_samples are used for demonstrations for ICL.
        If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        Otherwise, the same training set is used for all eval samples.
        """
        if one_train_set_per_eval_sample:
            logger.info(
                f"There are {len(eval_samples)} validation samples and one train set per eval sample"
            )
        else:
            logger.info(
                f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples"
            )

        # Prediction loop
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples, desc=description)):
            predictions.append(
                self.one_step_pred(
                    (
                        train_samples[eval_id]
                        if one_train_set_per_eval_sample
                        else train_samples
                    ),
                    eval_sample,
                    verbose=False,
                )
            )

        # Calculate metrics
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def fl_global_eval(self):
        eval_samples = self.task.sample_subset("valid", seed=0, num=self.args.num_eval)
        if self.fed_args.global_eval:  
            metrics = self.evaluate(
                [], eval_samples, description="Evaluating on the Gloabl Test Set"
            )
            _keys = list(metrics.keys())
            for m in _keys:
                metrics["g_test_" + m] = metrics[m]
            
        
            if self.args.local_rank <= 0:
                logger.info(metrics)
                wandb.log(metrics)
                write_metrics_to_file(
                    metrics,
                    (
                        "result/"
                        + result_file_tag(self.args)
                        + f"-fl{round}.json"
                        if self.fed_args.g_result_file is None
                        else self.fed_args.g_result_file
                    ),
                )
        if self.args.trainer != "none" and self.args.clean_model_at_end:
            self.delete_checkpoints()

    def train(
        self, train_samples, dev_samples, eval_samples, client_id=-1, current_round=-1
    ):
        """
        Training function
        if self.num_dev is not None, eval_samples are dev_samples
        """
        logger.info(f"Eval sample length is {len(eval_samples)}")
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(convert_(self, train_samples))
            eval_dataset = HFDataset(convert_(self, eval_samples))
            dev_dataset = HFDataset(convert_(self, dev_samples))

        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(
                self.model, type(self.model)
            )

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        self.trainer = OurTrainer(
            model=self.model,
            client_id=client_id,
            current_round=current_round,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=(
                DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8)
                if self.args.train_as_classification
                else collator(self.tokenizer, pad_to_multiple_of=8)
            ),
            eval_samples=eval_samples,
            dev_samples=dev_samples,
            evaluate_func=self.evaluate,
        )

        if self.args.trainer == "zo_fl" and self.fed_args.fed_alg == "fedavg":
            self.local_es_mangnitude_grad = self.trainer.local_projected_grads
            logging.info(
                f"Client [{client_id}]: local_es_mangnitude_grad: {self.local_es_mangnitude_grad}"
            )
            self.trainer.local_projected_grads = 0 # reset the local_projected_grads

        if self.args.save_on_interrupt:
            self.trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint

        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        # This calls the trainer._inner_training_loop()
        logging.info("last_checkpoint: %s" % last_checkpoint) 
        # if client_id != -1:
        #     last_checkpoint = os.path.join(last_checkpoint, f"client_{client_id}")
        self.trainer.train(resume_from_checkpoint=last_checkpoint)

        # Explicitly save the model
        if self.args.save_model:
            logger.info("Save model..")
            self.trainer.save_model()

        # FSDP compatibility
        self.model = self.trainer.model

        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info(
                    "This is an FSDP model now. Be careful when assigning back the original forward function"
                )
                self.model._fsdp_wrapped_module.forward = (
                    self.model._fsdp_wrapped_module.original_forward
                )
            else:
                self.model.forward = self.model.original_forward

    def delete_checkpoints(self):
        import shutil

        print(f"\nWARNING: Removing everything at end: {self.args.output_dir}")
        deleted_folders = [
            folder
            for folder in os.listdir(self.args.output_dir)
            if os.path.isdir(os.path.join(self.args.output_dir, folder))
            and folder.startswith("checkpoint-")
        ]
        for f in deleted_folders:
            shutil.rmtree(os.path.join(self.args.output_dir, f))
        print(f"deleted folders: ", deleted_folders)

    def start_run(self):
        train_sets = self.task.sample_train_sets(
            num_train=self.args.num_train,
            num_dev=self.args.num_dev,
            num_eval=self.args.num_eval,
            num_train_sets=self.args.num_train_sets,
            seed=self.args.train_set_seed,
        )

        if self.args.train_set_seed is not None or self.args.num_train_sets is not None:

            # Eval samples share one (or multiple) training set(s)
            for train_set_id, train_samples in enumerate(train_sets):
                train_set_seed = (
                    train_set_id
                    if self.args.train_set_seed is None
                    else self.args.train_set_seed
                )

                # Sample eval samples
                if self.args.num_eval is not None:
                    eval_samples = self.task.sample_subset(
                        data_split="valid", seed=train_set_seed, num=self.args.num_eval
                    )
                else:
                    eval_samples = self.task.valid_samples

                if self.args.trainer != "none":
                    # Here the training samples are seperated
                    if self.args.num_dev is not None:
                        # Dev samples
                        # assert args.num_dev + args.num_train <= len(train_samples), f"num_dev({args.num_dev})+num_train({args.num_train}) is more than actual num of training samples ({len(train_samples)})."
                        dev_samples = train_samples[-self.args.num_dev :]
                        train_samples = train_samples[: -self.args.num_dev]
                        logger.info("Dev samples: %d" % len(dev_samples))
                        logger.info("Train samples: %d" % len(train_samples))
                    else:
                        dev_samples = None
                        logger.info("Train samples: %d" % len(train_samples))
                        logger.info("No dev samples")
                    # Training

                    self.train(
                        train_samples,
                        dev_samples if dev_samples is not None else eval_samples,
                        eval_samples,
                    )

                    if not self.args.no_eval:  # This is True
                        metrics = self.evaluate(
                            [], eval_samples, description="Evaluating on the Test Set"
                        )
                        _keys = list(metrics.keys())
                        for m in _keys:
                            metrics["test_" + m] = metrics[m]
                        if dev_samples is not None:
                            dev_metrics = self.evaluate(
                                [],
                                dev_samples,
                                description="Evaluating on the Validation Set",
                            )
                            _keys = list(dev_metrics.keys())
                            for m in _keys:
                                metrics["val_" + m] = dev_metrics[m]
                else:
                    assert self.args.num_dev is None
                    # Zero-shot / in-context learning
                    metrics = self.evaluate(train_samples, eval_samples)
                logger.info(metrics)
                wandb.log(metrics)

                if not self.args.no_eval:
                    logger.info("===== Train set %d =====" % train_set_seed)
                    logger.info(metrics)
                    wandb.log(metrics)
                    if self.args.local_rank <= 0:
                        write_metrics_to_file(
                            metrics,
                            (
                                "result/"
                                + result_file_tag(self.args)
                                + f"-trainset{train_set_id}.json"
                                if self.args.result_file is None
                                else self.args.result_file
                            ),
                        )
                if self.args.trainer != "none" and self.args.clean_model_at_end:
                    self.delete_checkpoints()

        else:
            # For each eval sample, there is a training set. no training is allowed
            # This is for in-context learning (ICL)
            assert self.args.trainer == "none"
            if self.args.num_eval is not None:
                eval_samples = self.task.sample_subset(
                    data_split="valid", seed=0, num=self.args.num_eval
                )
            else:
                eval_samples = self.task.valid_samples
            metrics = self.evaluate(
                train_samples, eval_samples, one_train_set_per_eval_sample=True
            )
            logger.info(metrics)
            wandb.log(metrics)
            if self.args.local_rank <= 0:
                write_metrics_to_file(
                    metrics,
                    (
                        "result/" + result_file_tag(self.args) + "-onetrainpereval.json"
                        if self.args.result_file is None
                        else self.args.result_file
                    ),
                )

    def weighted_update(self, weight, client_id):
        model_state_dict = self.model.state_dict()
        for name in self.names_to_optm:
            model_state_dict[name] = weight * model_state_dict[name]
        self.model.load_state_dict(model_state_dict)

        
def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = (
        "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    )
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return (
        f"{args.task_name}-{save_model_name}"
        + sfc_tag
        + icl_sfc_tag
        + sample_eval_tag
        + sample_train_tag
        + sample_dev_tag
        + customized_tag
    )


def forward_wrap_with_option_len(
    self,
    input_ids=None,
    labels=None,
    option_len=None,
    num_options=None,
    return_dict=None,
    **kwargs,
):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    """
    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs

    # in prompt tuning, we need to remove the virtual tokens from the logits to match the input ids
    logits = outputs.logits

    loss = None
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    # for _i, _len in enumerate(option_len):
    #     shift_labels[_i, :-_len] = -100
    # re-write the above code to avoid the for loop
    non_option_len = shift_labels.shape[1] - option_len
    mask = torch.arange(shift_labels.shape[1], device=shift_labels.device).expand(
        shift_labels.shape[0], -1
    ) < non_option_len.unsqueeze(-1)
    shift_labels[mask] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None:
        # Train as a classification tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100  # Option part
        shift_labels[~mask] = 0  # So that it doesn't mess up with indexing

        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(
            -1
        )  # (bsz x num_options)

        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(
                    0
                )  # (1, num_options)
                _labels = labels[start_id:end_id][0].unsqueeze(0)  # (1)
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(
                -1, num_options
            )  # (bsz, num_options)
            labels = labels.view(-1, num_options)[
                :, 0
            ]  # Labels repeat so we only take the first one
            loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def encode_prompt(
    task,
    template,
    train_samples,
    eval_sample,
    tokenizer,
    max_length,
    sfc=False,
    icl_sfc=False,
    generation=False,
    generation_with_gold=False,
    max_new_tokens=None,
):
    """
    Encode prompts for eval_sample
    Input:
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, max_length: tokenizer and max length
    - sfc: generate prompts for calibration (surface form competition; https://arxiv.org/abs/2104.08315)
    - icl_sfc: generate prompts for ICL version calibration
    - generation: whether it is an generation task
    - generation_with_gold: whether to include the generation-task gold answers (for training)
    - max_new_tokens: max number of new tokens to generate so that we can save enough space
      (only for generation tasks)
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """

    # Demonstrations for ICL
    train_prompts = [
        template.verbalize(sample, sample.correct_candidate).strip()
        for sample in train_samples
    ]
    train_prompts = task.train_sep.join(train_prompts).strip()

    # sfc or icl_sfc indicates that this example is used for calibration
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc
        verbalize_fn = template.verbalize_sfc
    else:
        encode_fn = template.encode
        verbalize_fn = template.verbalize

    unverbalized_eval_prompt = encode_fn(eval_sample).strip(" ")
    if not generation:
        # We generate one prompt for each candidate (different classes in classification)
        # or different choices in multiple-choice tasks
        verbalized_eval_prompts = [
            verbalize_fn(eval_sample, cand).strip(" ")
            for cand in eval_sample.candidates
        ]
        unverbalized_eval_prompt_length = len(
            tokenizer.encode(unverbalized_eval_prompt)
        )
        option_lens = [
            (
                len(tokenizer.encode(verbalized_eval_prompt))
                - unverbalized_eval_prompt_length
            )
            for verbalized_eval_prompt in verbalized_eval_prompts
        ]

        if sfc:
            # Without demonstrations
            final_prompts = verbalized_eval_prompts
        else:
            # With demonstrations
            final_prompts = [
                (train_prompts + task.train_sep + eval_prompt).lstrip().strip(" ")
                for eval_prompt in verbalized_eval_prompts
            ]
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [
                verbalize_fn(eval_sample, eval_sample.correct_candidate)
            ]
            unverbalized_eval_prompt_length = len(
                tokenizer.encode(unverbalized_eval_prompt)
            )
            option_lens = [
                (
                    len(tokenizer.encode(verbalized_eval_prompt))
                    - unverbalized_eval_prompt_length
                )
                for verbalized_eval_prompt in verbalized_eval_prompts
            ]
            final_prompts = [
                (train_prompts + task.train_sep + eval_prompt).lstrip().strip(" ")
                for eval_prompt in verbalized_eval_prompts
            ]
        else:
            option_lens = [0]
            final_prompts = [
                (train_prompts + task.train_sep + unverbalized_eval_prompt)
                .lstrip()
                .strip(" ")
            ]

    # Tokenize
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    # Truncate (left truncate as demonstrations are less important)
    if generation and max_new_tokens is not None:
        max_length = max_length - max_new_tokens

    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    if hasattr(tokenizer, "add_bos_token") and tokenizer.add_bos_token:
        encodings = [
            encoding[0:1] + encoding[1:][-(max_length - 1) :] for encoding in encodings
        ]
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]

    return encodings, option_lens


@dataclass
class ICLCollator:
    """
    Collator for ICL
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        pad_id = self.tokenizer.pad_token_id

        pad_ids = {
            "input_ids": pad_id,
            "attention_mask": 0,
            "sfc_input_ids": pad_id,
            "sfc_attention_mask": 0,
            "labels": pad_id,
        }
        for key in first:
            pp = pad_ids[key]
            lens = [len(f[key]) for f in features]
            max_len = max(lens)
            feature = np.stack(
                [
                    np.pad(
                        f[key],
                        (0, max_len - lens[i]),
                        "constant",
                        constant_values=(0, pp),
                    )
                    for i, f in enumerate(features)
                ]
            )
            padded_feature = torch.from_numpy(feature).long()
            batch[key] = padded_feature

        return batch


@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class NondiffCollator(DataCollatorMixin):
    """
    Collator for non-differentiable objectives
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        no_labels_features = [
            {k: v for k, v in feature.items() if k != label_name and k != "gold"}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label)
                + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label))
                + to_list(label)
                for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        if "gold" in features[0]:
            batch["gold"] = [feature["gold"] for feature in features]

        return batch


class SIGUSR1Callback(transformers.TrainerCallback):
    """
    This callback is used to save the model when a SIGUSR1 signal is received
    (SLURM stop signal or a keyboard interruption signal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def write_predictions_to_file(final_preds, output):
    with open(output, "w") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, cls=EnhancedJSONEncoder) + "\n")


def write_metrics_to_file(metrics, output):
    json.dump(metrics, open(output, "w"), cls=EnhancedJSONEncoder, indent=4)
