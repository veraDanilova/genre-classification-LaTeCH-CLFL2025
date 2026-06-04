#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning BERT / RoBERTa-style models for masked language modeling (MLM).

All training hyperparameters and model selection are controlled via a JSON
config file.  Pass the path to a config as the sole argument:

    python run_mlm.py configs/hmbert.json

Available pre-built configs (see configs/):
    hmbert.json      – dbmdz/bert-base-historic-multilingual-cased  (20 epochs)
    mbert.json       – google-bert/bert-base-multilingual-cased      (5 epochs)
    xlmroberta.json  – FacebookAI/xlm-roberta-base                   (10 epochs)

You can also pass individual command-line arguments as usual (see --help).

GPU selection: set CUDA_VISIBLE_DEVICES in the environment before running,
e.g.:  CUDA_VISIBLE_DEVICES=0 python run_mlm.py configs/hmbert.json
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

check_min_version("4.45.0.dev0")
require_version("datasets>=2.14.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer to fine-tune."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint for weights initialisation. Omit to train from scratch."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type when training from scratch: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={"help": "Override default config when training from scratch, e.g. n_embd=10,resid_pdrop=0.2"},
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name/path if different from model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name/path if different from model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HuggingFace cache directory"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast (Rust-backed) tokenizer"})
    model_revision: str = field(default="main", metadata={"help": "Model version: branch name, tag, or commit id"})
    token: str = field(default=None, metadata={"help": "HuggingFace Hub auth token"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code from Hub"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Load model under this dtype (auto/bfloat16/float16/float32)", "choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    low_cpu_mem_usage: bool = field(default=False, metadata={"help": "Load model as empty shell then fill weights (saves RAM)"})

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides cannot be used with --config_name or --model_name_or_path")


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to data input for training and evaluation."""

    dataset_name: Optional[str] = field(default=None, metadata={"help": "HuggingFace Hub dataset name"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "HuggingFace dataset config name"})
    train_file: Optional[str] = field(default=None, metadata={"help": "Training text file (.txt / .csv / .json)"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Validation text file"})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite cached tokenised datasets"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "% of train set used as validation when no separate validation file is provided"},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Max total input sequence length after tokenisation; longer sequences are truncated"},
    )
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Processes for data preprocessing"})
    mlm_probability: float = field(default=0.15, metadata={"help": "Fraction of tokens masked for MLM"})
    line_by_line: bool = field(default=False, metadata={"help": "Treat each line as a separate sequence"})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Pad all samples to max_seq_length (static padding)"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate training set for debugging"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Truncate eval set for debugging"})
    streaming: bool = field(default=False, metadata={"help": "Enable dataset streaming mode"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "Streaming requires datasets>=2.0.0")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Provide either a dataset_name or train_file / validation_file.")
        for attr, name in [(self.train_file, "train_file"), (self.validation_file, "validation_file")]:
            if attr is not None and attr.split(".")[-1] not in ["csv", "json", "txt"]:
                raise ValueError(f"`{name}` must be a .csv, .json, or .txt file.")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_mlm", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
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

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {training_args.parallel_mode.value == 'distributed'}, "
        f"fp16: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Resume from checkpoint if available
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected – resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)

    # Load dataset
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name,
            cache_dir=model_args.cache_dir, token=model_args.token,
            streaming=data_args.streaming, trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir, token=model_args.token,
                streaming=data_args.streaming, trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir, token=model_args.token,
                streaming=data_args.streaming, trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        if data_args.train_file:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        extension = "text" if extension == "txt" else extension
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, token=model_args.token)
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                extension, data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir, token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                extension, data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir, token=model_args.token,
            )

    # Load model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir, "revision": model_args.model_revision,
        "token": model_args.token, "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("Instantiating new config from scratch.")
        if model_args.config_overrides:
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir, "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision, "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("Provide --tokenizer_name or --model_name_or_path.")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config, cache_dir=model_args.cache_dir,
            revision=model_args.model_revision, token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype, low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch.")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Determine max sequence length
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning("model_max_length > 1024; capping at 1024. Override with --max_seq_length.")
            max_seq_length = 1024
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    column_names = list(raw_datasets["train" if training_args.do_train else "validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Tokenisation
    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            examples[text_column_name] = [l for l in examples[text_column_name] if l and not l.isspace()]
            return tokenizer(
                examples[text_column_name], padding=padding,
                truncation=True, max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="line-by-line tokenisation"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenising dataset line-by-line",
            ) if not data_args.streaming else raw_datasets.map(
                tokenize_function, batched=True, remove_columns=[text_column_name],
            )
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="tokenisation"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenising dataset",
            ) if not data_args.streaming else raw_datasets.map(
                tokenize_function, batched=True, remove_columns=column_names,
            )

        def group_texts(examples):
            concatenated = {k: list(chain(*examples[k])) for k in examples}
            total = (len(concatenated[list(examples.keys())[0]]) // max_seq_length) * max_seq_length
            return {k: [t[i:i + max_seq_length] for i in range(0, total, max_seq_length)] for k, t in concatenated.items()}

        with training_args.main_process_first(desc="grouping texts"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping into chunks of {max_seq_length}",
            ) if not data_args.streaming else tokenized_datasets.map(group_texts, batched=True)

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train split.")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation split.")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

        def preprocess_logits_for_metrics(logits, labels):
            return (logits[0] if isinstance(logits, tuple) else logits).argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels, preds = labels.reshape(-1), preds.reshape(-1)
            mask = labels != -100
            return metric.compute(predictions=preds[mask], references=labels[mask])

    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = min(
            data_args.max_train_samples if data_args.max_train_samples else len(train_dataset),
            len(train_dataset),
        )
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = min(
            data_args.max_eval_samples if data_args.max_eval_samples else len(eval_dataset),
            len(eval_dataset),
        )
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name:
        kwargs["dataset_tags"] = data_args.dataset_name
        kwargs["dataset"] = (
            f"{data_args.dataset_name} {data_args.dataset_config_name}"
            if data_args.dataset_config_name else data_args.dataset_name
        )

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
