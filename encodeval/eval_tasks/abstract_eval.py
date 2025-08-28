import os
from dataclasses import dataclass
import subprocess
from typing import Callable, Dict, Literal

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import TripletDistanceMetric
from transformers import set_seed

from encodeval.tools import ModelTools


@dataclass
class EvalConfig:
    """
    A dataclass to hold and manage configuration for evaluation runs.

    Attributes:
        model_class: Class or callable to instantiate the model.
        model_kwargs: Dictionary of keyword arguments for model instantiation.
        tokenizer_class: Class or callable to instantiate the tokenizer.
        tokenizer_kwargs: Dictionary of keyword arguments for tokenizer instantiation.
        tr_args_class: Class to instantiate training arguments.
        tr_args_kwargs: Dictionary of keyword arguments for training arguments.
        max_length: Maximum sequence length (optional, inferred from model if None).
        load_dataset_from_custom_fn: Optional callable that loads a dataset.
        task_type: Task type identifier (SC, SR, TC, or IR).
        loss_fn: Optional callable for the training loss function.
        loss_kwargs: Optional keyword arguments for the loss function.
    """

    model_class: Callable = None
    model_kwargs: Dict = None
    tokenizer_class: Callable = None
    tokenizer_kwargs: Dict = None  
    tr_args_class: Callable = None
    tr_args_kwargs: Dict = None
    max_length: int = None
    load_dataset_from_custom_fn: Callable = None
    task_type: Literal["SC", "SR", "TC", "IR"] = None
    loss_fn: Callable = None
    loss_kwargs: Dict = None

    def __post_init__(self):
        """
        Initializes the evaluation configuration:
        - Loads the model and tokenizer
        - Sets training/evaluation parameters
        - Configures dataset and loss function
        - Prepares output/logging directories
        """
        # Extract and remove device and dtype from model kwargs
        self.model_dtype = self.model_kwargs.pop("dtype")
        self.device = self.model_kwargs.pop("device")

        # Handle loading fine-tuned model from disk if specified
        ft_model_config_dir = self.model_kwargs.pop("ft_model_config_dir", None)
        if ft_model_config_dir is not None:
            ft_model_path = f"{os.environ['EVAL_MODEL_PATH']}/evaluation/weights/{self.task_type}/{ft_model_config_dir}"
            print(f"Loading fine-tuned model at {ft_model_path}")
            if "pretrained_model_name_or_path" in self.model_kwargs:
                self.model_kwargs["pretrained_model_name_or_path"] = ft_model_path
            elif "model_name_or_path" in self.model_kwargs:
                self.model_kwargs["model_name_or_path"] = ft_model_path
               
        self.load_model()

        # Show the device and dtype of model parameters
        for _, param in self.model.named_parameters():
            print(f"Model weights stored on {param.device} in {param.dtype}")
            break

        # Initialize tokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained(**self.tokenizer_kwargs)

        # Print model summary (number of parameters, structure, etc.)
        ModelTools.model_summary(self.model)

        # Set and adjust maximum sequence length
        model_max_length = (
            self.model[0].max_seq_length if isinstance(self.model, SentenceTransformer) 
            else self.model.config.max_position_embeddings
        )
        self.max_length = model_max_length if self.max_length is None else min(model_max_length, self.max_length)
        self.max_length = round(0.95 * self.max_length)  # Apply 5% buffer
        print(f"Max sequence length set to {self.max_length}")

        # Sync special tokens from model config to tokenizer
        if hasattr(self.model, "config"):
            for attr in [
                "bos_token", "bos_token_id", 
                "eos_token", "eos_token_id", 
                "pad_token", "pad_token_id", 
                "mask_token", "mask_token_id"
            ]:
                if hasattr(self.model.config, attr):
                    setattr(self.tokenizer, attr, getattr(self.model.config, attr))

        # Fallback token setup
        if self.tokenizer.pad_token is None:
            print("Setting PAD token as EOS token")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.mask_token is None:
            print("Model does not have a mask token")

        # Initialize training arguments
        self.callbacks = self.tr_args_kwargs.pop("callbacks", None)
        output_subdir = self.tr_args_kwargs.pop("output_subdir", "")
        train_batch_size = self.tr_args_kwargs.pop("train_batch_size", None)
        self.tr_args = self.tr_args_class(**self.tr_args_kwargs)

        # Ensure CPU-only mode if specified
        if self.device == "cpu":
            self.tr_args.use_cpu = True

        # Adjust gradient accumulation if custom batch size is provided
        if train_batch_size is not None:
            # Handle CPU case where n_gpu is 0
            n_devices = max(1, self.tr_args.n_gpu)
            gradient_accumulation_steps = (
                train_batch_size / (n_devices * self.tr_args.per_device_train_batch_size)
            )
            if gradient_accumulation_steps < 1:
                self.tr_args.per_device_train_batch_size = int(
                    gradient_accumulation_steps * self.tr_args.per_device_train_batch_size
                )
            self.tr_args.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

        # Convert loss distance metric string to enum if applicable
        if self.loss_kwargs is not None:
            if "distance_metric" in self.loss_kwargs and isinstance(self.loss_kwargs["distance_metric"], str):
                self.loss_kwargs["distance_metric"] = getattr(
                    TripletDistanceMetric, self.loss_kwargs["distance_metric"]
                )

        # Set evaluation seed
        set_seed(self.tr_args.seed)

        # Load dataset (from user-defined loader if provided)
        if self.load_dataset_from_custom_fn is not None:
            self.dataset = self.load_dataset_from_custom_fn()
            self.dataset_name = self.load_dataset_from_custom_fn.__name__
        else:
            self.dataset_name = ""

        # Prepare output/log directories
        model_name = os.environ["EVAL_MODEL_PATH"].split("/")[-1]
        output_dir = self.tr_args.output_dir        
        output_subdir = (
            f"{self.task_type}/{self.dataset_name}/{ft_model_config_dir.replace('/', '_')}/{output_subdir}"
            if ft_model_config_dir is not None else f"{self.task_type}/{self.dataset_name}/{output_subdir}"
        )
        self.tr_args.output_dir = f"{os.environ['EVAL_MODEL_PATH']}/evaluation/weights/{output_subdir}"
        self.tr_args.logging_dir = f"{os.environ['EVAL_MODEL_PATH']}/evaluation/logs/{output_subdir}"
        self.results_dir = f"{output_dir}/{model_name}/{output_subdir}"

        # Clear old logs if logging directory is not empty
        if os.path.exists(self.tr_args.logging_dir) and len(os.listdir(self.tr_args.logging_dir)) > 0:
            subprocess.run(f"rm {self.tr_args.logging_dir}/*", shell=True, check=True)

    def load_model(self):
        """
        Loads the model using the specified class and keyword arguments.
        Converts model to the specified dtype and device.
        """
        if self.model_class.__name__ == "SentenceTransformer":
            self.model = self.model_class(**self.model_kwargs)
        else:
            self.model = self.model_class.from_pretrained(**self.model_kwargs)
        self.model = self.model.to(self.model_dtype).to(self.device)


class AbstractEval:
    """
    Abstract base class for implementing evaluation pipelines.

    Subclasses must implement:
        - train(): logic for training the model
        - test(): logic for evaluating the model

    Attributes:
        config: An EvalConfig instance holding all config objects.
        model: The initialized model.
        tokenizer: The tokenizer instance.
        dataset: The dataset to evaluate on.
        dataset_name: The name of the dataset function (if applicable).
        tr_args: Training arguments (e.g., TrainerArguments).
        callbacks: Optional callbacks for training.
        max_length: The maximum sequence length.
        loss_fn: Loss function to use.
        loss_kwargs: Keyword arguments for the loss function.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = config.model
        self.device = config.device
        self.tokenizer = config.tokenizer
        self.dataset = config.dataset
        self.dataset_name = config.dataset_name
        self.tr_args = config.tr_args
        self.callbacks = config.callbacks
        self.max_length = config.max_length
        self.loss_fn = config.loss_fn
        self.loss_kwargs = config.loss_kwargs
