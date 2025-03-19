import os
from dataclasses import dataclass
import subprocess
from typing import Callable, Dict, Literal

from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import TripletDistanceMetric
from transformers import set_seed

from optimus.trainer.model.tools import ModelTools


@dataclass
class EvalConfig:
    model_class: Callable = None
    model_kwargs: Dict = None
    tokenizer_class: Callable = None
    tokenizer_kwargs: Dict = None  
    tr_args_class: Callable = None
    tr_args_kwargs: Dict = None
    max_length: int = None
    load_dataset_from_custom_fn: Callable = None
    dataset = None
    task_type: Literal["SC", "SR", "TC", "IR"] = None
    mteb_kwargs: Dict = None
    connection_layer: int = None
    lora_config: LoraConfig = None
    mlm_probability: float = 0.5
    whole_word_masking: bool = False
    loss_fn: Callable = None
    loss_kwargs: Dict = None

    def __post_init__(self):
        # Load model    
        self.model_dtype = self.model_kwargs.pop("dtype")
        self.device = self.model_kwargs.pop("device")
        ft_model_config_dir = (
            self.model_kwargs.pop("ft_model_config_dir") 
            if "ft_model_config_dir" in self.model_kwargs else None
        )

        if ft_model_config_dir is not None:
            ft_model_path = f"{os.environ['EVAL_MODEL_PATH']}/evaluation/weights/{self.task_type}/{ft_model_config_dir}"
            print(f"Loading fine-tuned model at {ft_model_path}")
            if "pretrained_model_name_or_path" in self.model_kwargs:
                self.model_kwargs["pretrained_model_name_or_path"] = ft_model_path
            elif "model_name_or_path" in self.model_kwargs:
                self.model_kwargs["model_name_or_path"] = ft_model_path
               
        self.load_model()
        
        # Print model weights format
        for _, param in self.model.named_parameters():
            print(f"Model weights stored on {param.device} in {param.dtype}")
            break

        # Load tokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained(**self.tokenizer_kwargs)

        # Set-up LoRA training if relevant and print model summary
        if self.lora_config is not None:
            print("Setting LoRA training")
            self.set_lora_training()
        else:
            ModelTools.model_summary(self.model)

        # Set max sequence length
        model_max_length = (
            self.model[0].max_seq_length if isinstance(self.model, SentenceTransformer) 
            else self.model.config.max_position_embeddings
        )
        self.max_length = (
            model_max_length if self.max_length is None
            else min(model_max_length, self.max_length)
        )
        self.max_length = round(0.95 * self.max_length)
        print(f"Max sequence length set to {self.max_length}")

        # Check special tokens
        if hasattr(self.model, "config"):
            for attr in [
                "bos_token", "bos_token_id", 
                "eos_token", "eos_token_id", 
                "pad_token", "pad_token_id", 
                "mask_token", "mask_token_id"
            ]:
                if hasattr(self.model.config, attr):
                    setattr(self.tokenizer, attr, getattr(self.model.config, attr))
        
        if self.tokenizer.pad_token is None:
            print("Setting PAD token as EOS token")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.mask_token is None:
            print("Model does not have a mask token")

        # Set training arguments      
        self.callbacks = ( 
            self.tr_args_kwargs.pop("callbacks") 
            if "callbacks" in self.tr_args_kwargs else None
        )
        output_subdir = ( 
            self.tr_args_kwargs.pop("output_subdir") 
            if "output_subdir" in self.tr_args_kwargs else ""
        )  
        train_batch_size = (
            self.tr_args_kwargs.pop("train_batch_size") 
            if "train_batch_size" in self.tr_args_kwargs else None
        )
        self.tr_args = self.tr_args_class(**self.tr_args_kwargs)

        if self.device == "cpu":
            self.tr_args.use_cpu = True
        
        if train_batch_size is not None:
            gradient_accumulation_steps = (
                train_batch_size / (self.tr_args.n_gpu * self.tr_args.per_device_train_batch_size)
            )
            if gradient_accumulation_steps < 1:
                self.tr_args.per_device_train_batch_size = int(
                    gradient_accumulation_steps * self.tr_args.per_device_train_batch_size
                )
            self.tr_args.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        
        # Check loss arguments
        if self.loss_kwargs is not None:
            if "distance_metric" in self.loss_kwargs and isinstance(self.loss_kwargs["distance_metric"], str):
                self.loss_kwargs["distance_metric"] = getattr(TripletDistanceMetric, self.loss_kwargs["distance_metric"])

        # Set eval seed
        set_seed(self.tr_args.seed)

        # Load dataset
        if self.load_dataset_from_custom_fn is not None:
            self.dataset = self.load_dataset_from_custom_fn()
            self.dataset_name = self.load_dataset_from_custom_fn.__name__
        else:
            self.dataset_name = ""

        # Specify output, logs and results dirs
        model_name = os.environ["EVAL_MODEL_PATH"].split("/")[-1]
        output_dir = self.tr_args.output_dir        
        output_subdir = (
            f"{self.task_type}/{self.dataset_name}/{ft_model_config_dir.replace('/', '_')}/{output_subdir}"
            if ft_model_config_dir is not None else f"{self.task_type}/{self.dataset_name}/{output_subdir}"
        )
        self.tr_args.output_dir = f"{os.environ['EVAL_MODEL_PATH']}/evaluation/weights/{output_subdir}"
        self.tr_args.logging_dir = f"{os.environ['EVAL_MODEL_PATH']}/evaluation/logs/{output_subdir}"
        self.results_dir = f"{output_dir}/{model_name}/{output_subdir}"
        
        # Clear logging dir
        if os.path.exists(self.tr_args.logging_dir) and len(os.listdir(self.tr_args.logging_dir)) > 0:
            subprocess.run(f"rm {self.tr_args.logging_dir}/*", shell=True, check=True)
        
    def load_model(self):
        if self.model_class.__name__ == "SentenceTransformer":
            self.model = self.model_class(**self.model_kwargs)
        else:
            self.model = self.model_class.from_pretrained(**self.model_kwargs)
        self.model = self.model.to(self.model_dtype).to(self.device)

    def set_lora_training(self):
        model_family_name = (
            self.model._first_module().auto_model.__class__.__name__ 
            if isinstance(self.model, SentenceTransformer) 
            else self.model.__class__.__name__
        )
        if model_family_name.startswith("Optimus"):
            self.lora_config.target_modules = ["q_proj", "k_proj", "v_proj"]
        elif model_family_name.startswith("XLMRoberta"):
            self.lora_config.target_modules = ["query", "key", "value"]
        elif model_family_name.startswith("DebertaV2Model"):
            self.lora_config.target_modules = ["query_proj", "key_proj", "value_proj"]
        elif model_family_name.startswith("ModernBertModel"):
            self.lora_config.target_modules = ["Wqkv"]
        elif model_family_name.startswith("NewModel"):
            self.lora_config.target_modules = ["qkv_proj"]
        else:
            raise NotImplementedError
        if isinstance(self.model, SentenceTransformer) :
            self.model._first_module().auto_model = get_peft_model(
                self.model._first_module().auto_model, self.lora_config
            )
            self.model._first_module().auto_model.print_trainable_parameters()
        else:
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()


class AbstractEval:
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
        self.mteb_kwargs = config.mteb_kwargs
        self.mlm_probability = config.mlm_probability
        self.whole_word_masking = config.whole_word_masking
        self.loss_fn = config.loss_fn
        self.loss_kwargs = config.loss_kwargs

    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
