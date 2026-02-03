import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--fp8_models", default=None, help="Models with FP8 precision, comma-separated.")
    parser.add_argument("--offload_models", default=None, help="Models with offload, comma-separated. Only used in splited training.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    parser.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser


def add_adaptive_dropout_config(parser: argparse.ArgumentParser):
    """Add AdaptiveDropout LoRA configuration arguments."""
    parser.add_argument(
        "--use_adaptive_dropout",
        default=False,
        action="store_true",
        help="Use AdaptiveDropout LoRA instead of standard LoRA. Enables rank dropout, module dropout, and neuron dropout."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Neuron dropout probability for LoRA. Drops neurons in LoRA intermediate activations. (0.0-1.0)"
    )
    parser.add_argument(
        "--rank_dropout",
        type=float,
        default=None,
        help="Rank dropout probability for LoRA. Randomly drops entire rank dimensions during training. Recommended: 0.1-0.3 (0.0-1.0)"
    )
    parser.add_argument(
        "--module_dropout",
        type=float,
        default=None,
        help="Module dropout probability for LoRA. Randomly skips entire LoRA module during training. Recommended: 0.05-0.1 (0.0-1.0)"
    )
    return parser


def add_tag_frequency_dropout_config(parser: argparse.ArgumentParser):
    """Add Tag Frequency-based Dropout configuration arguments."""
    parser.add_argument(
        "--use_tag_frequency_dropout",
        default=False,
        action="store_true",
        help="Enable Tag Frequency-based Adaptive Dropout. Drops frequent tags more often than rare tags."
    )
    parser.add_argument(
        "--tag_dropout_min_rate",
        type=float,
        default=0.0,
        help="Minimum dropout probability for rare tags. (0.0-1.0)"
    )
    parser.add_argument(
        "--tag_dropout_max_rate",
        type=float,
        default=0.5,
        help="Maximum dropout probability for frequent tags. (0.0-1.0)"
    )
    parser.add_argument(
        "--tag_dropout_sigmoid_alpha",
        type=float,
        default=10.0,
        help="Controls how steep the sigmoid curve is for dropout probability."
    )
    parser.add_argument(
        "--tag_dropout_sigmoid_center",
        type=float,
        default=0.5,
        help="The frequency ratio around which sigmoid transitions (0.0-1.0)."
    )
    parser.add_argument(
        "--tag_dropout_trigger_tokens",
        type=str,
        default=None,
        help="Comma-separated tokens that should NEVER be dropped (e.g., character name)."
    )
    parser.add_argument(
        "--tag_dropout_shuffle",
        default=True,
        action="store_true",
        help="Shuffle tags after applying dropout."
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="prompt",
        help="Key in metadata for caption/prompt field."
    )
    # sFAD (Step-based FAD) parameters
    parser.add_argument(
        "--use_step_based_dropout",
        default=False,
        action="store_true",
        help="Enable sFAD (Step-based FAD). Dropout probability gradually increases over training steps."
    )
    parser.add_argument(
        "--step_dropout_schedule",
        type=str,
        default="exponential",
        choices=["linear", "cosine", "exponential", "exp_up", "jump_ramp"],
        help="Schedule for step-based dropout increase. (linear, cosine, exponential, exp_up, jump_ramp)"
    )
    parser.add_argument(
        "--step_dropout_start",
        type=float,
        default=0.0,
        help="Starting step multiplier for sFAD. Near 0 means almost no dropout at start. (0.0-1.0)"
    )
    parser.add_argument(
        "--step_dropout_end",
        type=float,
        default=1.0,
        help="Ending step multiplier for sFAD. 1.0 means full FAD dropout at end. (0.0-1.0)"
    )
    parser.add_argument(
        "--step_dropout_warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for sFAD. During warmup, dropout stays at start value. (0.0-1.0)"
    )
    parser.add_argument(
        "--use_step_dropout_warmup",
        default=True,
        action="store_true",
        help="Enable warmup period for sFAD."
    )
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_adaptive_dropout_config(parser)
    parser = add_tag_frequency_dropout_config(parser)
    parser = add_gradient_config(parser)
    return parser
