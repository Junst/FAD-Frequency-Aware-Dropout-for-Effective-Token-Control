#!/usr/bin/env python3
"""
Inference and metrics utilities for DiffSynth-Studio.

Subcommands:
  infer            Run inference and compute metrics.
  prepare-datasets Generate DiffSynth metadata JSONs for datasets.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.getenv("DIFFSYNTH_DATA_ROOT", "/scratch2/solbon1212/datasets/datasets")


# -----------------------------
# Common helpers
# -----------------------------

def load_metadata(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def calculate_dino_similarity(original_dir: str, generated_dir: str, metadata_path: str):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers -q")
        from transformers import AutoModel, AutoProcessor

    print("Loading DINO model...")
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    processor = AutoProcessor.from_pretrained("facebook/dinov2-base")

    metadata = load_metadata(metadata_path)
    similarities = []

    for item in metadata:
        original_path = item.get("image", item.get("file_name", ""))
        original_name = Path(original_path).stem
        generated_path = os.path.join(generated_dir, f"{original_name}.png")

        if not os.path.exists(original_path) or not os.path.exists(generated_path):
            continue

        try:
            orig_img = Image.open(original_path).convert("RGB")
            gen_img = Image.open(generated_path).convert("RGB")

            orig_inputs = processor(images=orig_img, return_tensors="pt").to("cuda")
            gen_inputs = processor(images=gen_img, return_tensors="pt").to("cuda")

            with torch.no_grad():
                orig_features = model(**orig_inputs).last_hidden_state.mean(dim=1)
                gen_features = model(**gen_inputs).last_hidden_state.mean(dim=1)

            similarity = torch.nn.functional.cosine_similarity(orig_features, gen_features).item()
            similarities.append(similarity)
        except Exception as exc:
            print(f"Error processing {original_name}: {exc}")
            continue

    if not similarities:
        return None

    return {
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "count": len(similarities),
    }


def calculate_fid(original_dir: str, generated_dir: str, metadata_path: str):
    try:
        from cleanfid import fid
    except ImportError:
        print("Installing clean-fid...")
        os.system("pip install clean-fid -q")
        from cleanfid import fid

    import tempfile

    metadata = load_metadata(metadata_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_tmp = os.path.join(tmpdir, "original")
        gen_tmp = os.path.join(tmpdir, "generated")
        os.makedirs(orig_tmp, exist_ok=True)
        os.makedirs(gen_tmp, exist_ok=True)

        count = 0
        for item in metadata:
            original_path = item.get("image", item.get("file_name", ""))
            original_name = Path(original_path).stem
            generated_path = os.path.join(generated_dir, f"{original_name}.png")

            if not os.path.exists(original_path) or not os.path.exists(generated_path):
                continue

            try:
                orig_img = Image.open(original_path).convert("RGB").resize((299, 299))
                gen_img = Image.open(generated_path).convert("RGB").resize((299, 299))

                orig_img.save(os.path.join(orig_tmp, f"{count:04d}.png"))
                gen_img.save(os.path.join(gen_tmp, f"{count:04d}.png"))
                count += 1
            except Exception as exc:
                print(f"Error processing {original_name}: {exc}")
                continue

        if count < 2:
            print("Not enough images for FID calculation")
            return None

        print(f"Calculating FID with {count} image pairs...")
        return fid.compute_fid(orig_tmp, gen_tmp)


# -----------------------------
# Inference backends
# -----------------------------

def run_inference_flux(
    lora_path: str,
    metadata_path: str,
    output_dir: str,
    num_images: Optional[int],
    steps: int,
):
    from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

    os.makedirs(output_dir, exist_ok=True)
    metadata = load_metadata(metadata_path)
    if num_images:
        metadata = metadata[:num_images]

    print(f"Loaded {len(metadata)} prompts from {metadata_path}")
    print(f"Using LoRA: {lora_path}")
    print(f"Output directory: {output_dir}")

    print("Loading Flux pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )

    print("Loading LoRA weights...")
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

    for idx, item in enumerate(metadata):
        prompt = item["prompt"]
        original_image_path = item["image"]
        original_name = Path(original_image_path).stem
        output_path = os.path.join(output_dir, f"{original_name}.png")

        if os.path.exists(output_path):
            print(f"[{idx+1}/{len(metadata)}] Skipping {original_name} (exists)")
            continue

        print(f"[{idx+1}/{len(metadata)}] Generating: {original_name}")

        try:
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                height=1024,
                width=1024,
            )
            image.save(output_path)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

    print(f"Inference complete. Outputs saved to {output_dir}")
    return output_dir


def run_inference_qwen(
    lora_path: str,
    metadata_path: str,
    output_dir: str,
    num_images: Optional[int],
    steps: int,
    seed: int,
):
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

    os.makedirs(output_dir, exist_ok=True)
    metadata = load_metadata(metadata_path)
    if num_images:
        metadata = metadata[:num_images]

    print(f"Loaded {len(metadata)} prompts from {metadata_path}")
    print(f"Using LoRA: {lora_path}")
    print(f"Output directory: {output_dir}")

    print("Loading Qwen-Image pipeline...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    )

    print("Loading LoRA weights...")
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

    for idx, item in enumerate(metadata):
        prompt = item.get("prompt", item.get("caption", ""))
        original_image_path = item.get("image", item.get("file_name", ""))
        original_name = Path(original_image_path).stem
        output_path = os.path.join(output_dir, f"{original_name}.png")

        if os.path.exists(output_path):
            print(f"[{idx+1}/{len(metadata)}] Skipping {original_name} (exists)")
            continue

        print(f"[{idx+1}/{len(metadata)}] Generating: {original_name}")

        try:
            image = pipe(
                prompt=prompt,
                seed=seed,
                num_inference_steps=steps,
                height=1024,
                width=1024,
            )
            image.save(output_path)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

    print(f"Inference complete. Outputs saved to {output_dir}")
    return output_dir


# -----------------------------
# Dataset preparation
# -----------------------------

def find_image_file(image_dir: str, filename: str) -> Optional[str]:
    base = Path(filename).stem
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".avif"]:
        candidate = os.path.join(image_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(image_dir, filename)
    if os.path.exists(candidate):
        return candidate
    return None


def prepare_dataset(name: str, config: Dict):
    print("=" * 60)
    print(f"Processing {name}")
    print("=" * 60)

    with open(config["source_json"], "r", encoding="utf-8") as handle:
        data = json.load(handle)

    trigger = config["trigger"]
    anchoring = config["anchoring"]
    old_trigger = config.get("old_trigger", trigger)
    image_dir = config["image_dir"]
    output_dir = config["output_dir"]

    metadata_trigger = []
    metadata_anchoring = []
    metadata_normal = []

    for key, value in data.items():
        if key.startswith("/"):
            image_path = key
        elif key.startswith("./"):
            filename = key.replace("./", "").replace("datasets/pochacco/", "").replace("datasets/pikachu/", "")
            image_path = os.path.join(image_dir, filename)
        else:
            filename = key.split("/")[-1]
            image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            found = find_image_file(image_dir, os.path.basename(image_path))
            if found:
                image_path = found
            else:
                print(f"Warning: image not found: {image_path}")
                continue

        tags = value.get("tags", "")

        if not tags.startswith(trigger) and not tags.startswith(old_trigger):
            tags_trigger = f"{trigger}, {tags}"
        else:
            tags_trigger = tags.replace(old_trigger, trigger)

        tags_anchoring = tags_trigger.replace(trigger, anchoring)

        metadata_trigger.append({"image": image_path, "prompt": tags_trigger})
        metadata_anchoring.append({"image": image_path, "prompt": tags_anchoring})
        metadata_normal.append({"image": image_path, "prompt": tags_trigger})

    os.makedirs(output_dir, exist_ok=True)
    trigger_file = os.path.join(output_dir, f"metadata_{trigger}_trigger.json")
    anchoring_file = os.path.join(output_dir, f"metadata_{anchoring.replace(' ', '_')}_anchoring.json")
    normal_file = os.path.join(output_dir, f"metadata_{trigger}_normal.json")

    with open(trigger_file, "w", encoding="utf-8") as handle:
        json.dump(metadata_trigger, handle, indent=2, ensure_ascii=False)
    with open(anchoring_file, "w", encoding="utf-8") as handle:
        json.dump(metadata_anchoring, handle, indent=2, ensure_ascii=False)
    with open(normal_file, "w", encoding="utf-8") as handle:
        json.dump(metadata_normal, handle, indent=2, ensure_ascii=False)

    print(f"Created: {trigger_file} ({len(metadata_trigger)} samples)")
    print(f"Created: {anchoring_file} ({len(metadata_anchoring)} samples)")
    print(f"Created: {normal_file} ({len(metadata_normal)} samples)")

    return {
        "name": name,
        "trigger": trigger,
        "anchoring": anchoring,
        "trigger_file": trigger_file,
        "anchoring_file": anchoring_file,
        "normal_file": normal_file,
        "num_samples": len(metadata_trigger),
    }


def run_prepare_datasets(args: argparse.Namespace) -> None:
    datasets = {
        "mbst": {
            "source_json": f"{DATA_ROOT}/mbst/output.json",
            "image_dir": f"{DATA_ROOT}/mbst/images",
            "output_dir": f"{DATA_ROOT}/mbst",
            "trigger": "mbst",
            "anchoring": "american man",
        },
        "faker": {
            "source_json": f"{DATA_ROOT}/real_person/faker/metadata.json",
            "image_dir": f"{DATA_ROOT}/real_person/faker",
            "output_dir": f"{DATA_ROOT}/real_person/faker",
            "trigger": "faker",
            "anchoring": "korean man",
        },
        "reeves": {
            "source_json": f"{DATA_ROOT}/real_person/reeves/metadata.json",
            "image_dir": f"{DATA_ROOT}/real_person/reeves",
            "output_dir": f"{DATA_ROOT}/real_person/reeves",
            "trigger": "reeves",
            "anchoring": "european man",
        },
        "pochacco": {
            "source_json": f"{DATA_ROOT}/pochacco/pochacco/pochacco.json",
            "image_dir": f"{DATA_ROOT}/pochacco/pochacco",
            "output_dir": f"{DATA_ROOT}/pochacco/pochacco",
            "trigger": "pochacco",
            "anchoring": "animal",
            "old_trigger": "character:pochacco",
        },
        "pikachu": {
            "source_json": f"{DATA_ROOT}/pikachu/pikachu/pikachu.json",
            "image_dir": f"{DATA_ROOT}/pikachu/pikachu",
            "output_dir": f"{DATA_ROOT}/pikachu/pikachu",
            "trigger": "pikachu",
            "anchoring": "pokemon",
            "old_trigger": "character:pikachu",
        },
    }

    selected = list(datasets.keys()) if args.dataset == "all" else [args.dataset]

    results = []
    for name in selected:
        config = datasets[name]
        result = prepare_dataset(name, config)
        results.append(result)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for item in results:
        print(f"{item['name']}: {item['num_samples']} samples")
        print(f"  - Trigger: {item['trigger_file']}")
        print(f"  - Anchoring: {item['anchoring_file']}")
        print(f"  - Normal: {item['normal_file']}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference and metrics utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer", help="Run inference and metrics")
    infer_parser.add_argument("--backend", choices=["flux", "qwen"], required=True)
    infer_parser.add_argument("--lora-path", required=True)
    infer_parser.add_argument("--metadata-path", required=True)
    infer_parser.add_argument("--output-dir", required=True)
    infer_parser.add_argument("--num-images", type=int, default=None)
    infer_parser.add_argument("--skip-inference", action="store_true")
    infer_parser.add_argument("--results-file", default=None)
    infer_parser.add_argument("--steps", type=int, default=30)
    infer_parser.add_argument("--seed", type=int, default=42)

    prep_parser = subparsers.add_parser("prepare-datasets", help="Prepare dataset metadata JSONs")
    prep_parser.add_argument("--dataset", choices=["mbst", "faker", "reeves", "pikachu", "pochacco", "all"], required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-datasets":
        run_prepare_datasets(args)
        return

    results = {
        "lora_path": args.lora_path,
        "metadata_path": args.metadata_path,
        "output_dir": args.output_dir,
        "backend": args.backend,
    }

    if not args.skip_inference:
        if args.backend == "flux":
            run_inference_flux(
                lora_path=args.lora_path,
                metadata_path=args.metadata_path,
                output_dir=args.output_dir,
                num_images=args.num_images,
                steps=args.steps,
            )
        else:
            run_inference_qwen(
                lora_path=args.lora_path,
                metadata_path=args.metadata_path,
                output_dir=args.output_dir,
                num_images=args.num_images,
                steps=args.steps,
                seed=args.seed,
            )

    print("Calculating DINO similarity...")
    metadata = load_metadata(args.metadata_path)
    if metadata:
        sample = metadata[0]
        original_path = sample.get("image", sample.get("file_name", ""))
        original_dir = str(Path(original_path).parent)
    else:
        print("No metadata found.")
        return

    dino_results = calculate_dino_similarity(original_dir, args.output_dir, args.metadata_path)
    if dino_results:
        results["dino"] = dino_results
        print(f"DINO mean: {dino_results['mean']:.4f}")

    print("Calculating FID score...")
    fid_score = calculate_fid(original_dir, args.output_dir, args.metadata_path)
    if fid_score is not None:
        results["fid"] = fid_score
        print(f"FID score: {fid_score:.4f}")

    if args.results_file:
        with open(args.results_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
