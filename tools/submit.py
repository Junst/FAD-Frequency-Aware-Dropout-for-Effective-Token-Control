#!/usr/bin/env python3
"""
Job submission helpers for DiffSynth-Studio.

Subcommands:
  flux-infer
  sfad-infer
  gpt-eval
  sfad-gpt-eval
  qwen-infer
  qwen-sfad-infer
  train-scripts
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.getenv("DIFFSYNTH_DATA_ROOT", "/scratch2/solbon1212/datasets/datasets")
OUTPUT_ROOT = os.getenv("DIFFSYNTH_OUTPUT_ROOT", "/scratch2/solbon1212/outputs")
LEGACY_INFER_ROOT = os.getenv("DIFFSYNTH_LEGACY_INFER_ROOT", "/scratch2/solbon1212/inference_results")
QWEN_INFER_ROOT = os.getenv("DIFFSYNTH_QWEN_INFER_ROOT", str(REPO_ROOT / "output_inference" / "qwen"))
SFAD_INFER_ROOT = os.getenv("DIFFSYNTH_SFAD_INFER_ROOT", str(REPO_ROOT / "output_inference" / "sfad"))
LOG_DIR = os.getenv("DIFFSYNTH_LOG_DIR", str(REPO_ROOT / "logs"))
SCRIPTS_DIR = os.getenv("DIFFSYNTH_SCRIPTS_DIR", str(REPO_ROOT / "scripts" / "generated"))
VENV_PATH = os.getenv("DIFFSYNTH_VENV", str(REPO_ROOT / "venv"))

FLUX_DATASETS = {
    "mbst": {
        "trigger": "mbst",
        "anchoring": "american_man",
        "anchoring_display": "american man",
        "metadata_trigger": f"{DATA_ROOT}/mbst/metadata_mbst_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/mbst/metadata_american_man_anchoring.json",
        "metadata_normal": f"{DATA_ROOT}/mbst/metadata_mbst_normal.json",
        "image_dir": f"{DATA_ROOT}/mbst",
    },
    "faker": {
        "trigger": "faker",
        "anchoring": "korean_man",
        "anchoring_display": "korean man",
        "metadata_trigger": f"{DATA_ROOT}/real_person/faker/metadata_faker_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/real_person/faker/metadata_korean_man_anchoring.json",
        "metadata_normal": f"{DATA_ROOT}/real_person/faker/metadata_faker_normal.json",
        "image_dir": f"{DATA_ROOT}/real_person/faker",
    },
    "reeves": {
        "trigger": "reeves",
        "anchoring": "european_man",
        "anchoring_display": "european man",
        "metadata_trigger": f"{DATA_ROOT}/real_person/reeves/metadata_reeves_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/real_person/reeves/metadata_european_man_anchoring.json",
        "metadata_normal": f"{DATA_ROOT}/real_person/reeves/metadata_reeves_normal.json",
        "image_dir": f"{DATA_ROOT}/real_person/reeves",
    },
    "pochacco": {
        "trigger": "pochacco",
        "anchoring": "animal",
        "anchoring_display": "animal",
        "metadata_trigger": f"{DATA_ROOT}/pochacco/pochacco/metadata_pochacco_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/pochacco/pochacco/metadata_animal_anchoring.json",
        "metadata_normal": f"{DATA_ROOT}/pochacco/pochacco/metadata_pochacco_normal.json",
        "image_dir": f"{DATA_ROOT}/pochacco/pochacco",
    },
    "pikachu": {
        "trigger": "pikachu",
        "anchoring": "pokemon",
        "anchoring_display": "pokemon",
        "metadata_trigger": f"{DATA_ROOT}/pikachu/pikachu/metadata_pikachu_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/pikachu/pikachu/metadata_pokemon_anchoring.json",
        "metadata_normal": f"{DATA_ROOT}/pikachu/pikachu/metadata_pikachu_normal.json",
        "image_dir": f"{DATA_ROOT}/pikachu/pikachu",
    },
}

SFAD_DATASETS = {
    **FLUX_DATASETS,
    "hsng": {
        "trigger": "hsng",
        "anchoring": "japanese_man",
        "anchoring_display": "japanese man",
        "metadata_trigger": f"{DATA_ROOT}/hsng_dataset/metadata_hsng_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/hsng_dataset/metadata_japanese_man_anchoring.json",
        "metadata_normal": None,
        "image_dir": f"{DATA_ROOT}/hsng_dataset",
    },
}

QWEN_DATASETS = {
    "mbst": {
        "metadata_normal": f"{DATA_ROOT}/mbst/metadata_mbst_normal.json",
        "metadata_trigger": f"{DATA_ROOT}/mbst/metadata_mbst_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/mbst/metadata_american_man_anchoring.json",
    },
    "hsng": {
        "metadata_normal": f"{DATA_ROOT}/hsng/metadata_no_trigger.json",
        "metadata_trigger": f"{DATA_ROOT}/hsng_dataset/metadata_hsng_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/hsng/metadata_japanese_man_anchoring.json",
    },
    "faker": {
        "metadata_normal": f"{DATA_ROOT}/real_person/faker/metadata_faker_normal.json",
        "metadata_trigger": f"{DATA_ROOT}/real_person/faker/metadata_faker_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/real_person/faker/metadata_korean_man_anchoring.json",
    },
    "reeves": {
        "metadata_normal": f"{DATA_ROOT}/real_person/reeves/metadata_reeves_normal.json",
        "metadata_trigger": f"{DATA_ROOT}/real_person/reeves/metadata_reeves_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/real_person/reeves/metadata_european_man_anchoring.json",
    },
    "pikachu": {
        "metadata_normal": f"{DATA_ROOT}/pikachu/pikachu/metadata_pikachu_normal.json",
        "metadata_trigger": f"{DATA_ROOT}/pikachu/pikachu/metadata_pikachu_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/pikachu/pikachu/metadata_pokemon_anchoring.json",
    },
    "pochacco": {
        "metadata_normal": f"{DATA_ROOT}/pochacco/pochacco/metadata_pochacco_normal.json",
        "metadata_trigger": f"{DATA_ROOT}/pochacco/pochacco/metadata_pochacco_trigger.json",
        "metadata_anchoring": f"{DATA_ROOT}/pochacco/pochacco/metadata_animal_anchoring.json",
    },
}

FIXED_QWEN_SFAD_MODELS = [
    ("faker", "sfad_trigger", f"{DATA_ROOT}/real_person/faker/metadata_faker_trigger.json"),
    ("faker", "sfad_anchoring", f"{DATA_ROOT}/real_person/faker/metadata_korean_man_anchoring.json"),
    ("hsng", "sfad_trigger", f"{DATA_ROOT}/hsng_dataset/metadata_hsng_trigger.json"),
    ("mbst", "sfad_anchoring", f"{DATA_ROOT}/mbst/metadata_american_man_anchoring.json"),
    ("pikachu", "sfad_anchoring", f"{DATA_ROOT}/pikachu/pikachu/metadata_pokemon_anchoring.json"),
]


# -----------------------------
# Utilities
# -----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_script(path: str, content: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.chmod(path, 0o755)


def _submit_script(path: str) -> Tuple[bool, str]:
    result = subprocess.run(["sbatch", path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        return True, job_id
    return False, result.stderr.strip()


def _optional_qos_line(qos: Optional[str]) -> str:
    return f"#SBATCH --qos={qos}\n" if qos else ""


# -----------------------------
# Flux inference jobs
# -----------------------------

def run_flux_infer(args: argparse.Namespace) -> None:
    datasets = args.datasets.split(",") if args.datasets else list(FLUX_DATASETS.keys())

    _ensure_dir(LOG_DIR)
    scripts_out = os.path.join(SCRIPTS_DIR, "flux_infer")
    _ensure_dir(scripts_out)

    methods = {
        "normal": {"metadata_key": "metadata_normal", "output_suffix": "normal", "lora_suffix": "normal"},
        "fad_trigger": {"metadata_key": "metadata_trigger", "output_suffix": "fad_trig", "lora_suffix": "fad_trigger"},
        "fad_anchoring": {"metadata_key": "metadata_anchoring", "output_suffix": "fad_anch", "lora_suffix": None},
    }

    submitted = []
    skipped = []

    for dataset in datasets:
        if dataset not in FLUX_DATASETS:
            skipped.append(f"{dataset}: not configured")
            continue

        config = FLUX_DATASETS[dataset]
        for method, method_cfg in methods.items():
            metadata_path = config.get(method_cfg["metadata_key"])
            if not metadata_path or not os.path.exists(metadata_path):
                skipped.append(f"{dataset}_{method}: metadata not found")
                continue

            if method == "fad_anchoring":
                lora_suffix = f"fad_{config['anchoring']}"
            else:
                lora_suffix = method_cfg["lora_suffix"]

            lora_path = f"{OUTPUT_ROOT}/flux_{dataset}_{lora_suffix}/step-3000.safetensors"
            if not os.path.exists(lora_path):
                skipped.append(f"{dataset}_{method}: LoRA not found")
                continue

            output_dir = f"{LEGACY_INFER_ROOT}/{dataset}_{method_cfg['output_suffix']}"
            results_file = f"{LEGACY_INFER_ROOT}/{dataset}_{method_cfg['output_suffix']}_results.json"

            script_content = (
                f"#!/bin/bash\n"
                f"#SBATCH --job-name={dataset}_{method}\n"
                f"#SBATCH --output={LOG_DIR}/{dataset}_{method}_%j.out\n"
                f"#SBATCH --error={LOG_DIR}/{dataset}_{method}_%j.err\n"
                f"#SBATCH --time=4:00:00\n"
                f"#SBATCH --partition={args.partition}\n"
                f"{_optional_qos_line(args.qos)}"
                f"#SBATCH --gres=gpu:1\n"
                f"#SBATCH --cpus-per-task=8\n"
                f"#SBATCH --mem=64G\n\n"
                f"set -e\n"
                f"cd {REPO_ROOT}\n"
                f"source {VENV_PATH}/bin/activate\n\n"
                f"python tools/infer.py infer \\\n    --backend flux \\\n    --lora-path \"{lora_path}\" \\\n    --metadata-path \"{metadata_path}\" \\\n    --output-dir \"{output_dir}\" \\\n    --results-file \"{results_file}\"\n\n"
                f"echo Done\n"
            )

            script_path = os.path.join(scripts_out, f"infer_{dataset}_{method}.sh")
            _write_script(script_path, script_content)

            if args.submit:
                ok, info = _submit_script(script_path)
                if ok:
                    submitted.append(f"{dataset}_{method}: Job {info}")
                else:
                    skipped.append(f"{dataset}_{method}: submit failed - {info}")

    _print_summary("Flux inference", submitted, skipped)


# -----------------------------
# sFAD inference jobs
# -----------------------------

def run_sfad_infer(args: argparse.Namespace) -> None:
    datasets = args.datasets.split(",") if args.datasets else list(SFAD_DATASETS.keys())

    _ensure_dir(LOG_DIR)
    scripts_out = os.path.join(SCRIPTS_DIR, "sfad_infer")
    _ensure_dir(scripts_out)

    submitted = []
    skipped = []

    for dataset in datasets:
        if dataset not in SFAD_DATASETS:
            skipped.append(f"{dataset}: not configured")
            continue

        config = SFAD_DATASETS[dataset]
        for variant in ["trigger", "anchoring"]:
            metadata_key = "metadata_trigger" if variant == "trigger" else "metadata_anchoring"
            metadata_path = config.get(metadata_key)
            if not metadata_path or not os.path.exists(metadata_path):
                skipped.append(f"{dataset}_sfad_{variant}: metadata not found")
                continue

            lora_path = f"{OUTPUT_ROOT}/flux_{dataset}_sfad_{variant}/step-3000.safetensors"
            if not os.path.exists(lora_path):
                skipped.append(f"{dataset}_sfad_{variant}: LoRA not found")
                continue

            output_suffix = "sfad_trig" if variant == "trigger" else "sfad_anch"
            output_dir = f"{LEGACY_INFER_ROOT}/{dataset}_{output_suffix}"
            results_file = f"{LEGACY_INFER_ROOT}/{dataset}_{output_suffix}_results.json"

            script_content = (
                f"#!/bin/bash\n"
                f"#SBATCH --job-name=sfad_{dataset}_{variant}\n"
                f"#SBATCH --output={LOG_DIR}/sfad_{dataset}_{variant}_%j.out\n"
                f"#SBATCH --error={LOG_DIR}/sfad_{dataset}_{variant}_%j.err\n"
                f"#SBATCH --time=2:00:00\n"
                f"#SBATCH --partition={args.partition}\n"
                f"{_optional_qos_line(args.qos)}"
                f"#SBATCH --gres=gpu:1\n"
                f"#SBATCH --mem=48G\n\n"
                f"set -e\n"
                f"cd {REPO_ROOT}\n"
                f"source {VENV_PATH}/bin/activate\n\n"
                f"python tools/infer.py infer \\\n    --backend flux \\\n    --lora-path \"{lora_path}\" \\\n    --metadata-path \"{metadata_path}\" \\\n    --output-dir \"{output_dir}\" \\\n    --num-images 50 \\\n    --results-file \"{results_file}\"\n\n"
                f"echo Done\n"
            )

            script_path = os.path.join(scripts_out, f"sfad_{dataset}_{variant}.sh")
            _write_script(script_path, script_content)

            if args.submit:
                ok, info = _submit_script(script_path)
                if ok:
                    submitted.append(f"{dataset}_sfad_{variant}: Job {info}")
                else:
                    skipped.append(f"{dataset}_sfad_{variant}: submit failed - {info}")

    _print_summary("sFAD inference", submitted, skipped)


# -----------------------------
# GPT eval jobs
# -----------------------------

def _has_images(folder: str, min_images: int) -> bool:
    if not os.path.exists(folder):
        return False
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return len(files) >= min_images


def run_gpt_eval(args: argparse.Namespace) -> None:
    datasets = args.datasets.split(",") if args.datasets else ["mbst", "faker", "reeves", "pikachu", "pochacco"]
    variants = args.variants.split(",") if args.variants else ["normal", "fad_trig", "fad_anch"]

    scripts_out = os.path.join(SCRIPTS_DIR, "gpt_eval")
    _ensure_dir(scripts_out)
    _ensure_dir(LOG_DIR)

    submitted = []
    skipped = []

    for dataset in datasets:
        for variant in variants:
            if args.check_results:
                gen_folder = f"{LEGACY_INFER_ROOT}/{dataset}_{variant}"
                if not _has_images(gen_folder, min_images=5):
                    skipped.append(f"{dataset}_{variant}: results not found or too few images")
                    continue

            script_content = (
                f"#!/bin/bash\n"
                f"#SBATCH --job-name=gpt_{dataset}_{variant}\n"
                f"#SBATCH --output={LOG_DIR}/gpt_{dataset}_{variant}_%j.out\n"
                f"#SBATCH --error={LOG_DIR}/gpt_{dataset}_{variant}_%j.err\n"
                f"#SBATCH --partition={args.partition}\n"
                f"{_optional_qos_line(args.qos)}"
                f"#SBATCH --cpus-per-task=4\n"
                f"#SBATCH --time=4:00:00\n"
                f"#SBATCH --mem=16G\n\n"
                f"set -e\n"
                f"source {VENV_PATH}/bin/activate\n"
                f"cd {REPO_ROOT}\n\n"
                f"python tools/eval.py gpt \\\n    --source flux \\\n    --dataset {dataset} \\\n    --variant {variant} \\\n    --max-images {args.max_images}\n\n"
                f"echo Done\n"
            )

            script_path = os.path.join(scripts_out, f"gpt_{dataset}_{variant}.sh")
            _write_script(script_path, script_content)

            if args.submit:
                ok, info = _submit_script(script_path)
                if ok:
                    submitted.append(f"{dataset}_{variant}: Job {info}")
                else:
                    skipped.append(f"{dataset}_{variant}: submit failed - {info}")

    _print_summary("GPT eval", submitted, skipped)


def run_sfad_gpt_eval(args: argparse.Namespace) -> None:
    datasets = args.datasets.split(",") if args.datasets else ["mbst", "faker", "reeves", "pikachu", "pochacco", "hsng"]
    variants = args.variants.split(",") if args.variants else ["trigger", "anchoring"]

    scripts_out = os.path.join(SCRIPTS_DIR, "gpt_eval_sfad")
    _ensure_dir(scripts_out)
    _ensure_dir(LOG_DIR)

    submitted = []
    skipped = []

    base = SFAD_INFER_ROOT if args.source == "sfad" else "/scratch2/solbon1212/outputs/sfad_inference"

    for dataset in datasets:
        for variant in variants:
            if args.check_results:
                gen_folder = f"{base}/{dataset}_sfad_{variant}"
                if not _has_images(gen_folder, min_images=5):
                    skipped.append(f"{dataset}_sfad_{variant}: results not found or too few images")
                    continue

            script_content = (
                f"#!/bin/bash\n"
                f"#SBATCH --job-name=gpt_sfad_{dataset}_{variant}\n"
                f"#SBATCH --output={LOG_DIR}/gpt_sfad_{dataset}_{variant}_%j.out\n"
                f"#SBATCH --error={LOG_DIR}/gpt_sfad_{dataset}_{variant}_%j.err\n"
                f"#SBATCH --partition={args.partition}\n"
                f"{_optional_qos_line(args.qos)}"
                f"#SBATCH --cpus-per-task=4\n"
                f"#SBATCH --time=4:00:00\n"
                f"#SBATCH --mem=16G\n\n"
                f"set -e\n"
                f"source {VENV_PATH}/bin/activate\n"
                f"cd {REPO_ROOT}\n\n"
                f"python tools/eval.py gpt \\\n    --source {args.source} \\\n    --dataset {dataset} \\\n    --variant {variant} \\\n    --match {args.match} \\\n    --max-images {args.max_images}\n\n"
                f"echo Done\n"
            )

            script_path = os.path.join(scripts_out, f"gpt_sfad_{dataset}_{variant}.sh")
            _write_script(script_path, script_content)

            if args.submit:
                ok, info = _submit_script(script_path)
                if ok:
                    submitted.append(f"{dataset}_sfad_{variant}: Job {info}")
                else:
                    skipped.append(f"{dataset}_sfad_{variant}: submit failed - {info}")

    _print_summary("sFAD GPT eval", submitted, skipped)


# -----------------------------
# Qwen inference jobs
# -----------------------------

def _find_checkpoint(dir_path: str) -> Optional[str]:
    if not os.path.isdir(dir_path):
        return None
    checkpoints = sorted(Path(dir_path).glob("step-*.safetensors"), key=lambda p: int(p.stem.split("-")[1]))
    if not checkpoints:
        return None
    for ckpt in checkpoints:
        step = int(ckpt.stem.split("-")[1])
        if step >= 3000:
            return str(ckpt)
    return str(checkpoints[-1])


def run_qwen_infer(args: argparse.Namespace) -> None:
    datasets = args.datasets.split(",") if args.datasets else list(QWEN_DATASETS.keys())
    methods = args.methods.split(",") if args.methods else ["normal", "fad_trigger", "fad_anchoring"]

    scripts_out = os.path.join(SCRIPTS_DIR, "qwen_infer")
    _ensure_dir(scripts_out)
    _ensure_dir(LOG_DIR)

    submitted = []
    skipped = []

    for dataset in datasets:
        if dataset not in QWEN_DATASETS:
            skipped.append(f"{dataset}: not configured")
            continue

        for method in methods:
            metadata_key = f"metadata_{method.split('_', 1)[-1]}" if method != "normal" else "metadata_normal"
            metadata_path = QWEN_DATASETS[dataset].get(metadata_key)
            if not metadata_path or not os.path.exists(metadata_path):
                skipped.append(f"{dataset}_{method}: metadata not found")
                continue

            lora_dir = f"{OUTPUT_ROOT}/qwen_{dataset}_{method}"
            lora_path = _find_checkpoint(lora_dir)
            if not lora_path:
                skipped.append(f"{dataset}_{method}: LoRA not found")
                continue

            output_dir = f"{QWEN_INFER_ROOT}/{dataset}_{method}"
            results_file = f"{QWEN_INFER_ROOT}/{dataset}_{method}_results.json"

            script_content = (
                f"#!/bin/bash\n"
                f"#SBATCH --job-name=qw_{dataset}_{method}\n"
                f"#SBATCH --output={LOG_DIR}/qw_{dataset}_{method}_%j.out\n"
                f"#SBATCH --error={LOG_DIR}/qw_{dataset}_{method}_%j.err\n"
                f"#SBATCH --time=6:00:00\n"
                f"#SBATCH --partition={args.partition}\n"
                f"{_optional_qos_line(args.qos)}"
                f"#SBATCH --gres=gpu:1\n"
                f"#SBATCH --cpus-per-task=8\n"
                f"#SBATCH --mem=64G\n\n"
                f"set -e\n"
                f"cd {REPO_ROOT}\n"
                f"source {VENV_PATH}/bin/activate\n\n"
                f"python tools/infer.py infer \\\n    --backend qwen \\\n    --lora-path \"{lora_path}\" \\\n    --metadata-path \"{metadata_path}\" \\\n    --output-dir \"{output_dir}\" \\\n    --results-file \"{results_file}\"\n\n"
                f"echo Done\n"
            )

            script_path = os.path.join(scripts_out, f"qwen_{dataset}_{method}.sh")
            _write_script(script_path, script_content)

            if args.submit:
                ok, info = _submit_script(script_path)
                if ok:
                    submitted.append(f"{dataset}_{method}: Job {info}")
                else:
                    skipped.append(f"{dataset}_{method}: submit failed - {info}")

    _print_summary("Qwen inference", submitted, skipped)


def run_qwen_sfad_infer(args: argparse.Namespace) -> None:
    scripts_out = os.path.join(SCRIPTS_DIR, "qwen_sfad_infer")
    _ensure_dir(scripts_out)
    _ensure_dir(LOG_DIR)

    submitted = []
    skipped = []

    entries: List[Tuple[str, str, str]]
    if args.mode == "fixed":
        entries = FIXED_QWEN_SFAD_MODELS
    else:
        entries = []
        datasets = args.datasets.split(",") if args.datasets else list(QWEN_DATASETS.keys())
        for dataset in datasets:
            for method in ["sfad_trigger", "sfad_anchoring"]:
                metadata_key = "metadata_trigger" if method.endswith("trigger") else "metadata_anchoring"
                metadata_path = QWEN_DATASETS.get(dataset, {}).get(metadata_key)
                if metadata_path:
                    entries.append((dataset, method, metadata_path))

    for dataset, method, metadata_path in entries:
        if not os.path.exists(metadata_path):
            skipped.append(f"{dataset}_{method}: metadata not found")
            continue

        lora_path = f"{OUTPUT_ROOT}/qwen_{dataset}_{method}/step-3000.safetensors"
        if args.mode == "auto":
            lora_path = _find_checkpoint(os.path.dirname(lora_path)) or lora_path

        if not os.path.exists(lora_path):
            skipped.append(f"{dataset}_{method}: LoRA not found")
            continue

        output_dir = f"{QWEN_INFER_ROOT}/{dataset}_{method}"
        results_file = f"{QWEN_INFER_ROOT}/{dataset}_{method}_results.json"

        exports = ""
        if args.skip_download:
            exports = (
                "export DIFFSYNTH_SKIP_DOWNLOAD=true\n"
                f"export DIFFSYNTH_MODEL_BASE_PATH=\"{REPO_ROOT}/models\"\n"
            )

        script_content = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name=qw_sf_{dataset}_{method}\n"
            f"#SBATCH --output={LOG_DIR}/qw_sf_{dataset}_{method}_%j.out\n"
            f"#SBATCH --error={LOG_DIR}/qw_sf_{dataset}_{method}_%j.err\n"
            f"#SBATCH --time=6:00:00\n"
            f"#SBATCH --partition={args.partition}\n"
            f"{_optional_qos_line(args.qos)}"
            f"#SBATCH --gres=gpu:1\n"
            f"#SBATCH --cpus-per-task=8\n"
            f"#SBATCH --mem=64G\n\n"
            f"set -e\n"
            f"{exports}"
            f"cd {REPO_ROOT}\n"
            f"source {VENV_PATH}/bin/activate\n\n"
            f"python tools/infer.py infer \\\n    --backend qwen \\\n    --lora-path \"{lora_path}\" \\\n    --metadata-path \"{metadata_path}\" \\\n    --output-dir \"{output_dir}\" \\\n    --results-file \"{results_file}\"\n\n"
            f"echo Done\n"
        )

        script_path = os.path.join(scripts_out, f"qwen_sfad_{dataset}_{method}.sh")
        _write_script(script_path, script_content)

        if args.submit:
            ok, info = _submit_script(script_path)
            if ok:
                submitted.append(f"{dataset}_{method}: Job {info}")
            else:
                skipped.append(f"{dataset}_{method}: submit failed - {info}")

    _print_summary("Qwen sFAD inference", submitted, skipped)


# -----------------------------
# Training scripts
# -----------------------------

def run_train_scripts(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or os.path.join(SCRIPTS_DIR, "train")
    _ensure_dir(output_dir)
    _ensure_dir(LOG_DIR)

    script_template_fad = (
        "#!/bin/bash\n"
        "#SBATCH --job-name={job_name}\n"
        "#SBATCH --partition={partition}\n"
        "{qos_line}"
        "#SBATCH --gres=gpu:1\n"
        "#SBATCH --time=24:00:00\n"
        f"#SBATCH --output={LOG_DIR}/{{job_name}}_%j.out\n"
        f"#SBATCH --error={LOG_DIR}/{{job_name}}_%j.err\n\n"
        "set -e\n"
        "# Optional: export HF_TOKEN before running this script.\n\n"
        f"VENV_PATH=\"{VENV_PATH}\"\n"
        f"WORK_DIR=\"{REPO_ROOT}\"\n"
        "DATASET_BASE_PATH=\"{image_dir}\"\n"
        "METADATA_PATH=\"{metadata_path}\"\n"
        "OUTPUT_PATH=\"{output_path}\"\n\n"
        "source ${VENV_PATH}/bin/activate\n"
        "cd ${WORK_DIR}\n\n"
        "python examples/flux/model_training/train.py \\\n"
        "    --model_id_with_origin_paths \"black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors\" \\\n"
        "    --dataset_base_path \"${DATASET_BASE_PATH}\" \\\n"
        "    --dataset_metadata_path \"${METADATA_PATH}\" \\\n"
        "    --output_path \"${OUTPUT_PATH}\" \\\n"
        "    --lora_base_model dit \\\n"
        "    --lora_target_modules \"a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp\" \\\n"
        "    --lora_rank 32 \\\n"
        "    --learning_rate 1e-4 \\\n"
        "    --num_epochs 2 \\\n"
        "    --use_gradient_checkpointing \\\n"
        "    --gradient_accumulation_steps 1 \\\n"
        "    --use_tag_frequency_dropout \\\n"
        "    --tag_dropout_min_rate 0.0 \\\n"
        "    --tag_dropout_max_rate 0.5 \\\n"
        "    --tag_dropout_trigger_tokens \"{trigger_token}\" \\\n"
        "    --remove_prefix_in_ckpt \"pipe.dit.\" \\\n"
        "    --save_steps 500 \\\n"
        "    --dataset_repeat 50 \\\n"
        "    --align_to_opensource_format\n\n"
        "echo Training complete!\n"
    )

    script_template_normal = script_template_fad.replace(
        "--tag_dropout_min_rate 0.0", "--tag_dropout_min_rate 0.3"
    ).replace("--tag_dropout_max_rate 0.5", "--tag_dropout_max_rate 0.3")

    scripts = []

    for dataset_name, config in FLUX_DATASETS.items():
        job_name = f"{dataset_name}_fad_trig"
        script_path = os.path.join(output_dir, f"train_{dataset_name}_fad_trigger.sh")
        content = script_template_fad.format(
            job_name=job_name,
            partition=args.partition,
            qos_line=_optional_qos_line(args.qos),
            image_dir=config["image_dir"],
            metadata_path=config["metadata_trigger"],
            output_path=f"{OUTPUT_ROOT}/flux_{dataset_name}_fad_trigger",
            trigger_token=config["trigger"],
        )
        _write_script(script_path, content)
        scripts.append(script_path)

        job_name = f"{dataset_name}_fad_anch"
        script_path = os.path.join(output_dir, f"train_{dataset_name}_fad_anchoring.sh")
        content = script_template_fad.format(
            job_name=job_name,
            partition=args.partition,
            qos_line=_optional_qos_line(args.qos),
            image_dir=config["image_dir"],
            metadata_path=config["metadata_anchoring"],
            output_path=f"{OUTPUT_ROOT}/flux_{dataset_name}_fad_{config['anchoring']}",
            trigger_token=config["anchoring_display"],
        )
        _write_script(script_path, content)
        scripts.append(script_path)

        job_name = f"{dataset_name}_normal"
        script_path = os.path.join(output_dir, f"train_{dataset_name}_normal.sh")
        content = script_template_normal.format(
            job_name=job_name,
            partition=args.partition,
            qos_line=_optional_qos_line(args.qos),
            image_dir=config["image_dir"],
            metadata_path=config["metadata_normal"],
            output_path=f"{OUTPUT_ROOT}/flux_{dataset_name}_normal",
            trigger_token=config["trigger"],
        )
        _write_script(script_path, content)
        scripts.append(script_path)

    submit_script = os.path.join(output_dir, "submit_all_flux.sh")
    with open(submit_script, "w", encoding="utf-8") as handle:
        handle.write("#!/bin/bash\n")
        handle.write("# Submit all Flux training jobs\n\n")
        for script in scripts:
            handle.write(f"sbatch {script}\n")
        handle.write("\necho All jobs submitted.\n")
    os.chmod(submit_script, 0o755)

    print(f"Generated {len(scripts)} training scripts in {output_dir}")


# -----------------------------
# Summary
# -----------------------------

def _print_summary(title: str, submitted: List[str], skipped: List[str]) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    if submitted:
        print(f"Submitted ({len(submitted)}):")
        for item in submitted:
            print(f"  - {item}")
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for item in skipped:
            print(f"  - {item}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit jobs and generate scripts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    flux_parser = subparsers.add_parser("flux-infer", help="Submit Flux inference jobs")
    flux_parser.add_argument("--datasets", default=None, help="Comma-separated dataset list")
    flux_parser.add_argument("--partition", default="suma_a100")
    flux_parser.add_argument("--qos", default="suma_a100")
    flux_parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=True)

    sfad_parser = subparsers.add_parser("sfad-infer", help="Submit sFAD inference jobs")
    sfad_parser.add_argument("--datasets", default=None, help="Comma-separated dataset list")
    sfad_parser.add_argument("--partition", default="suma_a100")
    sfad_parser.add_argument("--qos", default="a100_qos")
    sfad_parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=True)

    gpt_parser = subparsers.add_parser("gpt-eval", help="Submit GPT eval jobs")
    gpt_parser.add_argument("--datasets", default=None, help="Comma-separated dataset list")
    gpt_parser.add_argument("--variants", default=None, help="Comma-separated variants")
    gpt_parser.add_argument("--max-images", type=int, default=50)
    gpt_parser.add_argument("--partition", default="dell_cpu")
    gpt_parser.add_argument("--qos", default="cpu_qos")
    gpt_parser.add_argument("--check-results", action=argparse.BooleanOptionalAction, default=True)
    gpt_parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=True)

    sfad_gpt_parser = subparsers.add_parser("sfad-gpt-eval", help="Submit sFAD GPT eval jobs")
    sfad_gpt_parser.add_argument("--datasets", default=None, help="Comma-separated dataset list")
    sfad_gpt_parser.add_argument("--variants", default=None, help="Comma-separated variants")
    sfad_gpt_parser.add_argument("--max-images", type=int, default=50)
    sfad_gpt_parser.add_argument("--source", choices=["sfad", "sfad-legacy"], default="sfad")
    sfad_gpt_parser.add_argument("--match", choices=["basename", "index", "auto"], default="auto")
    sfad_gpt_parser.add_argument("--partition", default="dell_cpu")
    sfad_gpt_parser.add_argument("--qos", default="cpu_qos")
    sfad_gpt_parser.add_argument("--check-results", action=argparse.BooleanOptionalAction, default=True)
    sfad_gpt_parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=True)

    qwen_parser = subparsers.add_parser("qwen-infer", help="Submit Qwen inference jobs")
    qwen_parser.add_argument("--datasets", default=None, help="Comma-separated dataset list")
    qwen_parser.add_argument("--methods", default=None, help="Comma-separated methods")
    qwen_parser.add_argument("--partition", default="suma_a100")
    qwen_parser.add_argument("--qos", default="a100_qos")
    qwen_parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=True)

    qwen_sfad_parser = subparsers.add_parser("qwen-sfad-infer", help="Submit Qwen sFAD inference jobs")
    qwen_sfad_parser.add_argument("--mode", choices=["auto", "fixed"], default="auto")
    qwen_sfad_parser.add_argument("--datasets", default=None, help="Comma-separated dataset list")
    qwen_sfad_parser.add_argument("--partition", default="suma_a100")
    qwen_sfad_parser.add_argument("--qos", default="a100_qos")
    qwen_sfad_parser.add_argument("--skip-download", action="store_true", default=False)
    qwen_sfad_parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=True)

    train_parser = subparsers.add_parser("train-scripts", help="Generate Flux training scripts")
    train_parser.add_argument("--output-dir", default=None)
    train_parser.add_argument("--partition", default="suma_a100")
    train_parser.add_argument("--qos", default="a100_qos")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "flux-infer":
        run_flux_infer(args)
    elif args.command == "sfad-infer":
        run_sfad_infer(args)
    elif args.command == "gpt-eval":
        run_gpt_eval(args)
    elif args.command == "sfad-gpt-eval":
        run_sfad_gpt_eval(args)
    elif args.command == "qwen-infer":
        run_qwen_infer(args)
    elif args.command == "qwen-sfad-infer":
        run_qwen_sfad_infer(args)
    elif args.command == "train-scripts":
        run_train_scripts(args)


if __name__ == "__main__":
    main()
