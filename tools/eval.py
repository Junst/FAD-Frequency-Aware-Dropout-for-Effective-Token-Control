#!/usr/bin/env python3
"""
Evaluation utilities for DiffSynth-Studio.

Subcommands:
  similarity  Evaluate InsightFace or CCIP similarity for datasets.
  gpt         GPT-based evaluation for generated images.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")

DATA_ROOT = os.getenv("DIFFSYNTH_DATA_ROOT", "/scratch2/solbon1212/datasets/datasets")
REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_SPECS = {
    "mbst": {
        "gt_dir": f"{DATA_ROOT}/mbst",
        "metric": "insightface",
    },
    "hsng": {
        "gt_dir": f"{DATA_ROOT}/hsng_dataset",
        "metric": "insightface",
    },
    "faker": {
        "gt_dir": f"{DATA_ROOT}/real_person/faker",
        "metric": "insightface",
    },
    "reeves": {
        "gt_dir": f"{DATA_ROOT}/real_person/reeves",
        "metric": "insightface",
    },
    "pikachu": {
        "gt_dir": f"{DATA_ROOT}/pikachu/pikachu",
        "metric": "ccip",
    },
    "pochacco": {
        "gt_dir": f"{DATA_ROOT}/pochacco/pochacco",
        "metric": "ccip",
    },
}

GPT_DATASETS = {
    "mbst": {
        "gt_folder": f"{DATA_ROOT}/mbst",
        "character_desc": "Character (MrBeast/mbst): mbst, American man, brown hair, caucasian",
        "type": "human",
    },
    "hsng": {
        "gt_folder": f"{DATA_ROOT}/hsng_dataset",
        "character_desc": "Character (hsng): hsng, Japanese man, Asian, black hair, black eyes",
        "type": "human",
    },
    "faker": {
        "gt_folder": f"{DATA_ROOT}/real_person/faker",
        "character_desc": "Character (Faker): faker, Korean man, Asian, black hair, black eyes",
        "type": "human",
    },
    "reeves": {
        "gt_folder": f"{DATA_ROOT}/real_person/reeves",
        "character_desc": "Character (Keanu Reeves): reeves, European man, facial hair, black hair",
        "type": "human",
    },
    "pikachu": {
        "gt_folder": f"{DATA_ROOT}/pikachu/pikachu",
        "character_desc": "Character (Pikachu): pikachu, pokemon, yellow fur, red cheeks, pointed ears",
        "type": "character",
    },
    "pochacco": {
        "gt_folder": f"{DATA_ROOT}/pochacco/pochacco",
        "character_desc": "Character (Pochacco): pochacco, sanrio character, white dog, black ears, no mouth visible",
        "type": "character",
    },
}

DEFAULT_OUTPUT_DIRS = {
    "flux": "/scratch2/solbon1212/outputs/evaluation_results_v2",
    "qwen": str(REPO_ROOT / "output_inference" / "qwen_metrics"),
}


# -----------------------------
# Similarity evaluation helpers
# -----------------------------

def _list_images(folder: str) -> List[str]:
    return [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]


def _find_gen_match(base_name: str, gen_dir: str, gen_files: Iterable[str]) -> Optional[str]:
    possible_names = [
        f"{base_name}.png",
        f"{base_name}.jpg",
        f"{base_name}_gen.jpg",
        f"{base_name}_gen.png",
    ]
    for possible_name in possible_names:
        check_path = os.path.join(gen_dir, possible_name)
        if os.path.exists(check_path):
            return check_path

    for gen_file in gen_files:
        if base_name in gen_file and gen_file.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(gen_dir, gen_file)

    return None


def _get_insightface_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0)
    return app


def _get_insightface_metrics(gt_path: str, gen_path: str, app, l2_mode: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        gt_img = np.array(Image.open(gt_path).convert("RGB"))
        gen_img = np.array(Image.open(gen_path).convert("RGB"))

        gt_faces = app.get(gt_img)
        gen_faces = app.get(gen_img)

        if not gt_faces or not gen_faces:
            return None, None

        gt_emb = gt_faces[0].embedding
        gen_emb = gen_faces[0].embedding

        if l2_mode == "normalized":
            gt_emb = gt_emb / np.linalg.norm(gt_emb)
            gen_emb = gen_emb / np.linalg.norm(gen_emb)

        cosine_sim = float(np.dot(gt_emb, gen_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(gen_emb)))
        l2_dist = float(np.linalg.norm(gt_emb - gen_emb))

        return cosine_sim, l2_dist
    except Exception as exc:
        print(f"InsightFace error: {exc}")
        return None, None


def _get_ccip_similarity(gt_path: str, gen_path: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        from imgutils.metrics.ccip import ccip_difference, ccip_extract_feature

        gt_feat = ccip_extract_feature(gt_path)
        gen_feat = ccip_extract_feature(gen_path)

        diff = ccip_difference(gt_feat, gen_feat)
        similarity = 1 - (diff / 2)
        return float(similarity), float(diff)
    except Exception as exc:
        print(f"CCIP error: {exc}")
        return None, None


def evaluate_similarity_dataset(
    dataset_name: str,
    gt_dir: str,
    gen_dir: str,
    metric_type: str,
    l2_mode: str,
    app,
) -> List[Dict]:
    results = []
    gt_files = _list_images(gt_dir)
    gen_files = os.listdir(gen_dir)

    for gt_file in tqdm(gt_files, desc=f"{dataset_name}"):
        base_name = os.path.splitext(gt_file)[0]
        gt_path = os.path.join(gt_dir, gt_file)

        gen_path = _find_gen_match(base_name, gen_dir, gen_files)
        if gen_path is None or not os.path.exists(gen_path):
            continue

        if metric_type == "insightface":
            cosine_sim, l2_dist = _get_insightface_metrics(gt_path, gen_path, app, l2_mode)
            if cosine_sim is not None:
                results.append(
                    {
                        "gt_file": gt_file,
                        "gen_file": os.path.basename(gen_path),
                        "cosine_similarity": cosine_sim,
                        "l2_distance": l2_dist,
                    }
                )
        else:
            sim, diff = _get_ccip_similarity(gt_path, gen_path)
            if sim is not None:
                results.append(
                    {
                        "gt_file": gt_file,
                        "gen_file": os.path.basename(gen_path),
                        "similarity": sim,
                        "difference": diff,
                    }
                )

    return results


def _build_similarity_methods(
    preset: str,
    dataset: str,
    legacy_base: str,
    diffsynth_base: str,
    qwen_base: str,
) -> Dict[str, str]:
    if preset == "flux":
        if dataset == "hsng":
            return {
                "normal": f"{diffsynth_base}/normal/hsng_normal",
                "fad": f"{diffsynth_base}/fad/hsng_fad",
                "sfad_trigger": f"{diffsynth_base}/sfad/hsng_sfad_trigger",
                "sfad_anchoring": f"{diffsynth_base}/sfad/hsng_sfad_anchoring",
            }
        return {
            "normal": f"{legacy_base}/{dataset}_normal",
            "fad_trigger": f"{legacy_base}/{dataset}_fad_trig",
            "fad_anchoring": f"{legacy_base}/{dataset}_fad_anch",
            "sfad_trigger": f"{diffsynth_base}/sfad/{dataset}_sfad_trigger",
            "sfad_anchoring": f"{diffsynth_base}/sfad/{dataset}_sfad_anchoring",
        }
    if preset == "qwen":
        return {
            "normal": f"{qwen_base}/{dataset}_normal",
            "fad_trigger": f"{qwen_base}/{dataset}_fad_trigger",
            "fad_anchoring": f"{qwen_base}/{dataset}_fad_anchoring",
            "sfad_trigger": f"{qwen_base}/{dataset}_sfad_trigger",
            "sfad_anchoring": f"{qwen_base}/{dataset}_sfad_anchoring",
        }
    raise ValueError(f"Unknown preset: {preset}")


def run_similarity(args: argparse.Namespace) -> None:
    if args.dataset == "all":
        datasets = list(DATASET_SPECS.keys())
    else:
        datasets = [args.dataset]

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS.get(args.preset, str(REPO_ROOT / "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        metric_type = spec["metric"]
        gt_dir = spec["gt_dir"]

        app = None
        if metric_type == "insightface":
            app = _get_insightface_app()

        methods = _build_similarity_methods(
            args.preset,
            dataset,
            args.legacy_base,
            args.diffsynth_base,
            args.qwen_base,
        )

        if args.methods:
            keep = {m.strip() for m in args.methods.split(",") if m.strip()}
            methods = {k: v for k, v in methods.items() if k in keep}

        all_results = {}

        for method_name, gen_dir in methods.items():
            if not os.path.exists(gen_dir):
                print(f"Skip: {gen_dir} not found")
                continue

            print("=" * 60)
            print(f"Evaluating {dataset} - {method_name} ({metric_type})")
            print("=" * 60)

            results = evaluate_similarity_dataset(
                dataset_name=f"{dataset}/{method_name}",
                gt_dir=gt_dir,
                gen_dir=gen_dir,
                metric_type=metric_type,
                l2_mode=args.l2_mode,
                app=app,
            )

            if not results:
                continue

            if metric_type == "insightface":
                cosine_sims = [r["cosine_similarity"] for r in results]
                l2_dists = [r["l2_distance"] for r in results]
                all_results[method_name] = {
                    "cosine_sim_mean": float(np.mean(cosine_sims)),
                    "cosine_sim_std": float(np.std(cosine_sims)),
                    "l2_dist_mean": float(np.mean(l2_dists)),
                    "l2_dist_std": float(np.std(l2_dists)),
                    "count": len(results),
                    "details": results,
                }
                print(
                    f"{method_name}: cos_sim={np.mean(cosine_sims):.4f}, l2_dist={np.mean(l2_dists):.4f}, count={len(results)}"
                )
            else:
                similarities = [r["similarity"] for r in results]
                differences = [r["difference"] for r in results]
                all_results[method_name] = {
                    "similarity_mean": float(np.mean(similarities)),
                    "similarity_std": float(np.std(similarities)),
                    "difference_mean": float(np.mean(differences)),
                    "difference_std": float(np.std(differences)),
                    "count": len(results),
                    "details": results,
                }
                print(
                    f"{method_name}: sim={np.mean(similarities):.4f}, diff={np.mean(differences):.4f}, count={len(similarities)}"
                )

        output_file = os.path.join(output_dir, f"{dataset}_{metric_type}_results.json")
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(
                {"dataset": dataset, "metric": metric_type, "results": all_results},
                handle,
                indent=2,
            )

        print(f"Results saved to {output_file}")


# -----------------------------
# GPT evaluation helpers
# -----------------------------

def _build_prompt(character_desc: str, char_type: str) -> str:
    if char_type == "human":
        return (
            "You will be shown TWO images:\n"
            "- Image 1: the ground truth character image (reference only)\n"
            "- Image 2: a generated image that should depict the same character\n\n"
            "Your task is to critically evaluate how well Image 2 resembles the ground truth character in Image 1,\n"
            "and to assess the overall composition and image quality.\n\n"
            "Ground Truth Character key features (from Image 1)\n"
            f"{character_desc}\n\n"
            "Evaluation Dimensions\n"
            "1) Character Similarity - facial structure, hair, skin tone, and other distinctive features\n"
            "2) Composition and Image Quality - composition coherence, deformities, texture, lighting, color, clarity\n\n"
            "Scoring Criteria\n"
            "Character Similarity (10-point scale)\n"
            "10 = highly similar, 1 = completely different\n"
            "- Deduct 2 points for major facial mismatch\n"
            "- Deduct 1 point for missing or incorrect key features\n"
            "Composition and Image Quality (10-point scale)\n"
            "10 = excellent, 1 = very poor\n"
            "- Deduct 2 points for major deformities\n"
            "- Deduct 1 point for lighting/color/texture issues\n\n"
            "Output Format (strict)\n"
            "Image 2 Evaluation:\n"
            "- Concise bullet points listing observed issues or mismatches\n"
            "Scores:\n"
            "Character Similarity: [score]/10\n"
            "Composition and Image Quality: [score]/10\n\n"
            "Notes\n"
            "- Do not evaluate Image 1.\n"
            "- Be critical and specific.\n"
        )
    return (
        "You will be shown TWO images:\n"
        "- Image 1: the ground truth character image (reference only)\n"
        "- Image 2: a generated image that should depict the same character\n\n"
        "Your task is to critically evaluate how well Image 2 resembles the ground truth character in Image 1,\n"
        "and to assess the overall composition and image quality.\n\n"
        "Ground Truth Character key features (from Image 1)\n"
        f"{character_desc}\n\n"
        "Evaluation Dimensions\n"
        "1) Character Similarity - body shape, colors, and distinctive features\n"
        "2) Composition and Image Quality - style consistency, clarity, artifacts\n\n"
        "Scoring Criteria\n"
        "Character Similarity (10-point scale)\n"
        "10 = highly similar, 1 = completely different\n"
        "- Deduct 2 points for major shape/color mismatch\n"
        "- Deduct 1 point for missing or incorrect key features\n"
        "Composition and Image Quality (10-point scale)\n"
        "10 = excellent, 1 = very poor\n"
        "- Deduct 2 points for major deformities\n"
        "- Deduct 1 point for style inconsistency or minor artifacts\n\n"
        "Output Format (strict)\n"
        "Image 2 Evaluation:\n"
        "- Concise bullet points listing observed issues or mismatches\n"
        "Scores:\n"
        "Character Similarity: [score]/10\n"
        "Composition and Image Quality: [score]/10\n\n"
        "Notes\n"
        "- Do not evaluate Image 1.\n"
        "- Be critical and specific.\n"
    )


def _convert_image_to_png(image_path: str, temp_dir: str) -> Optional[str]:
    if not image_path.lower().endswith(VALID_EXTENSIONS):
        return None

    if image_path.lower().endswith(".png"):
        return image_path

    try:
        os.makedirs(temp_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        png_path = os.path.join(temp_dir, f"{base_name}.png")
        img = Image.open(image_path)
        img.save(png_path, "PNG")
        return png_path
    except Exception as exc:
        print(f"Failed to convert {image_path}: {exc}")
        return None


def _normalize_stems(filename: str) -> List[str]:
    base = os.path.splitext(filename)[0]
    if "." in base:
        alt = os.path.splitext(base)[0]
        return [base, alt]
    return [base]


def _find_gt_by_basename(gen_file: str, gt_folder: str) -> Optional[str]:
    gen_stems = _normalize_stems(gen_file)
    for gt_file in os.listdir(gt_folder):
        if not gt_file.lower().endswith(VALID_EXTENSIONS):
            continue
        gt_stems = _normalize_stems(gt_file)
        if any(gs == gt for gs in gen_stems for gt in gt_stems):
            return os.path.join(gt_folder, gt_file)
    return None


def _extract_index_from_name(name: str) -> Optional[int]:
    stem = os.path.splitext(name)[0]
    matches = re.findall(r"(\d+)", stem)
    if not matches:
        return None
    return int(matches[-1])


def _find_gt_by_index(gen_file: str, gt_folder: str) -> Optional[str]:
    index = _extract_index_from_name(gen_file)
    if index is None:
        return None
    for gt_file in os.listdir(gt_folder):
        if not gt_file.lower().endswith(VALID_EXTENSIONS):
            continue
        gt_index = _extract_index_from_name(gt_file)
        if gt_index == index:
            return os.path.join(gt_folder, gt_file)
    return None


def _find_gt_for_gen(gen_file: str, gt_folder: str, match_mode: str) -> Optional[str]:
    if match_mode == "basename":
        return _find_gt_by_basename(gen_file, gt_folder)
    if match_mode == "index":
        return _find_gt_by_index(gen_file, gt_folder)
    if match_mode == "auto":
        return _find_gt_by_basename(gen_file, gt_folder) or _find_gt_by_index(gen_file, gt_folder)
    raise ValueError(f"Unknown match mode: {match_mode}")


def _evaluate_image_pair(
    api_url: str,
    api_key: str,
    model: str,
    gt_path: str,
    gen_path: str,
    prompt: str,
    temp_dir: str,
) -> Optional[str]:
    try:
        converted_gt = _convert_image_to_png(gt_path, temp_dir)
        if not converted_gt:
            return None
        with open(converted_gt, "rb") as handle:
            gt_data = base64.b64encode(handle.read()).decode("utf-8")
        mime_gt = mimetypes.guess_type(converted_gt)[0] or "image/png"

        converted_gen = _convert_image_to_png(gen_path, temp_dir)
        if not converted_gen:
            return None
        with open(converted_gen, "rb") as handle:
            gen_data = base64.b64encode(handle.read()).decode("utf-8")
        mime_gen = mimetypes.guess_type(converted_gen)[0] or "image/png"

        message_content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "Image 1 (Ground Truth):"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_gt};base64,{gt_data}"}},
            {"type": "text", "text": "Image 2 (Generated):"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_gen};base64,{gen_data}"}},
        ]

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": 1000,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"API error: {exc}")
        return None


def _extract_scores(text: str) -> Tuple[Optional[int], Optional[int]]:
    char_match = re.search(r"Character Similarity:\s*(\d+)/10", text)
    comp_match = re.search(r"Composition (and|&) Image Quality:\s*(\d+)/10", text)
    if comp_match is None:
        comp_match = re.search(r"Composition and Image Quality:\s*(\d+)/10", text)
    char_score = int(char_match.group(1)) if char_match else None
    comp_score = int(comp_match.group(2)) if comp_match and comp_match.lastindex == 2 else None
    if comp_score is None and comp_match and comp_match.lastindex == 1:
        comp_score = int(comp_match.group(1))
    return char_score, comp_score


def _normalize_variant_for_source(source: str, variant: str) -> str:
    variant = variant.strip()
    if source == "flux":
        mapping = {
            "fad_trigger": "fad_trig",
            "fad_anchoring": "fad_anch",
            "sfad_trigger": "sfad_trig",
            "sfad_anchoring": "sfad_anch",
        }
        return mapping.get(variant, variant)
    if source == "qwen":
        mapping = {
            "fad_trig": "fad_trigger",
            "fad_anch": "fad_anchoring",
            "sfad_trig": "sfad_trigger",
            "sfad_anch": "sfad_anchoring",
        }
        return mapping.get(variant, variant)
    if source == "sfad":
        mapping = {
            "trig": "trigger",
            "anch": "anchoring",
            "sfad_trig": "trigger",
            "sfad_anch": "anchoring",
            "sfad_trigger": "trigger",
            "sfad_anchoring": "anchoring",
        }
        return mapping.get(variant, variant)
    return variant


def _resolve_gen_folder(
    source: str,
    dataset: str,
    variant: str,
    gen_base: Optional[str],
) -> str:
    if gen_base:
        base = gen_base
    else:
        if source == "flux":
            base = "/scratch2/solbon1212/inference_results"
        elif source == "qwen":
            base = str(REPO_ROOT / "output_inference" / "qwen")
        elif source == "sfad":
            base = str(REPO_ROOT / "output_inference" / "sfad")
        elif source == "sfad-legacy":
            base = "/scratch2/solbon1212/outputs/sfad_inference"
        elif source == "hsng":
            base = str(REPO_ROOT / "output_inference")
        else:
            raise ValueError(f"Unknown source: {source}")

    variant_norm = _normalize_variant_for_source(source, variant)

    if source == "hsng":
        if variant_norm == "normal":
            return f"{base}/normal/hsng_normal"
        if variant_norm == "fad":
            return f"{base}/fad/hsng_fad"
        if variant_norm in {"sfad_trigger", "sfad_anchoring"}:
            return f"{base}/sfad/hsng_{variant_norm}"
        return f"{base}/{variant_norm}"

    if source in {"sfad", "sfad-legacy"}:
        if variant_norm in {"trigger", "anchoring"}:
            return f"{base}/{dataset}_sfad_{variant_norm}"
        return f"{base}/{dataset}_{variant_norm}"

    return f"{base}/{dataset}_{variant_norm}"


def run_gpt_eval(args: argparse.Namespace) -> None:
    if args.api_key:
        api_key = args.api_key
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Pass --api-key or set env var.")

    datasets = list(GPT_DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    summaries = []
    default_output_dir = (
        "/scratch2/solbon1212/outputs/gpt_eval_results/qwen"
        if args.source == "qwen"
        else "/scratch2/solbon1212/outputs/gpt_eval_results"
    )
    summary_dir = args.output_dir or default_output_dir

    for dataset in datasets:
        config = GPT_DATASETS[dataset]
        gt_folder = config["gt_folder"]
        prompt = _build_prompt(config["character_desc"], config["type"])

        gen_folder = _resolve_gen_folder(args.source, dataset, args.variant, args.gen_base)
        if not os.path.exists(gen_folder):
            print(f"Skip: generated folder not found: {gen_folder}")
            continue

        os.makedirs(summary_dir, exist_ok=True)
        output_path = os.path.join(summary_dir, f"{dataset}_{args.variant}_gpt_eval.json")

        evaluations = []
        processed_pairs = set()
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as handle:
                    evaluations = json.load(handle)
                for item in evaluations:
                    processed_pairs.add((item["gt_image"], item["gen_image"]))
                print(f"Loaded {len(evaluations)} existing evaluations")
            except Exception:
                evaluations = []

        gen_files = [
            f for f in os.listdir(gen_folder) if f.lower().endswith(VALID_EXTENSIONS)
        ]
        gen_files = sorted(gen_files)[: args.max_images]

        print("=" * 60)
        print(f"GPT eval: {dataset} - {args.variant}")
        print(f"GT folder: {gt_folder}")
        print(f"Gen folder: {gen_folder}")
        print(f"Total images: {len(gen_files)}")
        print("=" * 60)

        new_count = 0
        for gen_file in gen_files:
            gen_path = os.path.join(gen_folder, gen_file)
            gt_path = _find_gt_for_gen(gen_file, gt_folder, args.match)
            if not gt_path:
                print(f"No GT match for: {gen_file}")
                continue

            gt_file = os.path.basename(gt_path)
            if (gt_file, gen_file) in processed_pairs:
                print(f"Skipping already processed: {gen_file}")
                continue

            result = _evaluate_image_pair(
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                gt_path=gt_path,
                gen_path=gen_path,
                prompt=prompt,
                temp_dir=args.temp_dir,
            )

            if result:
                char_score, comp_score = _extract_scores(result)
                evaluation = {
                    "gt_image": gt_file,
                    "gen_image": gen_file,
                    "evaluation": result,
                    "char_score": char_score,
                    "comp_score": comp_score,
                }
                evaluations.append(evaluation)
                processed_pairs.add((gt_file, gen_file))
                new_count += 1

                with open(output_path, "w", encoding="utf-8") as handle:
                    json.dump(evaluations, handle, ensure_ascii=False, indent=2)

                time.sleep(args.sleep)
            else:
                print(f"Failed to evaluate: {gen_file}")

        char_scores = [e["char_score"] for e in evaluations if e.get("char_score")]
        comp_scores = [e["comp_score"] for e in evaluations if e.get("comp_score")]

        summary = {
            "dataset": dataset,
            "variant": args.variant,
            "total_evaluations": len(evaluations),
            "new_evaluations": new_count,
            "avg_char_similarity": sum(char_scores) / len(char_scores) if char_scores else 0,
            "avg_comp_quality": sum(comp_scores) / len(comp_scores) if comp_scores else 0,
        }

        summaries.append(summary)
        print(f"Summary for {dataset}_{args.variant}: {summary}")

    if summaries:
        summary_path = os.path.join(summary_dir, "all_summaries.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)
        print(f"Summary saved to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_parser = subparsers.add_parser("similarity", help="InsightFace/CCIP evaluation")
    sim_parser.add_argument("--dataset", required=True, choices=list(DATASET_SPECS.keys()) + ["all"])
    sim_parser.add_argument("--preset", choices=["flux", "qwen"], default="flux")
    sim_parser.add_argument("--output-dir", default=None)
    sim_parser.add_argument("--methods", default=None, help="Comma-separated method names to include")
    sim_parser.add_argument("--l2-mode", choices=["normalized", "raw"], default="normalized")
    sim_parser.add_argument("--legacy-base", default="/scratch2/solbon1212/inference_results")
    sim_parser.add_argument("--diffsynth-base", default=str(REPO_ROOT / "output_inference"))
    sim_parser.add_argument("--qwen-base", default=str(REPO_ROOT / "output_inference" / "qwen"))

    gpt_parser = subparsers.add_parser("gpt", help="GPT-based evaluation")
    gpt_parser.add_argument("--dataset", required=True, choices=list(GPT_DATASETS.keys()) + ["all"])
    gpt_parser.add_argument("--variant", required=True, help="Variant or method name")
    gpt_parser.add_argument(
        "--source",
        choices=["flux", "qwen", "sfad", "sfad-legacy", "hsng"],
        default="flux",
    )
    gpt_parser.add_argument("--gen-base", default=None, help="Override generated images base folder")
    gpt_parser.add_argument("--output-dir", default=None)
    gpt_parser.add_argument("--max-images", type=int, default=50)
    gpt_parser.add_argument("--model", default="gpt-4.1")
    gpt_parser.add_argument("--api-url", default="https://api.openai.com/v1/chat/completions")
    gpt_parser.add_argument("--api-key", default=None)
    gpt_parser.add_argument("--match", choices=["basename", "index", "auto"], default="auto")
    gpt_parser.add_argument("--sleep", type=float, default=1.0)
    gpt_parser.add_argument("--temp-dir", default="/tmp/diffsynth_gpt_eval")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "similarity":
        run_similarity(args)
    elif args.command == "gpt":
        run_gpt_eval(args)


if __name__ == "__main__":
    main()
