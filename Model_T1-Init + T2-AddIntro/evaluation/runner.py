import json
import logging
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig

from .collator import FastDataCollator
from .config import (
    BATCH_SIZE, DATASET_PATH, GENERATION_CONFIGS, MODEL_PATH, NUM_WORKERS,
    OUTPUT_DIR, PAD_TO_MULTIPLE_OF, TASK_NAMES
)
from .dataset import EvaluationDataset
from .metrics import compute_task_metrics, load_metrics

logger = logging.getLogger(__name__)

def _build_generation_configs(tokenizer):
    configs = {}
    for task_id, options in GENERATION_CONFIGS.items():
        configs[task_id] = GenerationConfig(
            **options,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
    return configs

def run_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(OUTPUT_DIR, "evaluation.log"), encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
    )

    logger.info("Starting evaluation: T1-Init + T2-AddIntro Multi-task")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.to(device).eval()

    if device.type == "cuda":
        model.half()
        logger.info("Using FP16 model weights on CUDA")

    dataset = EvaluationDataset(DATASET_PATH, tokenizer)
    collator = FastDataCollator(tokenizer, PAD_TO_MULTIPLE_OF)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    logger.info("Evaluation items: %d; batches: %d", len(dataset), len(loader))

    generation_configs = _build_generation_configs(tokenizer)
    predictions = {task_id: [] for task_id in range(4)}
    references = {task_id: [] for task_id in range(4)}

    start = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="Generating")):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            task_ids = batch["task_id"]

            for task_id in range(4):
                mask = task_ids == task_id
                if not torch.any(mask):
                    continue

                generated = model.generate(
                    input_ids=input_ids[mask.to(device)],
                    attention_mask=attention_mask[mask.to(device)],
                    generation_config=generation_configs[task_id],
                )
                predictions[task_id].extend(
                    tokenizer.batch_decode(generated, skip_special_tokens=True)
                )

                labels = batch["labels"][mask].clone()
                labels[labels == -100] = tokenizer.pad_token_id
                references[task_id].extend(
                    tokenizer.batch_decode(labels, skip_special_tokens=True)
                )

            if (batch_index + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = (batch_index + 1) / max(elapsed, 1e-9)
                eta_minutes = (len(loader) - batch_index - 1) / rate / 60
                logger.info(
                    "Batch %d/%d; ETA %.1f min",
                    batch_index + 1, len(loader), eta_minutes
                )

    elapsed = time.time() - start
    logger.info("Text generation completed in %.2f minutes", elapsed / 60)

    for task_id in range(4):
        output_path = os.path.join(
            OUTPUT_DIR, f"task_{task_id}_predictions.csv"
        )
        pd.DataFrame({
            "prediction": predictions[task_id],
            "reference": references[task_id],
        }).to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Saved task %d predictions: %s", task_id, output_path)

    metric_objects = load_metrics()
    final_results = {}
    for task_id in range(4):
        result = compute_task_metrics(
            predictions[task_id], references[task_id], metric_objects
        )
        if result is None:
            logger.warning("Task %d has no predictions; skipping metrics", task_id)
            continue
        final_results[TASK_NAMES[task_id]] = result
        logger.info("Computed metrics for %s", TASK_NAMES[task_id])

    result_path = os.path.join(OUTPUT_DIR, "final_evaluation_results.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(final_results, handle, ensure_ascii=False, indent=4)

    logger.info("Evaluation finished. Results: %s", result_path)
    logger.info("Total generation time: %.2f minutes", elapsed / 60)
