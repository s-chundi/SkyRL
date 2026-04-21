import argparse
import glob
import json
import os
import subprocess
import fnmatch

import datasets
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger
from PIL import Image

COMMON_COLUMNS = ["question", "answer", "image", "width", "height", "bboxs", "dataset", "split"]

BBOX_TASK = "You are a visual reasoning model. Reason and locate the answer within the image and provide bounding box coordinates in the format `[x1, y1, x2, y2]`, where the coordinates are in a PIL format (x increases going right, y increases going down). x and y should be normalized to be within [0, 1000], i.e. (0, 0) is the top left and (1000, 1000) is the bottom right. After outputting the bounding box, you may also be asked to answer the question."

DATA_SOURCE = "deepcs233/Visual-CoT"


def download_and_extract_images(images_dir):
    """Download the split tar parts from HF and extract into images_dir."""
    os.makedirs(images_dir, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(DATA_SOURCE, repo_type="dataset")
    tar_parts = sorted(f for f in repo_files if f.startswith("cot_images_tar_split/"))

    logger.info(f"Downloading {len(tar_parts)} tar parts to {images_dir}")
    local_parts = []
    for part in tar_parts:
        local_path = hf_hub_download(DATA_SOURCE, part, repo_type="dataset")
        local_parts.append(local_path)
        logger.info(f"  Downloaded {part}")

    logger.info(f"Extracting images into {images_dir}")
    cat_cmd = ["cat"] + local_parts
    cat_proc = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE)
    tar_proc = subprocess.Popen(["tar", "xf", "-", "-C", images_dir], stdin=cat_proc.stdout)
    cat_proc.stdout.close()
    tar_proc.communicate()
    if tar_proc.returncode != 0:
        raise RuntimeError(f"tar extraction failed with return code {tar_proc.returncode}")
    logger.info("Extraction complete")


def load_visual_cot(images_dir, test_fraction=0.15, max_length=None):
    """Load JSONL files, split each into train/test, and resolve image paths."""
    api = HfApi()
    files = api.list_repo_files(DATA_SOURCE, repo_type="dataset")
    jsonl_files = [f for f in files if fnmatch.fnmatch(f, "metadata/*.jsonl")]

    train_datasets = []
    test_datasets = []
    total_rows = 0
    for jsonl_file in jsonl_files:
        logger.info(f"Loading {jsonl_file}")
        ds = datasets.load_dataset(
            DATA_SOURCE, data_files=jsonl_file, split="train"
        )
        cols_to_remove = [c for c in ds.column_names if c not in COMMON_COLUMNS]
        ds = ds.remove_columns(cols_to_remove)

        n_test = int(len(ds) * test_fraction)
        n_train = len(ds) - n_test
        train_ds = ds.select(range(n_train))
        test_ds = ds.select(range(n_train, len(ds)))

        train_datasets.append(train_ds)
        test_datasets.append(test_ds)
        total_rows += len(ds)
        if max_length is not None and total_rows >= max_length:
            logger.info(f"Reached {total_rows} rows (>= max_length={max_length}), stopping early")
            break

    combined_train = datasets.concatenate_datasets(train_datasets)
    combined_test = datasets.concatenate_datasets(test_datasets)
    if max_length is not None:
        combined_train = combined_train.select(range(min(max_length, len(combined_train))))
        combined_test = combined_test.select(range(min(max_length, len(combined_test))))
    return combined_train, combined_test


def make_map_fn(split, images_dir):
    def process_fn(example, idx):
        image_path = os.path.join(images_dir, "cot_image_data", example["dataset"], example["image"])
        image = Image.open(image_path).convert("RGB")

        prompt = [
            {"role": "system", "content": BBOX_TASK},
            {"role": "user", "content": "<image>\n" + example["question"]},
        ]
        reward_spec = {
            "method": "rule",
            "bbox": example["bboxs"],
            "answer": example["answer"],
        }
        extra_info = {
            "split": split,
            "width": example["width"],
            "height": example["height"],
            "index": idx,
        }
        data = {
            "data_source": DATA_SOURCE,
            "prompt": prompt,
            "images": [image],
            "env_class": "visual_cot",
            "reward_spec": json.dumps(reward_spec),
            "extra_info": json.dumps(extra_info),
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", 
        default="~/data/visual_cot"
    )
    parser.add_argument(
        "--images_dir", 
        default="~/data/visual_cot/images",
        help="Directory to extract/find images. Will download if empty."
    )
    parser.add_argument(
        "--max_dataset_length",
        type=int,
        default=None,
        help="If set, truncate the training split to this many examples.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.15,
        help="Fraction of each JSONL file to use as the test set (default: 0.15).",
    )
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    args.images_dir = os.path.expanduser(args.images_dir)

    if not os.path.isdir(args.images_dir) or not os.listdir(args.images_dir):
        logger.info("Images not found locally, downloading and extracting...")
        download_and_extract_images(args.images_dir)
    else:
        logger.info(f"Using existing images in {args.images_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset, test_dataset = load_visual_cot(
        args.images_dir,
        test_fraction=args.test_fraction,
        max_length=args.max_dataset_length,
    )

    for split, dataset in [("train", train_dataset), ("test", test_dataset)]:
        def _image_exists(example, _dir=args.images_dir):
            path = os.path.join(_dir, "cot_image_data", example["dataset"], example["image"])
            if not os.path.isfile(path):
                logger.warning(f"Image not found: {path}, skipping")
                return False
            return True

        dataset = dataset.filter(_image_exists)
        dataset = dataset.map(
            function=make_map_fn(split, args.images_dir),
            with_indices=True,
            remove_columns=dataset.column_names,
        )
        dataset = dataset.cast_column("images", datasets.Sequence(datasets.Image()))

        output_file = os.path.join(args.output_dir, f"{split}.parquet")
        dataset.to_parquet(output_file)
        logger.info(f"Saved {len(dataset)} examples to {output_file}")