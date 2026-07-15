# -*- coding: utf-8 -*-
"""Logits DeConfusion with CLIP for few-shot learning.

This module contains the paper-style LDC implementation used by LibFewShot.
The CLIP architecture is loaded from the author's reference repository rather
than copied into LibFewShot. Set ``ldc_root`` or the ``LDC_ROOT`` environment
variable to point to that repository.
"""
import csv
import importlib
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from core.utils import accuracy
from .metric_model import MetricModel

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


EUROSAT_CLASSNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}

DATASET_TEMPLATES = {
    "caltech101": ["a photo of a {}."],
    "dtd": ["{} texture."],
    "eurosat": ["a centered satellite photo of {}."],
    "fgvc": ["a photo of a {}, a type of aircraft."],
    "food101": ["a photo of {}, a type of food."],
    "oxford_flowers": ["a photo of a {}, a type of flower."],
    "oxford_pets": ["a photo of a {}, a type of pet."],
    "stanford_cars": ["a photo of a {}."],
    "stanfordcar": ["a photo of a {}."],
    "sun397": ["a photo of a {}."],
    "ucf101": ["a photo of a person doing {}."],
}

DATASET_SPLIT_JSON = {
    "caltech101": "split_zhou_Caltech101.json",
    "dtd": "split_zhou_DescribableTextures.json",
    "eurosat": "split_zhou_EuroSAT.json",
    "food101": "split_zhou_Food101.json",
    "oxford_flowers": "split_zhou_OxfordFlowers.json",
    "oxford_pets": "split_zhou_OxfordPets.json",
    "stanford_cars": "split_zhou_StanfordCars.json",
    "stanfordcar": "split_zhou_StanfordCars.json",
    "sun397": "split_zhou_SUN397.json",
    "ucf101": "split_zhou_UCF101.json",
}

DATASET_PATH_PREFIX = {
    "caltech101": "101_ObjectCategories",
    "dtd": "images",
    "eurosat": "2750",
    "food101": "images",
    "oxford_flowers": "jpg",
    "oxford_pets": "images",
    "stanford_cars": "",
    "stanfordcar": "",
    "sun397": "SUN397",
    "ucf101": "UCF-101-midframes",
}


@dataclass
class Datum:
    impath: str
    label: int
    classname: str


class LDCImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(item.impath).convert("RGB")
        return self.transform(image), item.label


class LDCConfig(object):
    def __init__(self, values):
        for key, value in values.items():
            setattr(self, key, value)

    def as_dict(self):
        return dict(self.__dict__)


class LDC(MetricModel):
    """LibFewShot classifier entry for LDC.

    The paper reproduction path is launched through :func:`run_ldc_from_config`.
    The inherited episodic forward methods are intentionally unavailable because
    the CVPR 2025 method trains CLIP adapters on fixed dataset splits rather
    than LibFewShot N-way episodic batches.
    """

    def __init__(self, paper_mode=True, **kwargs):
        super(LDC, self).__init__(**kwargs)
        self.paper_mode = paper_mode

    def set_forward(self, batch):
        raise RuntimeError(
            "LDC paper reproduction uses CLIP fixed-split training. "
            "Run it with `python run_ldc.py --config ./config/ldc.yaml`."
        )

    def set_forward_loss(self, batch):
        raise RuntimeError(
            "LDC paper reproduction uses CLIP fixed-split training. "
            "Run it with `python run_ldc.py --config ./config/ldc.yaml`."
        )


def is_ldc_paper_config(config):
    classifier = config.get("classifier", {})
    kwargs = classifier.get("kwargs") or {}
    return classifier.get("name") == "LDC" and kwargs.get("paper_mode", False)


def load_clip_backend(ldc_root):
    """Import ``clip_ldc`` from an explicit author-repository path."""
    ldc_root = os.path.abspath(os.path.expanduser(os.path.expandvars(ldc_root)))
    package_dir = os.path.join(ldc_root, "clip_ldc")
    if not os.path.isdir(package_dir):
        raise ImportError(
            "Cannot find clip_ldc under {}. Set ldc_root in the config or set "
            "the LDC_ROOT environment variable.".format(ldc_root)
        )
    if ldc_root not in sys.path:
        sys.path.insert(0, ldc_root)
    return importlib.import_module("clip_ldc")


def run_ldc_from_config(
    config,
    eval_only=False,
    checkpoint=None,
    check_data=False,
    smoke_forward=False,
):
    ldc_config = build_ldc_config(config)
    setup_seed(ldc_config.seed)
    os.makedirs(ldc_config.output_dir, exist_ok=True)

    with open(os.path.join(ldc_config.output_dir, "config.yaml"), "w", encoding="utf-8") as fout:
        yaml.safe_dump(config, fout, allow_unicode=True)

    train_items, val_items, test_items, classnames = build_datasets(ldc_config)
    ldc_config.num_classes = len(classnames)
    missing = count_missing_paths(train_items + val_items + test_items)
    print(
        "dataset={} classes={} shots={} train/val/test={}/{}/{}".format(
            ldc_config.dataset_name,
            ldc_config.num_classes,
            ldc_config.shots,
            len(train_items),
            len(val_items),
            len(test_items),
        )
    )
    print("missing_paths={}".format(missing))
    print("classnames={}".format(classnames[:10]))
    if check_data:
        if missing:
            raise FileNotFoundError(
                "{} paths from the selected split are missing under {}".format(
                    missing, ldc_config.data_root
                )
            )
        return None
    if missing:
        raise FileNotFoundError(
            "{} paths from the selected split are missing under {}".format(
                missing, ldc_config.data_root
            )
        )
    clip_backend = load_clip_backend(ldc_config.ldc_root)

    device = torch.device(
        ldc_config.device if torch.cuda.is_available() and ldc_config.device != "cpu" else "cpu"
    )
    model, _ = clip_backend.load(
        ldc_config.clip_model,
        download_root=ldc_config.clip_cache_dir,
        num_classes=ldc_config.num_classes,
        config=ldc_config,
        device=device,
    )
    model.eval()

    template = getattr(
        ldc_config,
        "template",
        DATASET_TEMPLATES.get(ldc_config.dataset_name.lower(), ["a photo of a {}."]),
    )
    text_cache = os.path.join(
        ldc_config.output_dir,
        "{}_{}_textfeats.pt".format(
            ldc_config.dataset_name, ldc_config.clip_model.replace("/", "-")
        ),
    )
    text_feats = clip_classifier(text_cache, classnames, template, model, device, clip_backend).to(
        device
    )

    train_loader = build_loader(
        train_items,
        transform_train(ldc_config.image_size),
        ldc_config.batch_size,
        ldc_config.num_workers,
        shuffle=True,
    )
    val_loader = build_loader(
        val_items,
        transform_test(ldc_config.image_size),
        ldc_config.eval_batch_size,
        ldc_config.num_workers,
        shuffle=False,
    )
    test_loader = build_loader(
        test_items,
        transform_test(ldc_config.image_size),
        ldc_config.eval_batch_size,
        ldc_config.num_workers,
        shuffle=False,
    )

    if smoke_forward:
        smoke_ldc_forward(model, train_loader, text_feats, device)
        return None
    if eval_only:
        checkpoint = checkpoint or os.path.join(ldc_config.output_dir, "model_best.pth")
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        beta = payload.get("beta")
        _, test_result = evaluate(model, test_loader, text_feats, device, beta=beta)
        print("Test result: {}".format(test_result))
        return test_result

    return train_ldc(
        ldc_config,
        model,
        train_loader,
        val_loader,
        test_loader,
        text_feats,
        device,
    )


def build_ldc_config(config):
    classifier_kwargs = config.get("classifier", {}).get("kwargs") or {}
    ldc_values = dict(classifier_kwargs.get("ldc", {}))

    aliases = {
        "ldc_root": "ldc_root",
        "clip_model": "clip_model",
        "clip_cache_dir": "clip_cache_dir",
        "dataset_name": "dataset_name",
        "data_root": "data_root",
        "split_json": "split_json",
        "path_prefix": "path_prefix",
        "output_dir": "output_dir",
        "shots": "shots",
        "seed": "seed",
        "device": "device",
        "image_size": "image_size",
        "batch_size": "batch_size",
        "eval_batch_size": "eval_batch_size",
        "num_workers": "num_workers",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "eps": "eps",
        "epochs": "epochs",
        "loss_lambda": "loss_lambda",
        "fuse_type": "fuse_type",
        "template": "template",
    }
    for source, target in aliases.items():
        if source in config:
            ldc_values[target] = config[source]

    ldc_values.setdefault("dataset_name", "eurosat")
    ldc_values.setdefault("data_root", "./data/EuroSAT")
    ldc_values.setdefault("clip_model", "RN50")
    environment_root = os.environ.get("LDC_ROOT")
    if environment_root:
        ldc_values["ldc_root"] = environment_root
    else:
        ldc_values.setdefault("ldc_root", "../LDC")
    ldc_values.setdefault("seed", config.get("seed", 2024))
    ldc_values.setdefault("shots", config.get("shot_num", 1))
    ldc_values.setdefault("epochs", config.get("epoch", 50))
    ldc_values.setdefault("batch_size", config.get("batch_size", 64))
    ldc_values.setdefault("eval_batch_size", 64)
    ldc_values.setdefault("num_workers", config.get("workers", 0))
    ldc_values.setdefault("lr", 0.001)
    ldc_values.setdefault("weight_decay", 0.0001)
    ldc_values.setdefault("eps", 0.0001)
    ldc_values.setdefault("device", "cuda")
    ldc_values.setdefault("image_size", 224)
    ldc_values.setdefault("fuse_type", 2)
    ldc_values.setdefault("loss_lambda", [1.0, 1.0, 1.0, 1.0, 1.0])

    dataset_key = ldc_values["dataset_name"].lower()
    ldc_values["ldc_root"] = os.path.normpath(
        os.path.expanduser(os.path.expandvars(ldc_values["ldc_root"]))
    )
    ldc_values.setdefault("clip_cache_dir", os.path.join(ldc_values["ldc_root"], "model", "clip"))
    split_filename = DATASET_SPLIT_JSON.get(dataset_key)
    if split_filename:
        ldc_values.setdefault(
            "split_json",
            os.path.join(ldc_values["ldc_root"], "datasets", "jsons", split_filename),
        )
    ldc_values.setdefault(
        "output_dir",
        "./results/LDC-CLIP-{}-{}shot".format(ldc_values["dataset_name"], ldc_values["shots"]),
    )
    ldc_values.setdefault("path_prefix", DATASET_PATH_PREFIX.get(dataset_key, "images"))
    if isinstance(ldc_values.get("template"), str):
        ldc_values["template"] = [ldc_values["template"]]
    for key in ["data_root", "clip_cache_dir", "output_dir"]:
        ldc_values[key] = os.path.normpath(ldc_values[key])
    if ldc_values.get("split_json"):
        ldc_values["split_json"] = os.path.normpath(ldc_values["split_json"])
    return LDCConfig(ldc_values)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def convert_image_to_rgb(image):
    return image.convert("RGB")


def transform_train(size, scale=(0.8, 1.0)):
    return Compose(
        [
            T.RandomResizedCrop(size=size, scale=scale, interpolation=BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def transform_test(size):
    return Compose(
        [
            Resize(size, interpolation=BICUBIC),
            CenterCrop(size),
            convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def clean_csv_text(value):
    return str(value).strip().lstrip("\ufeff").strip('"').strip("'")


def normalize_classname(label):
    label = str(label)
    return EUROSAT_CLASSNAMES.get(label, label.replace("_", " "))


def read_csv_rows(data_root, split):
    csv_path = os.path.join(data_root, "{}.csv".format(split))
    rows = []
    with open(csv_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            clean_row = {clean_csv_text(key): clean_csv_text(value) for key, value in row.items()}
            rows.append((clean_row["filename"], clean_row["label"]))
    return rows


def build_label_space(data_root):
    labels = set()
    for split in ["train", "val", "test"]:
        labels.update(label for _, label in read_csv_rows(data_root, split))
    labels = sorted(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_classname = {idx: normalize_classname(label) for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_classname


def read_csv_split(data_root, split, label_to_idx, idx_to_classname):
    image_dir = os.path.join(data_root, "images")
    rows = read_csv_rows(data_root, split)
    return [
        Datum(
            impath=os.path.join(image_dir, filename),
            label=label_to_idx[label],
            classname=idx_to_classname[label_to_idx[label]],
        )
        for filename, label in rows
    ]


def build_datasets(config):
    if getattr(config, "split_json", None):
        if not os.path.isfile(config.split_json):
            raise FileNotFoundError("Dataset split file not found: {}".format(config.split_json))
        train_all, val, test, classnames = build_datasets_from_split_json(
            config.data_root,
            config.split_json,
            path_prefix=getattr(config, "path_prefix", "images"),
        )
        train = sample_fewshot(train_all, config.shots, config.seed)
        return train, val, test, classnames

    label_to_idx, idx_to_classname = build_label_space(config.data_root)
    train_all = read_csv_split(config.data_root, "train", label_to_idx, idx_to_classname)
    val = read_csv_split(config.data_root, "val", label_to_idx, idx_to_classname)
    test = read_csv_split(config.data_root, "test", label_to_idx, idx_to_classname)
    train = sample_fewshot(train_all, config.shots, config.seed)
    classnames = [idx_to_classname[idx] for idx in range(len(idx_to_classname))]
    return train, val, test, classnames


def build_datasets_from_split_json(data_root, split_json, path_prefix="images"):
    with open(split_json, "r", encoding="utf-8") as fin:
        split = json.load(fin)

    image_dir = os.path.join(data_root, str(path_prefix))

    def convert(items):
        output = []
        for rel_path, label, classname in items:
            output.append(
                Datum(
                    impath=resolve_image_path(image_dir, rel_path, data_root),
                    label=int(label),
                    classname=classname,
                )
            )
        return output

    train = convert(split["train"])
    val = convert(split["val"])
    test = convert(split["test"])
    return train, val, test, classnames_from_items(train + val + test)


def resolve_image_path(image_dir, rel_path, data_root=None):
    candidates = [
        os.path.join(image_dir, rel_path),
        os.path.join(image_dir, rel_path.replace("/", "__").replace("\\", "__")),
        os.path.join(image_dir, os.path.basename(rel_path)),
    ]
    if data_root is not None:
        candidates.extend(
            [
                os.path.join(data_root, rel_path),
                os.path.join(data_root, rel_path.replace("/", "__").replace("\\", "__")),
                os.path.join(data_root, "images", rel_path),
                os.path.join(
                    data_root,
                    "images",
                    rel_path.replace("/", "__").replace("\\", "__"),
                ),
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def classnames_from_items(items):
    mapping = {}
    for item in items:
        mapping[item.label] = item.classname
    return [mapping[label] for label in sorted(mapping)]


def sample_fewshot(items, shots, seed):
    if shots < 1:
        return items
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for item in items:
        by_label[item.label].append(item)
    sampled = []
    for label in sorted(by_label):
        candidates = by_label[label]
        if len(candidates) >= shots:
            sampled.extend(rng.sample(candidates, shots))
        else:
            sampled.extend(rng.choices(candidates, k=shots))
    return sampled


def build_loader(items, transform, batch_size, workers, shuffle):
    return DataLoader(
        LDCImageDataset(items, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def clip_classifier(cache_path, classnames, template, clip_model, device, clip_backend):
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with torch.no_grad():
        weights = []
        for classname in classnames:
            texts = [prompt.format(classname.replace("_", " ")) for prompt in template]
            texts = clip_backend.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            weights.append(class_embedding)
        weights = torch.stack(weights, dim=1)
        torch.save(weights.cpu(), cache_path)
    return weights


def compute_loss(labels, clip_logits, mlp_logits, ada_logits, total_logits, lambdas):
    ce_mlp = F.cross_entropy(mlp_logits, labels) * lambdas[0]
    ce_ada = F.cross_entropy(ada_logits, labels) * lambdas[1]
    ce_total = F.cross_entropy(total_logits, labels) * lambdas[2]
    l1_mlp = F.l1_loss(mlp_logits, clip_logits) * lambdas[3]
    l1_ada = F.l1_loss(ada_logits, clip_logits) * lambdas[4]
    return l1_mlp + l1_ada + ce_mlp + ce_ada + ce_total


def fuse_logits(mlp_logits, ada_logits, beta):
    return beta * mlp_logits + (1.0 - beta) * ada_logits


def search_beta(mlp_logits, ada_logits, labels, steps=50):
    best_beta, best_acc = 0.0, -1.0
    for idx in range(steps + 1):
        beta = idx / steps
        acc = accuracy(fuse_logits(mlp_logits, ada_logits, beta), labels)
        if acc > best_acc:
            best_beta, best_acc = beta, acc
    return best_beta, best_acc


@torch.no_grad()
def evaluate(model, loader, text_feats, device, beta=None):
    model.eval()
    all_labels, all_mlp, all_ada, all_total, all_clip = [], [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        clip_logits, mlp_logits, ada_logits, total_logits = model.my_forward(images, text_feats)
        all_labels.append(labels)
        all_clip.append(clip_logits)
        all_mlp.append(mlp_logits)
        all_ada.append(ada_logits)
        all_total.append(total_logits)

    labels = torch.cat(all_labels)
    clip_logits = torch.cat(all_clip)
    mlp_logits = torch.cat(all_mlp)
    ada_logits = torch.cat(all_ada)
    total_logits = torch.cat(all_total)

    result = {
        "clip_logits": accuracy(clip_logits, labels),
        "mlp_logits": accuracy(mlp_logits, labels),
        "ada_logits": accuracy(ada_logits, labels),
        "tot_logits": accuracy(total_logits, labels),
    }
    if beta is None:
        beta, result["acc"] = search_beta(mlp_logits, ada_logits, labels)
    else:
        result["acc"] = accuracy(fuse_logits(mlp_logits, ada_logits, beta), labels)
    return beta, result


def train_ldc(config, model, train_loader, val_loader, test_loader, text_feats, device):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr / 10,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        max(1, config.epochs * len(train_loader)),
    )

    best_val_acc = -1.0
    best_payload = None
    for epoch in range(config.epochs):
        model.visual.adapter.train()
        model.adapter.train()
        loss_sum, acc_sum, count = 0.0, 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            clip_logits, mlp_logits, ada_logits, total_logits = model.my_forward(images, text_feats)
            loss = compute_loss(
                labels,
                clip_logits,
                mlp_logits,
                ada_logits,
                total_logits,
                config.loss_lambda,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item()
            acc_sum += accuracy(mlp_logits, labels)
            count += 1

        beta, val_result = evaluate(model, val_loader, text_feats, device)
        _, test_result = evaluate(model, test_loader, text_feats, device, beta=beta)
        print(
            "Epoch {:03d}: train_loss={:.4f} train_acc={:.2f} "
            "val_acc={:.2f} test_acc={:.2f} beta={:.2f}".format(
                epoch,
                loss_sum / max(count, 1),
                acc_sum / max(count, 1),
                val_result["acc"],
                test_result["acc"],
                beta,
            )
        )
        if val_result["acc"] > best_val_acc:
            best_val_acc = val_result["acc"]
            best_payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "beta": beta,
                "val": val_result,
                "test": test_result,
            }
            torch.save(best_payload, os.path.join(config.output_dir, "model_best.pth"))

    torch.save({"model": model.state_dict()}, os.path.join(config.output_dir, "model_last.pth"))
    print("Best payload: {}".format(best_payload))
    return best_payload


@torch.no_grad()
def smoke_ldc_forward(model, loader, text_feats, device):
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    clip_logits, mlp_logits, ada_logits, total_logits = model.my_forward(images, text_feats)
    print(
        "smoke_forward batch={} labels={} logits={}/{}/{}/{}".format(
            images.size(0),
            tuple(labels.shape),
            tuple(clip_logits.shape),
            tuple(mlp_logits.shape),
            tuple(ada_logits.shape),
            tuple(total_logits.shape),
        )
    )


def count_missing_paths(items):
    return sum(not os.path.exists(item.impath) for item in items)
