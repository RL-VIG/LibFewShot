# LDC reproduction

This integration reproduces **Logits DeConfusion with CLIP for Few-Shot
Learning (CVPR 2025)** inside the LibFewShot workspace. LDC uses fixed dataset
splits and CLIP adapters rather than LibFewShot's usual episodic training loop,
so it has a dedicated `run_ldc.py` entry point.

## External dependency

The original `clip_ldc` package is not copied into this repository. Clone the
authors' repository next to LibFewShot:

```bash
git clone https://github.com/LiShuo1001/LDC.git ../LDC
```

If it is stored elsewhere, either edit `ldc_root` in `config/ldc.yaml` or set
the `LDC_ROOT` environment variable. Dataset paths, downloaded CLIP weights,
and generated checkpoints remain local and are ignored by Git.

## Configuration

The example [`config/ldc.yaml`](../../config/ldc.yaml) targets EuroSAT 1-shot.
Set at least these values for a different experiment:

```yaml
ldc_root: ../LDC
data_root: ./data/EuroSAT
dataset_name: eurosat
path_prefix: "2750"
shot_num: 1
```

By default, the CoOp-style split JSON and CLIP cache are resolved below
`ldc_root`. You may override `split_json`, `clip_cache_dir`, or `output_dir`
explicitly in the YAML file.

## Commands

Run all commands from the LibFewShot repository root.

Check dataset and split paths without loading CLIP:

```bash
python run_ldc.py --config ./config/ldc.yaml --check-data
```

Run a single forward-pass smoke test:

```bash
python run_ldc.py --config ./config/ldc.yaml --smoke-forward
```

Train:

```bash
python run_ldc.py --config ./config/ldc.yaml
```

Evaluate the best checkpoint from the configured output directory:

```bash
python run_ldc.py --config ./config/ldc.yaml --eval-only
```

To evaluate another checkpoint, add `--checkpoint PATH`.
