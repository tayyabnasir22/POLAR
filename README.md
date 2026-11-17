<!-- TODO before publishing: replace [PAPER TITLE], the BibTeX fields, and the checkpoint link below. -->

# POLAR: [PAPER TITLE]

A centralized repository for **training** and **evaluating** the POLAR model at different scaling factors, on both synthetic and real-world low-resolution depth.

## Citation

If you use this repository or build upon it in your research, please cite:

> **POLAR: [PAPER TITLE]**

```bibtex
@article{polar2026,
  title   = {POLAR: [PAPER TITLE]},
  author  = {[AUTHORS]},
  journal = {arXiv preprint arXiv:[ARXIV ID]},
  year    = {2026}
}
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is required.

Training from scratch initializes the frozen DINOv3 ViT-S/16 teacher from `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`. Request the weights from the [DINOv3 repository](https://github.com/facebookresearch/dinov3) and place the file in `~/.cache/torch/hub/checkpoints/`. The file is not needed for testing or for resuming a run, since POLAR checkpoints already contain the teacher weights.

### Training

1. Open `train.py`.
2. Make the desired changes to dataset (`BENCHMARK`), batch size, repeat, and other settings.
3. Run the script to begin model training:

```bash
python train.py [scale] polar   # synthetic (bicubic) low-resolution depth
python train.py [scale] real    # real-world sensor depth
```

- `polar` trains `POLAR` and is meant for the synthetic benchmarks (`NYUV2`, `RGBDD`, `TOFDSRD`, `HYPERSIM`, `DIML`).
- `real` trains `POLAR_Real`, which adds the depth restoration branch, and is meant for `RGBDDReal` and `TOFDSRDReal`. Here the low-resolution depth comes from the sensor, so `[scale]` is only used to name the checkpoint folder.
- Datasets are read from `PathManager.BASE_PATH` (default `./`), one folder per benchmark, in the format produced by the data processing repository linked below. Synthetic training also reports RMSE on `./MIDDLE` after every epoch, so that folder must be present.
- If a `last.pth` already exists for the same configuration, training resumes from it.

### Testing

1. Open `test.py`.
2. Uncomment the desired testing scale and datasets.
3. Run the script to evaluate your trained model:

```bash
python test.py
```

The checkpoint used by each test is set in `TestingOrchestrators/TestingOrchestrator.py` and defaults to a folder under `./StableWeights/`. Copy your trained (or downloaded) checkpoint folder there, or edit the path. Results are appended to `metrics.csv` inside the checkpoint folder.

## Configurations

The repository includes several configurable components to enable flexible experimentation:

- **Training and evaluation strategies** — classes within the `Pipelines` folder
- **Datasets** — the `BenchmarkType` class
- **Model variants** — the `ModelType` class (`POLAR`, `POLAR_Real`)
- **Scale, batch size, and repeat** — the `TrainingOrchestrator` class, set from `train.py`

## Saving and Loading Models

The model saving and loading mechanism automatically constructs model paths based on the configuration parameters used during training. This ensures that each model checkpoint is uniquely identified by its model variant, dataset, patch size, and scale, allowing for seamless testing and reproducibility.

### How It Works

Each training run saves its model weights under a directory name that encodes the full configuration, for example:

```
model_states_POLAR_NYUV2_Patch_448_Scale_8/
├── last.pth        # written after every epoch, used for resuming and testing
├── epoch_5.pth     # snapshot every 5 epochs
├── log.txt
└── tensorboard/
```

When performing evaluation, the **same configuration** must be used to correctly locate and load the desired checkpoint.

## Checkpoints

Pretrained checkpoints are available [here]([CHECKPOINT LINK]).

## Related Repositories

- **Data processing:** [GDSR-Data-Preperation](https://github.com/tayyabnasir22/GDSR-Data-Preperation)
- **Base code:** This code is built on top of [NAIMA-GDSR](https://github.com/tayyabnasir22/NAIMA-GDSR). If you find this code useful, please cite those works as well.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).