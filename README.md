# QADC: Query-Aware Dynamic Compression for Efficient MLLMs

## Project Structure
```
QADC/
├── README.md
├── requirements.txt
├── config.py                          # All hyperparameters in one place
├── data/
│   ├── download_datasets.py           # Step 0: download TextVQA + VQAv2
│   └── generate_pseudo_labels.py      # Step 1: generate pseudo labels
├── models/
│   └── qvfp.py                        # QVFP module definition
├── training/
│   ├── train_sl.py                    # Step 2a: supervised learning
│   └── train_rl.py                    # Step 2b: RL (GRPO)
├── evaluation/
│   └── evaluate.py                    # Step 3: run all benchmarks
├── experiments/
│   └── ablations.py                   # Step 4: all ablation studies
└── utils/
    ├── vlm_inference.py               # Qwen2.5-VL wrapper
    └── metrics.py                     # VQA accuracy + token metrics
```

## Colab Setup (run once)
```python
!pip install transformers datasets torch torchvision \
             qwen-vl-utils accelerate tqdm pillow \
             scikit-learn matplotlib pandas -q

from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/drive/MyDrive/QADC')
```

## Execution Order
```
Session 1 (~2h):  python data/download_datasets.py
                  python data/generate_pseudo_labels.py
Session 2 (~1h):  python training/train_sl.py
                  python training/train_rl.py
Session 3 (~1h):  python evaluation/evaluate.py
                  python experiments/ablations.py
```
