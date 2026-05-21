"""
run_pipeline.py — 一键跑通 QADC 全流程

步骤顺序：
  Step 0  下载数据集          (data/download_datasets.py)
  Step 1  生成伪标签           (data/generate_pseudo_labels.py)
  Step 2a 训练 QVFP-SL        (training/train_sl.py)
  Step 2b 训练 QVFP-RL x3     (training/train_rl.py)
  Step 3  主评估               (evaluation/evaluate.py)
  Step 4  消融实验              (experiments/ablations.py)
  Step 5  生成论文图表          (experiments/plot_results.py)

每步完成后会打印耗时。已有输出文件的步骤自动跳过（支持断点续跑）。

用法：
  python run_pipeline.py           # 跑全部步骤
  python run_pipeline.py --skip-rl # 跳过 RL 训练（节省时间）
  python run_pipeline.py --from 3  # 从 Step 3 开始（之前步骤已完成）
  python run_pipeline.py --only 0  # 只跑 Step 0
"""

import argparse
import os
import sys
import time

# 保证项目根目录在 sys.path 里
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ── 颜色输出 ─────────────────────────────────────────────────────────────────

def _c(code, msg): return f"\033[{code}m{msg}\033[0m"
def green(s):  return _c("32", s)
def yellow(s): return _c("33", s)
def red(s):    return _c("31", s)
def bold(s):   return _c("1",  s)


def banner(title):
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {bold(title)}")
    print(bar)


def done(elapsed):
    print(green(f"  ✓ 完成  ({elapsed:.1f}s)"))


def skip(reason):
    print(yellow(f"  → 跳过：{reason}"))


# ── 各步骤实现 ────────────────────────────────────────────────────────────────

def step0_download():
    """Step 0: 下载并缓存数据集"""
    from config import DATA_DIR
    train = os.path.join(DATA_DIR, "textvqa_train.json")
    val   = os.path.join(DATA_DIR, "textvqa_val.json")
    vqa2  = os.path.join(DATA_DIR, "vqav2_val.json")

    if all(os.path.exists(p) for p in [train, val, vqa2]):
        skip("3 个数据集 JSON 已存在")
        return

    from data.download_datasets import main
    main()


def step1_pseudo_labels():
    """Step 1: 生成伪标签 + 嵌入向量（可断点续跑）"""
    from config import LABEL_DIR
    labels_f = os.path.join(LABEL_DIR, "pseudo_labels.json")
    embeds_f = os.path.join(LABEL_DIR, "embeddings.pt")

    if os.path.exists(labels_f) and os.path.exists(embeds_f):
        import json, torch
        with open(labels_f) as f:
            n = len(json.load(f))
        from config import TEXTVQA_TRAIN_SAMPLES
        if n >= TEXTVQA_TRAIN_SAMPLES:
            skip(f"已有 {n} 条伪标签，全部完成")
            return
        else:
            print(f"  已完成 {n}/{TEXTVQA_TRAIN_SAMPLES}，继续生成 …")

    from data.generate_pseudo_labels import main
    main()


def step2a_train_sl():
    """Step 2a: 监督学习训练 QVFP-SL"""
    from config import CKPT_DIR
    ckpt = os.path.join(CKPT_DIR, "qvfp_sl_concat_3lvl_best.pt")

    if os.path.exists(ckpt):
        import torch
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        if meta.get("normalize_inputs", False):
            skip(f"SL checkpoint 已存在: {ckpt}")
            return
        print("  检测到旧版 SL checkpoint（缺少输入归一化），将重新训练 …")

    from training.train_sl import train_qvfp_sl
    train_qvfp_sl(fusion_mode="concat", num_levels=3)


def step2b_train_rl():
    """Step 2b: GRPO 强化学习训练 QVFP-RL（3 个 λ 值）"""
    from config import CKPT_DIR, RL_LAMBDA_VALUES
    from training.train_rl import train_qvfp_rl

    for lam in RL_LAMBDA_VALUES:
        ckpt = os.path.join(CKPT_DIR, f"qvfp_rl_lambda{lam}_best.pt")
        if os.path.exists(ckpt):
            import torch
            meta = torch.load(ckpt, map_location="cpu", weights_only=False)
            if meta.get("normalize_inputs", False):
                skip(f"RL λ={lam} checkpoint 已存在")
                continue
            print(f"  检测到旧版 RL λ={lam} checkpoint，将重新训练 …")
        print(f"  训练 RL λ={lam} …")
        train_qvfp_rl(lam=lam, rl_samples=1000)


def step3_evaluate():
    """Step 3: 主评估（所有 baseline + QADC）"""
    from config import DATA_DIR, LABEL_DIR, RESULT_DIR
    import json
    out = os.path.join(RESULT_DIR, "main_results.json")

    if os.path.exists(out):
        oracle_cache = os.path.join(LABEL_DIR, "oracle_val_levels.json")
        expected_keys = []
        for split_name, fname in [
            ("textvqa_val", "textvqa_val.json"),
            ("vqav2_val", "vqav2_val.json"),
        ]:
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    expected_keys.extend(
                        f"{split_name}:{sample['id']}" for sample in json.load(f)
                    )

        if os.path.exists(oracle_cache):
            with open(oracle_cache) as f:
                oracle_map = json.load(f)
            missing = [key for key in expected_keys if key not in oracle_map]
        else:
            missing = expected_keys

        if not missing:
            skip(f"main_results.json 已存在，oracle 缓存覆盖当前 {len(expected_keys)} 条验证样本")
            return

        print(f"  main_results.json 已存在，但 oracle 缓存缺失 {len(missing)} 条；重新评估 …")

    from evaluation.evaluate import main
    main()


def step4_ablations():
    """Step 4: 消融实验"""
    from config import CKPT_DIR, RESULT_DIR
    import torch
    files = ["ablation_fusion.json", "ablation_levels.json", "ablation_keywords.json"]
    paths = [os.path.join(RESULT_DIR, f) for f in files]
    ckpts = [
        os.path.join(CKPT_DIR, f"qvfp_sl_{mode}_3lvl_best.pt")
        for mode in ["concat", "cross_attn", "query_only", "image_only"]
    ] + [
        os.path.join(CKPT_DIR, f"qvfp_sl_concat_{n}lvl_best.pt")
        for n in [2, 3, 4]
    ]

    def current_checkpoint(path):
        if not os.path.exists(path):
            return False
        meta = torch.load(path, map_location="cpu", weights_only=False)
        return bool(meta.get("normalize_inputs", False))

    stale_ckpts = [p for p in ckpts if os.path.exists(p) and not current_checkpoint(p)]
    if all(os.path.exists(p) for p in paths):
        existing_ckpts = [p for p in ckpts if os.path.exists(p)]
        newest_result = min(os.path.getmtime(p) for p in paths)
        newest_ckpt = max([os.path.getmtime(p) for p in existing_ckpts], default=0)
        if not stale_ckpts and newest_result >= newest_ckpt:
            skip("3 个消融 JSON 文件均已存在")
            return
        print("  检测到旧版/更新后的消融 checkpoint，将重新生成消融实验 …")

    from experiments.ablations import main
    main()


def step5_figures():
    """Step 5: 生成论文图表"""
    from config import RESULT_DIR
    fig_dir = os.path.join(RESULT_DIR, "figures")
    if os.path.isdir(fig_dir) and len(os.listdir(fig_dir)) >= 6:
        skip(f"figures/ 目录已有 {len(os.listdir(fig_dir))} 个文件")
        return

    from experiments.plot_results import (
        plot_main_results, plot_training_curves,
        plot_fusion_ablation, plot_levels_ablation,
        plot_keyword_analysis, plot_sl_vs_rl,
    )
    plot_main_results()
    plot_training_curves()
    plot_fusion_ablation()
    plot_levels_ablation()
    plot_keyword_analysis()
    plot_sl_vs_rl()


# ── 步骤注册表 ────────────────────────────────────────────────────────────────

STEPS = [
    (0,  "下载数据集",          step0_download),
    (1,  "生成伪标签",          step1_pseudo_labels),
    (2,  "训练 QVFP-SL",        step2a_train_sl),
    (3,  "训练 QVFP-RL",        step2b_train_rl),
    (4,  "主评估",              step3_evaluate),
    (5,  "消融实验",            step4_ablations),
    (6,  "生成图表",            step5_figures),
]


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QADC 全流程一键运行")
    parser.add_argument("--from",    dest="from_step", type=int, default=0,
                        help="从第几步开始（默认 0）")
    parser.add_argument("--only",    dest="only_step", type=int, default=None,
                        help="只运行某一步")
    parser.add_argument("--skip-rl", action="store_true",
                        help="跳过 RL 训练（Step 3，节省约 30 分钟）")
    args = parser.parse_args()

    print(bold("\nQADC Pipeline Runner"))
    print(f"环境: {'Google Colab' if _on_colab() else '本地'}\n")

    total_start = time.time()

    for step_id, name, fn in STEPS:
        # 过滤逻辑
        if args.only_step is not None and step_id != args.only_step:
            continue
        if step_id < args.from_step:
            continue
        if args.skip_rl and step_id == 3:
            banner(f"Step {step_id}: {name}")
            skip("--skip-rl 已设置")
            continue

        banner(f"Step {step_id}: {name}")
        t0 = time.time()
        try:
            fn()
            done(time.time() - t0)
        except KeyboardInterrupt:
            print(red("\n  用户中断，进度已保存（支持断点续跑）"))
            sys.exit(0)
        except Exception as e:
            print(red(f"\n  ✗ Step {step_id} 出错: {e}"))
            import traceback; traceback.print_exc()
            print(red(f"  流程中止，请修复后重新运行（加 --from {step_id} 可从此步继续）"))
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{green(bold('全部流程完成！'))}  总耗时: {total/60:.1f} 分钟")
    print(f"结果目录: {_result_dir()}")


def _on_colab():
    try:
        import google.colab; return True
    except ImportError:
        return False


def _result_dir():
    from config import RESULT_DIR
    return RESULT_DIR


if __name__ == "__main__":
    main()
