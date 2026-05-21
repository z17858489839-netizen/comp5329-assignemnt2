"""
utils/vlm_inference.py
Thin wrapper around Qwen2.5-VL-3B for batched + single inference.
Handles model loading, image resizing, and answer extraction.

Auto-detects CUDA availability so the same code runs on Colab (GPU)
and locally (CPU / Apple Silicon MPS).
"""

import torch
import string
from PIL import Image
import transformers
from transformers import AutoProcessor
from packaging.version import Version

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Run: pip install qwen-vl-utils")


# ── Device / dtype helpers ────────────────────────────────────────────────────

def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # MPS skipped: Qwen2.5-VL vision encoder triggers a hard LLVM crash on MPS
    # (incompatible matmul shapes in the visual RoPE kernel, unfixable via fallback).
    return "cpu"


def _default_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.bfloat16  # ~6.7 GB on CPU; fits in 16 GB unified memory


# ── Answer normalisation (standard VQA eval) ─────────────────────────────────

def normalise_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def is_correct(prediction: str, gt_answers: list) -> bool:
    pred = normalise_answer(str(prediction))
    return any(normalise_answer(str(gt)) == pred for gt in gt_answers)


# ── Image resizing ────────────────────────────────────────────────────────────

def resize_image(image: Image.Image, ratio: float) -> Image.Image:
    if ratio == 1.0:
        return image
    w, h = image.size
    new_w = max(28, int(w * ratio))
    new_h = max(28, int(h * ratio))
    return image.resize((new_w, new_h), Image.LANCZOS)


# ── Model loader (singleton pattern — load once per session) ──────────────────

_model     = None
_processor = None


def _require_qwen25_vl_support() -> None:
    min_version = Version("4.49.0")
    current = Version(transformers.__version__)
    if current < min_version:
        raise RuntimeError(
            "Qwen2.5-VL requires transformers>=4.49.0. "
            f"Loaded transformers=={transformers.__version__}. "
            "Install/upgrade transformers, restart the Python runtime, then rerun from Cell 0b/2a."
        )


def _model_loader_class():
    for name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "Qwen2_5_VLForConditionalGeneration",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "Could not find a Qwen2.5-VL compatible model loader in transformers. "
        f"Loaded transformers=={transformers.__version__}; install transformers>=4.49.0 "
        "and restart the Python runtime."
    )


def load_model(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device:   str | None = None,
) -> tuple:
    """
    Load Qwen2.5-VL and its processor.  Call once; subsequent calls return
    the cached instance.

    device: "cuda" | "mps" | "cpu" | None (auto-detect)
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    if device is None:
        device = _default_device()

    _require_qwen25_vl_support()

    dtype = _default_dtype(device)
    print(f"Loading {model_id}  [device={device}, dtype={dtype}] …")

    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_loader = _model_loader_class()

    device_map = "auto" if device == "cuda" else device

    # Prefer the generic multimodal auto-loader when available, otherwise use
    # Qwen2.5-VL's dedicated class to avoid the qwen2_5_vl -> qwen2_vl mismatch.
    load_kwargs = dict(torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
    try:
        _model = model_loader.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **load_kwargs
        )
        print("Flash Attention 2 enabled.")
    except (ValueError, ImportError):
        _model = model_loader.from_pretrained(model_id, **load_kwargs)
    _model.eval()
    print(f"Model loaded. class={type(_model).__name__}")
    return _model, _processor


def unload_model() -> None:
    global _model, _processor
    _model     = None
    _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Single-sample inference ───────────────────────────────────────────────────

def infer_single(
    model,
    processor,
    image: Image.Image,
    query: str,
    max_new_tokens: int = 64,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "Answer with a single word or short phrase only. No explanation.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  query},
            ],
        },
    ]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, *_ = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ── Extract frozen embeddings for QVFP ───────────────────────────────────────

def extract_embeddings(
    model,
    processor,
    image: Image.Image,
    query: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        q_emb : (hidden_dim,)  — mean-pooled text embedding
        v_emb : (hidden_dim,)  — mean-pooled visual embedding (1/4-res input)

    Both are detached float32 CPU tensors.
    """
    # ── Text embedding ────────────────────────────────────────────────────────
    text_tokens = processor.tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(model.device)

    with torch.no_grad():
        # get_input_embeddings() works for Qwen2VL and Qwen2_5VL alike
        text_out = model.get_input_embeddings()(text_tokens["input_ids"])  # (1, L, D)
    q_emb = text_out[0].mean(dim=0).float().cpu()                          # (D,)

    # ── Visual embedding (always 1/4-res as QVFP input) ──────────────────────
    small_image = resize_image(image, 0.25)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": small_image},
                {"type": "text",  "text":  "describe this image"},
            ],
        }
    ]
    text_input   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, *_ = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[-1]: last layer only, shape (1, seq_len, D)
    # Using [-1] avoids materialising all intermediate layers in Python
    v_emb = outputs.hidden_states[-1][0].mean(dim=0).float().cpu()  # (D,)
    return q_emb, v_emb
