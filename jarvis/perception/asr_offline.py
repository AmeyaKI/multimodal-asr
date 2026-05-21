"""Parakeet offline ASR fallback (from legacy asr_class)."""

from __future__ import annotations

import numpy as np


def transcribe_parakeet(audio: np.ndarray) -> str:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    model_name = "nvidia/parakeet-ctc-0.6b"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.decode(ids[0], skip_special_tokens=True)
