from __future__ import annotations

import gc
import logging
import os

import pytest
import torch

from vlmbox import VLM

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("VLMBOX_RUN_GPU_INTEGRATION") != "1",
        reason="set VLMBOX_RUN_GPU_INTEGRATION=1 to load cached VLM checkpoints",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def test_internvl_uses_untied_embeddings_without_tying_warning(caplog) -> None:
    caplog.set_level(logging.WARNING)
    vlm = VLM.from_pretrained("OpenGVLab/InternVL3_5-2B-HF", local_files_only=True)
    try:
        assert vlm.model.config.tie_word_embeddings is False
        input_embeddings = vlm.model.model.language_model.embed_tokens.weight
        output_embeddings = vlm.model.lm_head.weight
        assert input_embeddings.data_ptr() != output_embeddings.data_ptr()
        assert not any(
            "tied weights mapping" in record.message for record in caplog.records
        )

        inputs = vlm.prepare([{"role": "user", "content": "Say hello."}])
        inputs = inputs.to(vlm.input_device)
        with torch.inference_mode():
            logits = vlm.model(**inputs).logits
        assert logits.ndim == 3
        assert torch.isfinite(logits).all()
    finally:
        del vlm
        gc.collect()
        torch.cuda.empty_cache()
