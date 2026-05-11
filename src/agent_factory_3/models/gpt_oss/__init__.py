"""gpt-oss-120b model-specific support.

This package bundles everything needed to train gpt-oss-120b under this RL
framework: an mxfp4-to-bf16/fp8 dequant quantizer, supporting Triton
kernels, an FSDP2 monkey-patch, and the routing-replay activator.

Typical usage in a training script:

    # 1. Apply the FSDP2 BF16 fix and register the quantizer (side effects).
    from agent_factory_3.models.gpt_oss import fsdp2_fix         # noqa: F401
    from agent_factory_3.models.gpt_oss import quantizer_config  # noqa: F401

    # 2. Public API.
    from agent_factory_3.models.gpt_oss import (
        Mxfp4Bf16DequantConfig,
        enable_routing_replay,
    )

    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.quantization_config = Mxfp4Bf16DequantConfig(dequant_dtype="bf16").to_dict()
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, dtype=torch.bfloat16)
    enable_routing_replay(model)
"""
from .quantizer_config import Mxfp4Bf16DequantConfig
from .routing_replay import enable_routing_replay

__all__ = ["Mxfp4Bf16DequantConfig", "enable_routing_replay"]
