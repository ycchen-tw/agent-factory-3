# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Quantizer config and registration for mxfp4_bf16_dequant:
MXFP4 weights stored as-is, dynamically dequantized to BF16 or FP8 during forward.
Supports dequant_dtype="bf16" (default) or "fp8" (FP8 E5M2 + per-token activation scale).
"""

from typing import TYPE_CHECKING

from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.auto import register_quantizer, register_quantization_config

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

from transformers.utils import (
    is_accelerate_available,
    is_kernels_available,
    is_torch_available,
    is_triton_available,
    logging,
)
from transformers.utils.quantization_config import Mxfp4Config
from transformers.quantizers.quantizers_utils import get_module_from_name

if is_torch_available():
    import torch
    from transformers.core_model_loading import WeightConverter

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@register_quantization_config("mxfp4_bf16_dequant")
class Mxfp4Bf16DequantConfig(Mxfp4Config):
    def __init__(self, dequant_dtype="bf16", fuse_swiglu=False, **kwargs):
        super().__init__(**kwargs)
        self.quant_method = "mxfp4_bf16_dequant"
        self.dequant_dtype = dequant_dtype  # "bf16" or "fp8"
        self.fuse_swiglu = fuse_swiglu

    def to_dict(self):
        d = super().to_dict()
        d["dequant_dtype"] = self.dequant_dtype
        d["fuse_swiglu"] = self.fuse_swiglu
        return d


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------
@register_quantizer("mxfp4_bf16_dequant")
class Mxfp4Bf16DequantQuantizer(HfQuantizer):
    """
    Dynamic dequant: MXFP4 weights stored as-is,
    dequantized to BF16 or FP8 E5M2 on-the-fly during forward pass.
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.triton_kernels_hub = None

    def _lazy_import_kernels(self):
        if self.triton_kernels_hub is None:
            try:
                from transformers.integrations.hub_kernels import get_kernel
                self.triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels")
            except ImportError:
                raise ImportError("kernels package is required for MXFP4-BF16 dequant quantization")
        return self.triton_kernels_hub

    def validate_environment(self, *args, **kwargs):
        if not is_torch_available():
            raise ImportError("Using mxfp4_bf16_dequant quantization requires torch")

        if self.quantization_config.dequantize:
            return

        assert torch.cuda.is_available() or torch.xpu.is_available(), \
            "MXFP4 dequant requires a GPU"
        assert is_accelerate_available(), \
            "MXFP4 dequant requires Accelerate: `pip install accelerate`"

        # FP8 mode requires Hopper GPU (SM >= 9.0) for FP8 matmul
        if getattr(self.quantization_config, "dequant_dtype", "bf16") == "fp8":
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability()
                if cc < (9, 0):
                    logger.warning_once(
                        "FP8 dequant requires Hopper GPU (SM >= 9.0), falling back to BF16"
                    )
                    self.quantization_config.dequant_dtype = "bf16"

        # BF16 dequant path uses standard bf16 matmul — no Hopper requirement.
        # Only need triton + kernels for the ragged matmul API.
        if torch.xpu.is_available():
            assert is_triton_available("3.5.0") and is_kernels_available(), \
                "MXFP4 dequant requires Triton >= 3.5.0 and kernels"
        else:
            assert is_triton_available("3.4.0") and is_kernels_available(), \
                "MXFP4 dequant requires Triton >= 3.4.0 and kernels"

        if not self.pre_quantized:
            self._lazy_import_kernels()

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded a BF16 dequant model on CPU. Set device_map='cuda' to use GPU."
            )
        elif isinstance(device_map, dict):
            if not self.pre_quantized and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "Cannot load BF16 dequant model with CPU/disk in device_map when quantizing on the fly."
                )

    def param_needs_quantization(self, model, param_name, **kwargs):
        from .quantization import Mxfp4Bf16DequantGptOssExperts
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, Mxfp4Bf16DequantGptOssExperts):
            if tensor_name in ["down_proj_bias", "gate_up_proj_bias"]:
                return False
            return True
        return False

    def _process_model_after_weight_loading(self, model, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()

    def _process_model_before_weight_loading(self, model, use_kernels=False, **kwargs):
        from .quantization import replace_with_dequant_experts
        print("[mxfp4_bf16_dequant] _process_model_before_weight_loading")

        if use_kernels:
            logger.warning_once(
                "Using full precision kernels, will dequantize to bf16."
            )
            self.quantization_config.dequantize = True

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        model = replace_with_dequant_experts(
            model, modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config
        )

    def update_tp_plan(self, config):
        if "GptOssConfig" in config.__class__.__name__:
            if getattr(config, "base_model_tp_plan", None) is not None:
                config.base_model_tp_plan.update({
                    "layers.*.mlp.experts.gate_up_proj_blocks": "grouped_gemm",
                    "layers.*.mlp.experts.gate_up_proj_scales": "grouped_gemm",
                    "layers.*.mlp.experts.down_proj_blocks": "grouped_gemm",
                    "layers.*.mlp.experts.down_proj_scales": "grouped_gemm",
                })
        return config

    def update_ep_plan(self, config):
        if "GptOssConfig" in config.__class__.__name__:
            if getattr(config, "base_model_ep_plan", None) is not None:
                config.base_model_ep_plan.update({
                    "layers.*.mlp.experts.gate_up_proj_blocks": "grouped_gemm",
                    "layers.*.mlp.experts.gate_up_proj_scales": "grouped_gemm",
                    "layers.*.mlp.experts.down_proj_blocks": "grouped_gemm",
                    "layers.*.mlp.experts.down_proj_scales": "grouped_gemm",
                })
        return config

    def get_state_dict_and_metadata(self, model):
        state_dict = model.state_dict()
        metadata = {}
        return state_dict, metadata

    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return True

    def get_quantize_ops(self):
        from .quantization import Mxfp4Fp8DequantQuantize
        return Mxfp4Fp8DequantQuantize(self)

    def get_weight_conversions(self):
        from .quantization import Mxfp4Fp8DequantDequantize, Mxfp4Fp8DequantDeserialize
        print(f"[mxfp4_bf16_dequant] get_weight_conversions (pre_quantized={self.pre_quantized})")

        if self.pre_quantized:
            if self.quantization_config.dequantize:
                return [
                    WeightConverter(
                        source_patterns=["_blocks", "_scales"],
                        target_patterns="",
                        operations=[Mxfp4Fp8DequantDequantize(self)],
                    )
                ]
            else:
                return [
                    WeightConverter(
                        source_patterns=["_blocks", "_scales"],
                        target_patterns="",
                        operations=[Mxfp4Fp8DequantDeserialize(self)],
                    )
                ]
        return []
