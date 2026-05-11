# Agent Factory 3

English | [中文](README_zh.md) | [日本語](README_ja.md)

---

Agent Factory 3 is a high-performance agentic RL framework focused on efficient agentic reinforcement learning for **gpt-oss** under limited compute. Highlights:

1. Stable agentic RL for gpt-oss: gpt-oss already has strong agent capabilities; our framework efficiently lifts its performance on specific downstream tasks.
2. mxfp4-aware LoRA training: we wrote backward kernels for gpt-oss's mxfp4 quantization with FSDP2 multi-GPU support. Combined with LoRA, gpt-oss-120b agentic RL runs efficiently on 2× H100.
3. Fully async pipeline RL: rollout and training run in parallel on separate processes — no waiting for long-tail rollouts. Stabilized by R3 and DDIS.
4. MCP-based environment management: define rollout environments easily via MCP servers.
5. DFlash-accelerated rollouts: sglang integrates DFlash speculative decoding for significantly higher rollout throughput.
6. Ray-free: built on huggingface accelerate instead of ray — lightweight and easy to debug.
7. Rich training metrics: logs everything to wandb, including a convenient agent rollout visualizer.
8. Detail optimizations: sequence packing, liger-kernel, activation checkpointing — efficient training even at 128k long context.

## Installation

```bash
git clone https://github.com/ycchen-tw/agent-factory-3.git
cd agent-factory-3
uv sync                                 # main training env
cd sglang_venv && uv sync               # sglang inference env (separate venv)
```

## Quick Start

Training examples:

- [examples/wordle](examples/wordle/README.md): English Wordle, 4× H100
- [examples/minesweeper](examples/minesweeper/README.md): 8×8 no-guess Minesweeper, 4× H100

## Blogs

(coming soon)

## Changelog

- **2026-05-11**: Initial public release.

## Acknowledgements

Agent Factory 3 stands on these projects:

- [sglang](https://github.com/sgl-project/sglang) — inference engine
- [transformers](https://github.com/huggingface/transformers) — model loading and mxfp4 quantizer
- [openai-harmony](https://github.com/openai/harmony) — gpt-oss tokenizer
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — fused kernels
- [accelerate](https://github.com/huggingface/accelerate) — FSDP2 wrapper
- [PEFT](https://github.com/huggingface/peft) — LoRA adapter
- [fastmcp](https://github.com/jlowin/fastmcp) — MCP client/server
- [trl](https://github.com/huggingface/trl) — RL training reference
- [slime](https://github.com/THUDM/slime) — async RL pipeline design
- [verl](https://github.com/volcengine/verl) — RL framework reference
- [DFlash](https://github.com/z-lab/dflash) — speculative decoding

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Note

Parts of the documentation and code in this repo were drafted with LLM assistance.
Hallucinations (made-up numbers, plausible-but-wrong claims) may slip through review —
if something looks off, please open an issue.
