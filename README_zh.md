# Agent Factory 3

[English](README.md) | 中文 | [日本語](README_ja.md)

---

Agent Factory 3 是一個高性能的 agentic RL framework，我們專注於**gpt-oss**於有限資源下的高效能agentic強化學習，我們的框架有以下特點：

1. 穩定的 gpt-oss 的 agentic 強化學習： gpt-oss 已經有相當優秀的 agent 能力，我們的框架可以高效的提昇 gpt-oss 在特定業務場景的表現。
2. 基於 mxfp4 的 lora 訓練：我們為 gpt-oss 的 mxfp4 撰寫 backward kernel，並支援 fsdp2 多卡訓練。與 lora 結合，我們可以在 2xH100 上高效的進行 gpt-oss-120b 的 agentic RL。
3. 完全異步 pipeline RL：rollout 與 training 在獨立 process 並行執行，不用等待長尾 rollout。 並透過 R3, DDIS 等演算法穩定訓練。
4. 基於 mcp 的環境管理：使用者可以透過 mcp server 容易的定義 rollout 環境。
5. dflash 加速 rollout：sglang 整合 DFlash 投機解碼，顯著提昇 rollout throughput。
6. Ray free ：不使用 ray 做管理，而是基於 huggingface accelerate 建構輕量好 debug 的訓練架構
7. 豐富的實驗 metrics：紀錄各種訓練指標到 wandb，並包含非常方便的 agent rollout visualizer
8. 各種細節優化：透過 sequence packing, liger-kernel, activation checkpointing 等等，agent factory 3 在 128k 的 long context 下也能高效訓練。

## Installation

```bash
git clone https://github.com/ycchen-tw/agent-factory-3.git
cd agent-factory-3
uv sync                                 # 主訓練環境
cd sglang_venv && uv sync               # sglang 推論環境（獨立 venv）
```

## Quick Start

跑訓練範例：

- [examples/wordle](examples/wordle/README_zh.md)：英文 Wordle，4 卡 H100
- [examples/minesweeper](examples/minesweeper/README_zh.md)：8×8 no-guess Minesweeper，4 卡 H100

## Blogs

(尚未發佈)

## Changelog

- **2026-05-11**：首次公開發佈。

## Acknowledgements

Agent Factory 3 站在這些 project 上：

- [sglang](https://github.com/sgl-project/sglang) — 推論引擎
- [transformers](https://github.com/huggingface/transformers) — model loading 與 mxfp4 quantizer
- [openai-harmony](https://github.com/openai/harmony) — gpt-oss tokenizer
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — fused kernels
- [accelerate](https://github.com/huggingface/accelerate) — FSDP2 wrapper
- [PEFT](https://github.com/huggingface/peft) — LoRA adapter
- [fastmcp](https://github.com/jlowin/fastmcp) — MCP client/server
- [trl](https://github.com/huggingface/trl) — RL training 參考
- [slime](https://github.com/THUDM/slime) — async RL pipeline 設計參考
- [verl](https://github.com/volcengine/verl) — RL framework 參考
- [DFlash](https://github.com/z-lab/dflash) — 投機解碼演算法

## License

Apache License 2.0。詳情看 [LICENSE](LICENSE)。

## 備註

本 repo 的部分文件與程式碼透過 LLM 協助撰寫，可能會有幻覺（憑空的數字、看似合理但實際錯誤的描述）漏網。
發現問題請提 issue。