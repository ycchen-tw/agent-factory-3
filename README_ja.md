# Agent Factory 3

[English](README.md) | [中文](README_zh.md) | 日本語

---

Agent Factory 3 は **gpt-oss** に特化した高性能なエージェント RL フレームワークです。限られたリソース下での効率的なエージェント強化学習に焦点を当てています。特徴は以下の通り：

1. gpt-oss の安定したエージェント RL：gpt-oss は既に優れたエージェント能力を持っており、本フレームワークは特定の業務シナリオにおける gpt-oss の性能を効率的に向上させます。
2. mxfp4 ベースの LoRA 訓練：gpt-oss の mxfp4 向け backward カーネルを実装し、FSDP2 マルチ GPU 訓練に対応しています。LoRA と組み合わせることで、2× H100 上で gpt-oss-120b のエージェント RL を効率的に実行できます。
3. 完全非同期パイプライン RL：rollout と訓練を別プロセスで並列実行し、ロングテール rollout を待たずに進行可能。R3、DDIS などのアルゴリズムにより訓練を安定化。
4. MCP ベースの環境管理：MCP server を通じて rollout 環境を簡単に定義できます。
5. DFlash による rollout 高速化:sglang に DFlash 投機的デコーディングを統合し、rollout スループットを大幅に向上。
6. Ray フリー:ray を使用せず、huggingface accelerate ベースで軽量かつデバッグしやすい訓練アーキテクチャを構築。
7. 豊富な実験 metrics:各種訓練指標を wandb に記録し、エージェント rollout の便利なビジュアライザーも内蔵。
8. 細部にわたる最適化:sequence packing、liger-kernel、activation checkpointing などにより、128k のロングコンテキストでも効率的に訓練可能。

## Installation

```bash
git clone https://github.com/ycchen-tw/agent-factory-3.git
cd agent-factory-3
uv sync                                 # メイン訓練環境
cd sglang_venv && uv sync               # sglang 推論環境（独立 venv）
```

## Quick Start

訓練の例：

- [examples/wordle](examples/wordle/README.md):英語 Wordle、4× H100
- [examples/minesweeper](examples/minesweeper/README.md):8×8 no-guess Minesweeper、4× H100

## Blogs

（近日公開予定）

## Changelog

- **2026-05-11**：初回公開リリース。

## Acknowledgements

Agent Factory 3 は以下のプロジェクトの上に成り立っています:

- [sglang](https://github.com/sgl-project/sglang) — 推論エンジン
- [transformers](https://github.com/huggingface/transformers) — モデルロードと mxfp4 quantizer
- [openai-harmony](https://github.com/openai/harmony) — gpt-oss tokenizer
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — fused kernels
- [accelerate](https://github.com/huggingface/accelerate) — FSDP2 ラッパー
- [PEFT](https://github.com/huggingface/peft) — LoRA アダプター
- [fastmcp](https://github.com/jlowin/fastmcp) — MCP client/server
- [trl](https://github.com/huggingface/trl) — RL training の参考
- [slime](https://github.com/THUDM/slime) — 非同期 RL パイプライン設計の参考
- [verl](https://github.com/volcengine/verl) — RL framework の参考
- DFlash — 投機的デコーディング（sglang fork に統合）

## License

Apache License 2.0。詳細は [LICENSE](LICENSE) を参照。
