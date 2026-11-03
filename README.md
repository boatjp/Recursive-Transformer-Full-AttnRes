# Recursive Transformer + Full AttnRes

日本語Causal Language Modelのモデル本体とユニットテストです。学習ループ・データセット処理・事前学習は含みません。既存Tokenizerの作成手順は `tokenizer/README.md` を参照してください。

## セットアップと検証

Python 3.13、PyTorch 2.15.0+cpu、SentencePiece 0.2.1で検証済みです。

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe recursive_transformer.py
```

最後のコマンドは実際の `tokenizer/output-full/tokenizer.model` から語彙数・特殊IDを読み、両方式のパラメータ内訳を表示します。メタデバイス上で構築するため、モデルの実メモリ確保や学習は行いません。別のTokenizerは `--tokenizer path/to/tokenizer.model` で指定します。SentencePieceの未定義ID（-1）はConfigで `None` とします。

実ファイルから語彙数16,384、PAD=3、BOS=1、EOS=2、UNK=0を確認しました。

## 使用例

```python
import sentencepiece as spm
import torch
from recursive_transformer import ModelConfig, RecursiveTransformerLM, set_seed

set_seed(42)
tokenizer_path = "tokenizer/output-full/tokenizer.model"
tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
config = ModelConfig.from_tokenizer(tokenizer_path, residual_type="attnres")
model = RecursiveTransformerLM(config)
input_ids = torch.tensor([tokenizer.encode("日本語の言語モデルを実装します。")])
logits = model(input_ids)
result = model(input_ids, labels=input_ids, return_details=True)
result.loss.backward()
print(logits.shape, result.loss.item())
print(model.parameter_counts())
```

`residual_type="residual"` で通常Residualへ切り替えます。`forward(input_ids)` は `[batch, sequence, vocabulary]` のTensor、`labels` または `return_details=True` を渡した場合は `CausalLMOutput` を返します。`labels` は入力と同じ形状で、内部で次トークン予測へシフトします。`-100` はloss対象外です。

`attention_mask` は `[batch, sequence]` の0/1マスクです。指定時はこのマスクを優先し、省略時はTokenizerのpad IDから推定します。pad ID未定義の場合は全トークンを有効とします。左右Paddingに対応し、RoPE位置は有効トークンの累積数から作ります。Paddingをまたぐ予測ペアはlossから除外し、全Padding・全labels無視・長さ1の場合は微分可能なゼロlossを返します。1行は1つの系列として扱うため、複数文書のpacked attentionは未実装です。

bf16は `model.to(torch.bfloat16)` または `with torch.autocast("cpu", dtype=torch.bfloat16):` に対応します。GPUではモデルと入力をCUDAへ移し、autocastのデバイスを `"cuda"` にします。今回の実行検証対象はCPUです。RMSの統計、RoPE角度、AttnResのスコア・Softmax・重み付き集約、lossはfp32で計算します。

## 構造と重み共有

既定値はHidden 640、10 heads × 64、FFN 10,240、Embedding rank 192、再帰16回です。`block` は物理的に1個だけで、Q/K/V/O、Gate、FFNの上下Linearを全ステップで再利用します。入力は `Embedding(V,192) → Linear(192,640)`、出力は `Final RMSNorm → Linear(640,192) → Linear(192,V)` です。入力と出力の行列はすべて非共有です。

AttentionはCausal SDPAです。各headの64次元Q/Kを別々のRMSNormで正規化した後にRoPEを適用し、SDPA既定の `1/sqrt(head_dim)` スケーリングを使います。Q/K Normのscaleはhead間で共有、再帰ステップ間では独立です。Gateは `sigmoid(Linear(hidden, num_heads))` とし、トークン別・head別にSDPA出力へ乗算してからO projectionします。GateのLinearも再帰間で共有し、biasはありません。FFNは `down(relu(up(x)) ** 2)` です。

[Recursive Transformer論文の§4](https://arxiv.org/html/2608.26973v1#S4)では、Attention/FFN用Normに追加biasがあり、QK-Normもステップ別です。本実装では各Attention/FFN入力を `RMSNorm(x) * scale[step] + bias[step]` としてから共有サブレイヤーへ渡します。biasはLinearではなくNorm後の加算です。QK-Normにはbiasを付けません。

## Full AttnResと最終Readout

[Attention Residuals論文の§3.1、式(1)–(4)、§3.2の最終出力](https://arxiv.org/pdf/2603.15031)に基づき、履歴は初期Embedding表現と各Attention/FFNの生の出力です。通常の加算Residualは併用しません。各トークンについて以下を計算します。

```text
history: [batch, sequence, depth, hidden]
scores[b,t,d] = dot(query, parameter_free_rmsnorm(history[b,t,d]))
weights = softmax(scores, dim=depth)
input[b,t] = sum_d(weights[b,t,d] * history[b,t,d])
```

深さAttentionに `1/sqrt(hidden)` は掛けません。KeyだけパラメータなしRMSNormを適用し、Valueは未正規化の履歴です。履歴はdetachせず、トークン間・バッチ間を混合しません。32個のサブレイヤー入力Queryと1個の最終Readout専用Queryは独立した学習パラメータで、`queries.shape == (33,640)` です。

最終FFNまで含む33要素の履歴も集約し、その後Final RMSNormとLM Headへ渡します。**最終Readoutは最後のFFN入力Queryとは独立した専用Queryを使用します。** `queries[0:32]` がサブレイヤー入力用、`queries[32]`（`queries[-1]`）がReadout用です。当初の32 Query構成から専用Queryを追加したため、合計33個・21,120パラメータとなり、640パラメータ増えました。一般には `2 * num_steps + 1` 個です。`return_details=True` の `depth_weights` は32回の入力集約と最終Readoutの計33 Tensorで、各形状は `[batch, sequence, history_depth]` です。

最初のAttention入力は候補がEmbeddingのみなので、そのQueryの勾配は数学的に常にゼロです。独立Query数を守るため、このベクトルも保持します。Queryをゼロ初期化するため初期状態は深さ方向の均等重みですが、学習後はトークン内容に応じた重みになり得ます。

## 初期化と論文との差異

Embedding・Linearは平均0、標準偏差0.02の正規分布、Padding行は0、Norm scaleは1、Norm biasとAttnRes Queryは0です。Attention OとFFN downの標準偏差は `0.02 / sqrt(2 * num_steps)` に縮小します。ベース論文Appendix BのO/downゼロ初期化とは異なります。両比較モデルで同じ初期化方式を使い、最初のBackwardから内部projectionへの勾配経路を検証します。`set_seed` はPythonとPyTorchのCPU/CUDA seedを設定しますが、異なるデバイス・PyTorch版間のビット単位一致を保証するものではありません。

ベース論文の英語BPE（32,768語彙）を既存の日本語SentencePiece（16,384語彙）へ置換し、FlashAttentionの可変長packed処理を通常のSDPA＋Padding/Causalマスクへ置換しています。Recursive共有ブロックとAttnResの組み合わせ、パラメータなしのAttnRes Key Normは今回の比較設計です。`task.md` の当初の32 Query指定に対しては、最終Readout用に1個を追加する設計へ変更しています。

## 正確なパラメータ数

16,384語彙・既定サイズで、`parameter_counts()` が実際のParameterから重複なく集計します。

| 内訳 | residual | attnres |
|---|---:|---:|
| Attention（Gate含む） | 1,644,800 | 1,644,800 |
| FFN | 13,107,200 | 13,107,200 |
| 入力Embedding＋projection | 3,268,608 | 3,268,608 |
| LM Head＋projection | 3,268,608 | 3,268,608 |
| ステップ別Normとbias（QK含む） | 43,008 | 43,008 |
| Final RMSNorm | 640 | 640 |
| AttnRes Query（Readout専用Query含む） | 0 | 21,120 |
| **合計** | **21,332,864** | **21,353,984** |

目安の約21.37Mとの差は構造を変更して埋めず、上記のNorm・bias・Gate・Readoutの仕様で報告します。

## テスト対象

`tests/test_recursive_transformer.py` は、形状、正確なパラメータ数、全ステップの共有モジュール呼び出し、Norm/bias/QK-Normの独立性、Query数・形状、Depth Softmaxの和、数値例との一致、トークンごとの重み変化、トークン分離、未来情報漏洩防止、fp32/bf16のForward/BackwardとNaN/Inf、最終FFNのlogitsへの影響と履歴の勾配、Embedding/Head非共有、左右/全Padding、lossシフト・無視ラベル、seed再現性、実Tokenizerのメタデータを検査します。通常Residualが混入していないことも、ゼロQuery時の全履歴平均との一致で確認します。

詳細テストは小型Configを使い、さらに実Tokenizer＋指定どおりの640次元・16ステップ構成で両方式のForward/Backwardを実行します。最適化ステップや事前学習は行いません。

2026-11-01の検証結果はモデル16テスト成功（5.613秒）、既存Tokenizer 8テスト成功（0.230秒）です。指定サイズの両モデルでfp32のForward/Backward、小型の両モデルでネイティブbf16とCPU autocastのForward/Backwardを確認しました。モデル情報表示コマンドも成功しています。SentencePiece由来のSWIG DeprecationWarningは出ますが、テスト失敗はありません。

2026-11-03にReadout専用Queryを追加した構成でモデル17テストが成功しました（8.445秒）。Query数・パラメータ数の期待値を更新し、最終Readoutだけが専用Queryを使うことと、そのQueryへ勾配が流れることを検証しています。
