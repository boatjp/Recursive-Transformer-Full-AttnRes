# Japanese FineWiki SentencePiece Tokenizer

`HuggingFaceFW/finewiki` の日本語サブセットをストリーミングで取得し、日本語10,000,000語のTrainコーパスと SentencePiece Unigram Tokenizer を作成します。Transformerモデルや事前学習コードは含みません。

## 設計

- データセット: `HuggingFaceFW/finewiki`, config `ja`, split `train`, `streaming=True`
- 本文列: `text`（開始時に実際のスキーマを検査）
- シャッフル: Hugging Face IterableDataset の seed 固定バッファシャッフル
- 除外: 空文書、SudachiPy SplitMode.C で50語未満の文書（変更可能）、本文重複
- 重複判定: 改行コード統一と外側空白除去を**ハッシュ計算だけ**に用いた SHA-256 完全一致。本文は変更しない
- メモリ対策: 重複ハッシュはSQLite、データは逐次書き込み、評価も行単位
- 分割: 文書IDとseedのSHA-256値による Train/Validation/Test = 90/5/5。ストリーム順に依存しない
- 語数: SudachiPy `Tokenizer.SplitMode.C` の形態素数。Trainが目標語数以上になった文書で停止するため、文書1件分だけ超過し得る
- Tokenizer学習入力: `train.txt` の生テキストのみ。SudachiPyの分かち書きは使わない
- 再現性: seed、全設定、集計方法を `tokenizer_stats.json` に保存。SentencePieceは入力シャッフル無効・1 thread

FineWikiの本文には見出し、表、リスト等のMarkdown風表現が含まれます。本処理では、それらを不必要に削除・正規化しません。各文書の後に区切りとして空行を1行だけ追加します。

## セットアップ

Python 3.11を推奨します。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Hugging Face Hubへアクセスできるネットワーク環境が必要です。認証が必要な環境では事前に `HF_TOKEN` を設定してください。

## まず小規模動作確認

```powershell
python tokenizer_pipeline.py --mode smoke --output-dir output-smoke
```

smokeモードは Train 50,000語、語彙1,024、シャッフルバッファ1,000です。これは配線確認用で、本番Tokenizerではありません。値は明示的に上書きできます。

```powershell
python tokenizer_pipeline.py --mode smoke --target-train-words 100000 --vocab-size 2048
```

SentencePieceは指定語彙数を厳密に要求します。小さすぎるコーパスに大きな語彙数を指定すると、設定を自動変更せずエラー終了します。

依存関係の導入後、ネットワークを使わない単体テストは次で実行できます。

```powershell
python -m unittest -v test_tokenizer_pipeline.py
```

## 本番実行

```powershell
python tokenizer_pipeline.py --mode full --output-dir output-full
```

fullモードの固定既定値は次の通りです。

- Train目標: 10,000,000 Sudachi形態素
- 語彙数: 16,384
- model type: Unigram
- character coverage: 0.9995
- byte fallback: true
- normalization: `nmt_nfkc`
- 特殊トークンID: `<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3
- seed: 20251031
- shuffle buffer: 20,000 documents

実行途中の進捗と障害原因は標準エラーおよび `pipeline.log` に出ます。失敗時は `tokenizer_stats.json` の `status`, `error_type`, `error` にも記録されます。SentencePiece学習が失敗しても設定を自動変更しません。

## 出力

出力ディレクトリに以下を生成します。

- `tokenizer.model`
- `tokenizer.vocab`
- `train.txt`
- `validation.txt`
- `test.txt`
- `tokenizer_stats.json`
- `pipeline.log`

`tokenizer_stats.json` と標準出力には、TrainのSudachi語数、各splitの文書数・Tokenizerトークン数、Trainの1語あたり平均トークン数、Trainでの語彙使用率・低頻度トークン数、Encode→Decode復元テストを記録します。

`nmt_nfkc` とSentencePieceの空白処理により、Decode結果が生入力と文字単位で一致しないケースは正常に起こり得ます。そのため、生文字列の完全一致に加えて、Decode後に再Encodeしたtoken ID列が安定するかも検査します。

## 主なオプション

```text
--seed INT
--target-train-words INT
--vocab-size INT
--shuffle-buffer-size INT
--min-document-words INT
--low-frequency-threshold INT
```

異なるseedや閾値を使う場合も、実値は必ず `tokenizer_stats.json` に残ります。
