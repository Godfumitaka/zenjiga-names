# zenjiga-names：夜間の探索（2026-11-18〜19）

## 朝に読むもの
- `REPORT_v3.md`：本命（v3、本来の模型の最初の段）の報告
- `REPORT.md`：偵察（v2、名前の層）の報告
- `DECISIONS.md`：仕様に書かれていなかった点と、途中で決めたこと
- `PREDICTIONS_v3.md`・`PREDICTIONS.md`：走らせる前に固定した予想（`main` の最初のコミット `c085fca` から変えていない）

## ブランチと PR（積み重ね。各 PR の基点は一つ前の段階）
`main` ← #1 `v3-stage1` ← #2 `v3-stage2` ← #3 `v3-stage3` ← #4 `v2-stageA` ← #5 `v2-stageB` ← #6 `v3-stage4` ← #7 `reports` ← #8 `v3-stage4-more`

最後のブランチ `v3-stage4-more` に全部が入っている（報告もここが最新）。

## 走らせ方（Python 3.12、numpy・scipy・matplotlib）
```
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python numpy scipy matplotlib
.venv/bin/python -m v3.stage1      # 段 1（smoke2 の再現、R1・R2）約 20 秒
.venv/bin/python -m v3.stage2      # 段 2（命名ゲームの N^1.5）約 1 分
.venv/bin/python -m v3.stage3      # 段 3（五条件、種 30）約 1 分
.venv/bin/python -m v3.stage4      # 段 4（掃引）約 15 分
.venv/bin/python -m v3.analyze4 && .venv/bin/python -m v3.report_tables
.venv/bin/python -m v3.stage4_more # 段 4 の追加（小さい p の種を 130 に）約 6 分
.venv/bin/python -m v2.stageA      # 段階 A
.venv/bin/python -m v2.stageB && .venv/bin/python -m v2.stageB long_sobs02   # 段階 B 約 8 分
.venv/bin/python -m v2.analyzeB && .venv/bin/python -m v2.report_tables
```
結果は `results/v3/`・`results/v2/` に、設定（JSON）と乱数の種と一緒に置いてある。
