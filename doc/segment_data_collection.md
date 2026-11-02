# 区間ごとのデモ収集と学習（SpaceMouse + フットペダル）

展示会タスク「カップを掴んで、印（コースター）の上に置く」を、学習区間 `grasp` / `place` とスクリプト区間に分けて動かすための、デモ収集から学習までの手順です。
全体の流れと設計判断は、計画書（plan_spacemouse_dagger.md）と `robo_manip_baselines/common/base/SegmentRunnerBase.py` の冒頭コメントを参照してください。

コマンドはすべて `robo_manip_baselines/` で実行します（Windows は Git Bash）。

```
[S] home へ MoveJ
[L] grasp : カップに近づいて掴む       ← SpaceMouse でデモを集める（約30本）
[S] 真上に退避 → carry へ MoveJ
[L] place : 印の上に置いて離す         ← SpaceMouse でデモを集める（約30本、方法 B）
[S] 真上に退避 → home へ
```

## 0. 準備（最初に1回）

### 0-1. SpaceMouse Compact
- 設定ファイル: `teleop/configs/SpaceMouseRealFairino5.yaml`
  - 左ボタン: グリッパの開閉（押すたびに切り替え）
  - 右ボタン: 精密モード（速度 1/3）の切り替え。位置合わせの最後に使う
  - 速度はループ周期に依存しない速度指定（`max_lin_vel` 0.05 m/s、`max_ang_vel` 0.5 rad/s）
- Windows では `pyspacemouse` が `hidapi.dll` を必要とします（PATH か python.exe と同じフォルダに置く）。
- 3Dconnexion 公式ドライバ（3DxWare）が常駐していると、デバイスを掴まれて開けない、または SpaceMouse の操作が OS のマウス操作としても入る場合があります。開けないときは 3DxWare を終了するか、python.exe に対して無効にしてください。**実機でどちらが必要だったかを、確認後この節に追記すること。**

### 0-2. フットペダル
1. `python ./misc/CheckPedalKeys.py` を起動し、左・中・右のペダルを順に踏んで、表示されるキー名を控えます。
2. `common/configs/pedal_default.yaml` の `fail`（左）/ `intervene`（中）/ `success`（右）の先頭に、そのキー名を書きます。
3. 数字キー 1 / 2 / 3 は予備です。ペダルはシステム全体で拾うので、収集中は他のウィンドウで文字を打たないでください。

| ペダル | Teleop（収集）での意味 | RolloutSegments での意味 |
|---|---|---|
| 左 `fail` | エピソード破棄（= `f`） | 区間失敗 → リトライ（準備モードのみ） |
| 中 `intervene` | 次へ / 開始 / 終了（= `n`） | 介入の開始・終了（準備モードのみ） |
| 右 `success` | 保存（= `s`） | 区間成功 / 開始 / 停止後の再開 |

Esc（OpenCV ウィンドウ）で終了します。

### 0-3. 名前付き姿勢
`envs/configs/poses/CupPickPlace.yaml` の `home` / `carry` / `fixture_above` / `fixture_grasp` を実機で登録します。未登録（`null`）の姿勢があると、収集も実行も起動時に止まります。

```console
$ python ./misc/RecordNamedPose.py --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml \
    --poses_file ./envs/configs/poses/CupPickPlace.yaml \
    --names home carry fixture_above fixture_grasp --input_device spacemouse
```
SpaceMouse でアームを動かし、SUCCESS ペダル（または Enter / `s`）でその時点の関節角を記録します（`k` でスキップ）。

- `home`: grasp の開始姿勢。カップが置かれうる範囲全体がハンドカメラに映る高さ
- `carry`: place の開始姿勢。カップを持ったまま、印が置かれうる範囲全体が映る高さ
- `fixture_above` / `fixture_grasp`: カップ置き場（治具）の真上と、指がカップを囲む高さ

### 0-4. env 設定
収集にも実行にも `envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml` を使います。カメラ設定・背景マスク・TCP は `RealFairino5DemoEnv_UmiPolicy_BgMask_5Hz.yaml` と同じです。**この設定で集めたデータで学習した policy は、この設定で動かしてください**（グリッパの状態の表し方も含めて、学習時と同じ観測になるようにするため）。

## 1. 先に動作確認（dry run、ハードなし）

```console
$ python ./bin/RolloutSegments.py RealFairino5Demo \
    --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse_DryRun.yaml \
    --task ./envs/configs/segments/CupPickPlace.yaml \
    --poses ./envs/configs/poses/DryRunExample.yaml \
    --replace_learned_with_teleop --input_device keyboard --num_cycles 1
```
`n` で開始し、学習区間ではキーボード（`z` で閉じる、`x` で開く）か SUCCESS で次に進みます。`logs/command_log_dryrun_*.csv` の `max_abs_sent_vs_measured_deg` 列が `max_joint_pos_delta_deg`（6 度）以下であれば、切り替え時に指令は飛んでいません。

## 2. grasp のデモ収集

```console
$ python ./bin/Teleop.py RealFairino5Demo \
    --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml \
    --input_device spacemouse --input_device_config ./teleop/configs/SpaceMouseRealFairino5.yaml \
    --pre_segments ./envs/configs/segments/collect_grasp.yaml \
    --pedal_config ./common/configs/pedal_default.yaml \
    --demo_name CupPickPlace_grasp_SM_iter000
```

1回の流れ:
1. 中ペダル（`n`）→ アームが初期姿勢 → `home` に移動して待機（Standby）
2. 中ペダル（`n`）で SpaceMouse の操作と記録が同時に始まる（Teleop）
3. SpaceMouse でカップに近づき、左ボタンでグリッパを閉じる → 閉じて 5 ループ経つと自動で記録終了（中ペダルで手動終了も可）
4. 右ペダル（保存）または左ペダル（破棄）
5. 自動でグリッパが開き、8 cm 上に退避 → 次のエピソードへ

保存先: `dataset/CupPickPlace_grasp_SM_iter000_<日時>/`（セッションごとに1フォルダ。学習時にまとめて使います）

## 3. place のデモ収集（方法 B: 固定の置き場から掴む）

```console
$ python ./bin/Teleop.py RealFairino5Demo \
    --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml \
    --input_device spacemouse --input_device_config ./teleop/configs/SpaceMouseRealFairino5.yaml \
    --pre_segments ./envs/configs/segments/collect_place.yaml \
    --pedal_config ./common/configs/pedal_default.yaml \
    --demo_name CupPickPlace_place_SM_iter000
```

1回の流れ:
1. カップを治具に置き、印をテーブルの別の場所に置く
2. 中ペダル → `home` → 治具の上 → 下降 → 閉じる → 10 cm 上 → `carry`（スクリプト）
3. 中ペダルで操作と記録を開始。印の上にカップを置き、左ボタンでグリッパを開く → 開いて 5 ループで自動終了
4. 右ペダル（保存）または左ペダル（破棄）→ 8 cm 上に退避

治具の位置を固定するのは、グリッパ内でのカップの持ち方を毎回揃えるためです。

## 4. 良いデモの条件（両区間共通）

- **配置を均等にばらつかせる**: カップの位置・向き（grasp）、印の位置（place）を、想定範囲（例: ±10 cm、±30°）全体に均等に散らします。範囲の端も必ず含めます。同じ場所に偏らせない。
- **迷った動きは破棄**: 行き過ぎて戻った、掴み直した、止まって考えた、などを含むエピソードは左ペダルで破棄します。policy は迷いも真似します。
- **ゆっくり一定の速度で**: 急な加減速を避け、毎回同じリズムで動かします。最後の位置合わせは精密モード（右ボタン）で。
- **開始と終了の条件を守る**: 開始は必ずスクリプトが連れてきた姿勢から（途中から始めない）。終了はグリッパの開閉で（自動終了に任せる）。
- 目安: 各区間約30本。

## 5. 学習

```console
$ ./train_segment.sh CupPickPlace grasp 0
$ ./train_segment.sh CupPickPlace place 0
```

- `dataset/CupPickPlace_<区間>_SM_iter000*` を 5 Hz に再サンプリングし（`dataset/resampled_5hz/`）、`--skip 1` で Diffusion Policy を学習します。チェックポイントは `checkpoint/DiffusionPolicy/CupPickPlace_<区間>_iter000/`。
- 行動は `command_eef_pose`（観測時刻の姿勢に対する相対軌跡、`--relative_eef_action`）+ `command_gripper_joint_pos`、状態は `measured_gripper_joint_pos` だけです。
- 実行時間・終了コードは `summary.log` に追記されます。
- エポック数などは環境変数で変えられます（`NUM_EPOCHS=300 ./train_segment.sh ...`）。

再サンプリングだけ確認したいとき:
```console
$ python ./misc/ResampleRmbData.py --in_dir ./dataset/CupPickPlace_grasp_SM_iter000_20261105_100000 --check
```

## 6. 通しで実行

```console
$ python ./bin/RolloutSegments.py RealFairino5Demo \
    --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml \
    --task ./envs/configs/segments/CupPickPlace.yaml
```
右ペダル（または `n`）で開始します。最初は `--num_cycles 1` で1サイクルずつ確認してください。
policy ができる前に通しを試すには `--replace_learned_with_teleop`（学習区間を SpaceMouse 操作に置き換える）を付けます。

M3 の完了条件: grasp・place それぞれ 20 試行中 16 回以上成功し、通しのサイクルが動くこと。

## 7. 介入（準備モード、HG-DAgger 用のデータ集め）

```console
$ python ./bin/RolloutSegments.py RealFairino5Demo \
    --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml \
    --task ./envs/configs/segments/CupPickPlace.yaml \
    --enable_intervention --save_rollout --iteration 1
```

- **方法 A（基本）**: 学習区間の実行中、policy がずれ始めたら**早めに**中ペダル（INTERVENE、または `i`）。その場から SpaceMouse で操作できます（アームは退避しません）。画面は赤枠と "INTERVENTION"。
  - もう一度中ペダルで介入終了。その時点でグリッパが区間の終了状態（grasp なら閉、place なら開）なら区間成功として次へ進み、そうでなければ policy に戻ります。
  - 介入中は policy の推論もタイムアウトも止まり、終了条件の自動判定もしません（終了は必ず中ペダルで）。
  - 同じペダルを 0.3 秒以内に2回踏んでも1回として扱います。
- **方法 B（補助）**: リトライ上限で止まったとき、中ペダルを踏むとその区間を開始姿勢から SpaceMouse で行えます（右ペダルはこれまでどおりサイクルを最初から再開）。
- 左ペダル（FAIL）で区間を失敗にしてリトライ、右ペダル（SUCCESS）で区間を成功として終了できます。
- 記録: 学習区間の1回の試行ごとに `dataset/CupPickPlace_<区間>_rollout_iter001/` へ保存され、各ステップに `intervention`（人が操作中なら 1）と `intervention_input_active`（実際に SpaceMouse を動かしていれば 1）、属性に `num_interventions` / `intervention_time` / `intervention_methods` / `segment_result` などが入ります。
- `--iteration N` は「このデータを使う学習の反復番号」です（iter000 の policy で集めたデータは `--iteration 1`）。
- 展示モード（`--mode exhibit`）では介入できません。

## 8. 介入データでの反復学習（iter001 以降）

1. iter000 の policy で `--enable_intervention --save_rollout --iteration 1` を回し、介入（人の修正）が約5本たまったら1反復回します。
2. 学習:
   ```console
   $ ./train_segment.sh CupPickPlace grasp 1
   ```
   - データ: 初期デモ `SM_iter000*` + 追加デモ `SM_iter001*`（あれば）+ `rollout_iter001`（いずれも 5 Hz に再サンプリング）
   - `--rollout_data_filter d8`: 介入付きの試行からは、**最終的に成功した試行の、人が実際に操作していた区間だけ**を重み 2.5 で使います。介入直前 1 秒、介入直後の反応遅れ（SpaceMouse を触る前）、切り替え前後 2 ステップは捨てます。失敗した試行・policy だけで動いた区間は使いません。初期デモは全体を重み 1 で使います。
   - 1サンプルは行動 16 ステップ（5 Hz で約 3 秒）がすべて使える区間に収まる必要があります。**介入は区間の最後まで続けるか、少なくとも 4 秒程度は操作してから戻す**と学習に使えるサンプルになります（短すぎる介入は使われません。学習ログの `Rollout data filter 'd8': ... samples` で数を確認）。
   - 前の反復のチェックポイントから warm start し、前の反復の正規化統計を使います（範囲外のデータは警告）。エポック数は 50 + early stopping（`NUM_EPOCHS` / `PATIENCE` で変更）。
   - 検証データは初期デモから固定（`dataset/resampled_5hz/CupPickPlace_grasp_val.txt`、初回に作成）。反復の間で val loss を比べられます。
   - 暴走防止のステップ上限（`max_episode_len`）は初期デモだけから計算されます。
3. 新しいチェックポイント `checkpoint/DiffusionPolicy/CupPickPlace_grasp_iter001/` を区間 YAML（または `--checkpoint grasp=...`）に設定し、`--iteration 2` でデータを集める……を繰り返します。
4. 効果の確認:
   ```console
   $ python ./misc/SummarizeIterations.py --task CupPickPlace
   ```
   区間・反復ごとの成功率、介入回数 / 試行、平均介入時間、リトライ率、失敗理由と、失敗した試行の最初のカメラ画像の一覧（`logs/iteration_summary/`）が出ます。苦手な位置・向きが見えたら、そこのデモを `SM_iter00N` として追加するのも有効です（介入が間に合わない場合の代替）。

M6 の完了条件: iter000 → iter001 で、同じ条件の試行における「介入回数 / 試行」が減る、または成功率が上がること。
