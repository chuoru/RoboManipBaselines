# 区間分割 policy + SpaceMouse + 介入学習: 実機で確認すること

ブランチ `feature/segment-dagger` の各マイルストーンで、コード側は dry run と単体テストで確認済みです。ここに挙げるのは、**実機でしか確かめられないこと**です。確認したら `[x]` にし、測った値や気づいたことを書き足してください。

コマンドは `robo_manip_baselines/` で実行します。手順の詳細は [segment_data_collection.md](./segment_data_collection.md)。

## M1. SpaceMouse とペダル

- [ ] SpaceMouse Compact が開ける（`device_params.device: "SpaceMouse Compact"`）。Linux は udev ルール（segment_data_collection.md §0-1）を入れて挿し直す
- [ ] `python ./misc/CheckSpacemouse.py` で、手を離すと 0 のまま（ドリフトなし）、ボタン 2 つが効く
- [ ] ハンドカメラの映像を見ながら、キャップの向きと映像の動きが一致する（右 = 映像の右、奥 = 映像の上、押し下げ = 前進）。ずれていれば `tool_view_yaw_deg`、1 軸だけ逆なら `axis_sign` を直す（値: ）
- [ ] ひねる・傾けると指先を中心に回る（指先が振れない）。振れるなら `ik_eef_offset_pose` の z（現在 0.25 m = フランジ面 + 150 mm）を直す
- [ ] 速度（`max_lin_vel` 0.05 m/s, `max_ang_vel` 0.5 rad/s）とデッドゾーン（0.08）が操作しやすいか。手を置いただけでドリフトしないか
- [ ] 左ボタンでグリッパが開閉し、右ボタンで精密モード（1/3 速度）が切り替わる
- [ ] `python ./misc/CheckPedalKeys.py` で3つのペダルのキー名を確認し、`common/configs/pedal_default.yaml` に書いた
- [ ] OpenCV ウィンドウにフォーカスがなくてもペダルが効く（Teleop で 中=`n`、右=保存、左=破棄）
- [ ] ペダルを踏みっぱなしにしても1回しか反応しない

## M2. スクリプト動作と区間実行

- [ ] `misc/RecordNamedPose.py` で `home` / `carry` / `fixture_above` / `fixture_grasp` を登録した
- [ ] 各姿勢間の MoveJ が安全（速度 `vel: 10`。周囲に当たらない）
- [ ] `rel_xyz` の退避（IK + MoveJ）がほぼ真上に動く。横にずれる・回る場合は `mode: stream` を試す
- [ ] `--replace_learned_with_teleop` で通しのサイクル（home → grasp(SpaceMouse) → 退避 → carry → place(SpaceMouse) → 退避 → home）が段差なく動く
- [ ] 切り替え時に指令が飛ばない: `logs/command_log_*.csv` の `max_abs_sent_vs_measured_deg` が `max_joint_pos_delta_deg`（6 度）以下
- [ ] スクリプト区間（MoveJ）の前後で ServoJ ストリームの停止・再開が問題なく行われる（1 回あたり約 1.5 s の待ちがある）
- [ ] Esc / Ctrl+C でアームがその場で止まり、記録中の区間が保存される

## M3. デモ収集と初期学習

- [ ] `collect_grasp.yaml`: home から開始 → グリッパを閉じると自動終了 → 保存/破棄 → グリッパが開いて上に退避
- [ ] `collect_place.yaml`: 治具からスクリプトで掴み → carry → 置いて開くと自動終了 → 退避。グリッパ内のカップの持ち方が毎回同じになる
- [ ] teleop の記録周期（`ResampleRmbData.py --check` で確認）が 5 Hz 以上ある（下回ると再サンプリングでフレームが重複する）
- [ ] 背景マスク（0.6 m）がデモ中も効いている（記録映像を確認）
- [ ] grasp・place 各約30本を収集（配置は想定範囲全体に均等）
- [ ] `./train_segment.sh CupPickPlace grasp 0` / `place 0` が完走し、`summary.log` の exit=0
- [ ] **完了条件**: grasp・place それぞれ 20 試行中 16 回以上成功し、通しのサイクルが動く

## M4. 失敗の自動判定・リトライ・展示モード

- [ ] ハンドカメラに指の AprilTag が両方映る（開・空で閉・カップを掴んで閉、の3状態で）。`python ./misc/ViewGripperMarkerDetection.py` 等で確認
- [ ] tool_do グリッパが閉じたとき、カップの幅で指が止まる（全閉まで押し切る構造なら判定方法を変える必要がある）
- [ ] このカメラ（CV2L3600005M）の 1280x800 の内部パラメータを取得し、`envs/configs/gripper_marker/CupPickPlace.yaml` の `camera_matrix` / `dist_coeffs` を置き換えた。`nominal_z` も実測値にした
- [ ] `python ./misc/CalibrateGripperMarkerWidth.py --env RealFairino5DemoEnv-v0 --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml --gripper_marker ./envs/configs/gripper_marker/CupPickPlace.yaml --save_to ./envs/configs/gripper_marker/CupPickPlace.yaml` で開・空閉・カップ閉の幅を測り、しきい値を書き込んだ（値をここに記録: ）
- [ ] 終了時のログ `Finger-tag detection: median __ ms per frame` が 100 ms（カメラ周期）より十分小さい
- [ ] わざと空振りさせる（カップを取り除く）→ `grasp_check: empty` → グリッパを開いて退避 → home → 再試行
- [ ] 持ち上げ後にカップを手で外す → `lift_check` で `dropped` → grasp からやり直し。その grasp のエピソードが `_failure` で保存される
- [ ] タグを手で隠す → grasp 終了時は `undetected`（defer）で進み、lift_check で失敗になる
- [ ] place で印の上に置いて開く → `released_check: released`
- [ ] リトライ上限（2回）を超えると開始姿勢で止まり、画面に「係員を呼んでください」と理由が出る。SUCCESS でグリッパを開いてからサイクルが home から再開する
- [ ] `--mode exhibit`: 左ペダル・中ペダルを踏んでも何も起きない。右ペダル（SUCCESS）は停止後の再開にだけ効く
- [ ] カメラの USB を抜く → 数秒で停止し、理由が表示される
- [ ] 1時間ほど動かしてメモリが増え続けない（結果 YAML の各サイクルの `memory_available_gb`）
- [ ] 展示モードの記録が `dataset/CupPickPlace_<区間>_rollout_iterNNN/` に `_success` / `_failure` 付きで保存される

## M5. 介入（準備モード）

- [ ] `--enable_intervention` で、学習区間の実行中に中ペダルを踏むとその場から SpaceMouse 操作に切り替わり、アームが跳ねない（`command_log` の `max_abs_sent_vs_measured_deg` が 6 度以下）
- [ ] 介入中に grasp を完了（グリッパを閉じる）→ 中ペダル → 判定（grasp_check）→ 次の区間へ
- [ ] 介入を途中で終える → policy が自然に再開する（跳ねない、古い行動チャンクを実行しない）
- [ ] リトライ上限で止まったときの中ペダル（方法 B）で、開始姿勢から操作できる
- [ ] 介入時の SpaceMouse の操作感（5 Hz ループ、速度指定）が十分か。粗い場合は `--control_period 0.1`（+ 学習データを 10 Hz に再サンプリングして学習し直す）を試し、運用する周期をここに書く: 
- [ ] 記録された試行の `intervention` / `intervention_input_active` が操作と合っている（`python ./misc/VisualizeData.py <rmb>` 等）
- [ ] Esc / Ctrl+C の挙動が変わっていない

## M6. 介入データでの反復学習

- [ ] `./train_segment.sh CupPickPlace grasp 1` が完走する（warm start、統計の再利用、`Rollout data filter 'd8'` のサンプル数がログに出る）
- [ ] 介入由来のサンプル数が十分か（少なければ介入を長めに / 回数を増やす）
- [ ] 統計範囲外の警告が出ていない（出たら、その次元のデータを確認）
- [ ] iter000 と iter001 を同じ配置条件で比べ、`misc/SummarizeIterations.py` の「介入回数 / 試行」が減る、または成功率が上がる（値を記録: ）
- [ ] 失敗した試行の初期画像一覧（`logs/iteration_summary/failures_*.png`）で苦手な配置を確認し、必要ならその配置のデモを追加した

## M7. 展示モードでの連続稼働試験（最終確認）

- [ ] 展示会と同じ設置（照明・背景・テーブル・カップ・印）にした
- [ ] `--mode exhibit` で連続 30 サイクル以上（目安 1 時間）。印は毎サイクル違う場所（範囲の端も含める）
- [ ] 明るさを変えた条件でも数サイクル（背景マスクとタグ検出が崩れないこと）
- [ ] 指標を集計して `doc/evaluation_results.md` に追記した（`python ./misc/SummarizeSegmentResults.py <結果YAML>`）
  - 区間ごとの成功率（リトライなしで成功 / リトライで成功 / 停止）
  - 係員を呼ぶ停止の回数と理由
  - 1 サイクルの所要時間
  - メモリ使用量の推移
- [ ] **合格基準**: 係員を呼ぶ停止が 1 回以下。満たさなければ失敗理由の内訳を見て M3（デモ追加）か M6（反復学習）に戻る
