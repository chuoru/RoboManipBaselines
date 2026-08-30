"""Build the beginner-friendly PDF write-up of the Rollout/RealFairino5 bugfixes
found while getting a DiffusionPolicy checkpoint to run cleanly on the real FR5
arm (bin/Rollout.py and the RealEnvBase/ArmManager/MotionManager stack it
drives).

Unlike make_umi_replay_fix_pdf.py (reportlab, built paragraph by paragraph),
this one renders a self-contained HTML document with headless Chrome/Chromium
-- simpler to keep in sync with the write-up's tables/callout boxes, and
Chrome's own font stack (Noto Sans/Serif CJK) handles the Japanese text
without the .ttc-to-TrueType conversion reportlab needed.
"""

import os
import shutil
import subprocess

OUT_PDF = os.path.join(os.path.dirname(__file__), "rollout_bugfix_explained.pdf")

HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Rollout実機不具合 修正報告書</title>
<style>
  @page { size: A4; margin: 20mm 18mm; }
  * { box-sizing: border-box; }
  body {
    font-family: "Noto Sans CJK JP", "Noto Sans JP", "Hiragino Kaku Gothic ProN", sans-serif;
    color: #1a1a1a;
    line-height: 1.75;
    font-size: 10.5pt;
  }
  h1 { font-size: 19pt; margin: 0 0 4mm 0; border-bottom: 3px solid #1f4e79; padding-bottom: 3mm; }
  .subtitle { color: #555; font-size: 10pt; margin-bottom: 10mm; }
  h2 {
    font-size: 13.5pt; color: #1f4e79; margin-top: 11mm; margin-bottom: 3mm;
    padding: 2mm 3mm; background: #eaf1f8; border-left: 5px solid #1f4e79;
    page-break-after: avoid;
  }
  h3 { font-size: 11.5pt; color: #1f4e79; margin-top: 6mm; margin-bottom: 2mm; page-break-after: avoid; }
  h4 { font-size: 10.5pt; color: #333; margin: 4mm 0 1.5mm 0; page-break-after: avoid; }
  p { margin: 2mm 0; }
  .bug-block { page-break-inside: avoid; margin-bottom: 4mm; }
  .toc { background: #f7f7f7; border: 1px solid #ddd; border-radius: 3mm; padding: 5mm 7mm; margin: 6mm 0; }
  .toc h2 { background: none; border: none; padding: 0; margin: 0 0 2mm 0; }
  .toc ol { margin: 0; padding-left: 5mm; columns: 1; }
  .toc li { margin: 1mm 0; }
  .toc a { color: #1f4e79; text-decoration: none; }
  table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 9.5pt; }
  th, td { border: 1px solid #ccc; padding: 1.8mm 3mm; text-align: left; vertical-align: top; }
  th { background: #1f4e79; color: white; font-weight: 600; }
  tr:nth-child(even) td { background: #f5f8fb; }
  code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    background: #eef0f2; padding: 0.3mm 1.2mm; border-radius: 1mm; font-size: 9pt;
  }
  pre {
    background: #1e1e2e; color: #dcdce4; padding: 3mm 4mm; border-radius: 2mm;
    overflow-x: auto; font-size: 8.7pt; line-height: 1.5; margin: 2.5mm 0;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    page-break-inside: avoid;
  }
  pre code { background: none; padding: 0; color: inherit; }
  .label {
    display: inline-block; font-size: 8.3pt; font-weight: 700; color: white;
    padding: 0.6mm 2.5mm; border-radius: 1mm; margin-right: 1.5mm; letter-spacing: 0.3px;
  }
  .sev-critical { background: #c0392b; }
  .sev-high { background: #d35400; }
  .sev-mid { background: #b8860b; }
  .sev-info { background: #2e7d32; }
  .file-badge {
    display: inline-block; font-size: 8.5pt; color: #1f4e79; background: #eaf1f8;
    border: 1px solid #cfe0ef; padding: 0.5mm 2mm; border-radius: 1mm; font-family: monospace;
    margin: 0.5mm 1mm 0.5mm 0;
  }
  .callout {
    border-radius: 2mm; padding: 3mm 4.5mm; margin: 3mm 0; font-size: 9.7pt;
  }
  .callout.symptom { background: #fdf2f2; border-left: 4px solid #c0392b; }
  .callout.cause { background: #fdf6e8; border-left: 4px solid #b8860b; }
  .callout.fix { background: #eef8f0; border-left: 4px solid #2e7d32; }
  .callout.note { background: #eef4fb; border-left: 4px solid #1f4e79; }
  .callout .ctitle { font-weight: 700; margin-bottom: 1.5mm; display: block; }
  .metric-row { display: flex; gap: 4mm; flex-wrap: wrap; margin: 2mm 0; }
  .metric {
    background: #f5f8fb; border: 1px solid #dbe6f0; border-radius: 2mm;
    padding: 2mm 4mm; font-size: 9pt; min-width: 40mm;
  }
  .metric .k { color: #666; display: block; font-size: 8pt; }
  .metric .v { font-weight: 700; color: #1f4e79; font-size: 10.5pt; }
  .arrow { color: #c0392b; font-weight: 700; margin: 0 1mm; }
  .divider { border: none; border-top: 1px dashed #ccc; margin: 6mm 0; }
  .summary-table td:first-child { white-space: nowrap; font-weight: 700; text-align: center; }
  .footer-note { font-size: 8.5pt; color: #777; margin-top: 3mm; }
  ul, ol { margin: 2mm 0; padding-left: 6mm; }
  li { margin: 1mm 0; }
  .newfile { background: #f0f7f0; border: 1px solid #cfe8cf; border-radius: 2mm; padding: 3mm 4.5mm; margin: 3mm 0; }
</style>
</head>
<body>

<h1>Rollout実機不具合 修正報告書</h1>
<div class="subtitle">
  対象: DiffusionPolicy を用いた RealFairino5Demo（FR5実機）でのポリシー実行（<code>bin/Rollout.py</code>）まわり<br>
  期間中に発見・修正した不具合を、発見の経緯・原因・対処の順にまとめる
</div>

<div class="toc">
<h2>目次</h2>
<ol>
  <li><a href="#overview">全体概要</a></li>
  <li><a href="#bug1">① 状態(state)入力バグ ― 方策に絶対座標が渡っていた</a></li>
  <li><a href="#bug2">② IK内部状態の巻き込み(ワインドアップ)</a></li>
  <li><a href="#bug3">③ 速度クランプの経過時間が固定値だった</a></li>
  <li><a href="#bug4">④ ループのペーシングが正のフィードバックで暴走</a></li>
  <li><a href="#bug5">⑤ カメラのブロッキング待ちで制御ループが10Hzに律速</a></li>
  <li><a href="#bug6">⑥ 速度クランプの上限緩和が招いた起動時の急加速</a></li>
  <li><a href="#bug7">⑦ 速度クランプでの減速が閉ループを破綻させる（振動）</a></li>
  <li><a href="#bug8">⑧ タスク完了後に無限に同じ動作を繰り返す</a></li>
  <li><a href="#bug9">⑨ 絶対上限値が厳しすぎてタスクが完了不能に</a></li>
  <li><a href="#bug10">⑩ 起動直後の急加速（バン！）― 観測の同期漏れ</a></li>
  <li><a href="#bug11">⑪ グリッパ状態読み取り警告のスパム</a></li>
  <li><a href="#tested">検証したが効果がなかった対策</a></li>
  <li><a href="#tools">新規に作成した診断ツール</a></li>
  <li><a href="#final">最終的な推奨設定</a></li>
</ol>
</div>

<h2 id="overview">全体概要</h2>
<p>
DiffusionPolicyで学習したチェックポイントを実機のFR5アームで動かす過程で、次々と不具合が発覚した。
症状は「推論ができていないように見える」「動きがガクガクする」「起動時に急加速する」「タスクを何度も繰り返す」など多岐にわたったが、
調査の結果、原因は<strong>モデルの学習内容そのものではなく、実機を動かすための周辺コード（状態の受け渡し・安全機構・タイミング制御）に潜む複数の独立した不具合</strong>であったことが判明した。
</p>
<p>
本書では、発見された不具合を時系列に沿って解説する。多くの不具合は連鎖しており、ひとつを直すと別の不具合が表面化するという展開になったため、
「なぜその修正が必要だったか」「その修正が次にどんな問題を引き起こしたか」を合わせて記載する。
</p>

<table class="summary-table">
<tr><th>#</th><th>不具合</th><th>深刻度</th><th>影響</th></tr>
<tr><td>①</td><td>状態入力に絶対座標を使用</td><td><span class="label sev-critical">重大</span></td><td>方策が学習時と全く異なる入力を受け取り、出力が支離滅裂に</td></tr>
<tr><td>②</td><td>IK内部状態のワインドアップ</td><td><span class="label sev-critical">重大</span></td><td>速度制限下で指令が実機から際限なく乖離</td></tr>
<tr><td>③</td><td>速度クランプの経過時間が固定</td><td><span class="label sev-high">高</span></td><td>推論の遅延ティックで指令が飢餓状態に</td></tr>
<tr><td>④</td><td>ペーシングの正のフィードバック</td><td><span class="label sev-high">高</span></td><td>制御ループが2Hzまで低下</td></tr>
<tr><td>⑤</td><td>カメラ待ちで10Hz律速</td><td><span class="label sev-high">高</span></td><td>1指令あたりの移動量が過大に</td></tr>
<tr><td>⑥</td><td>クランプ上限緩和で急加速</td><td><span class="label sev-critical">重大</span></td><td>実機で最大約10度の単発ジャンプ</td></tr>
<tr><td>⑦</td><td>速度クランプでの減速が振動を誘発</td><td><span class="label sev-critical">重大</span></td><td>クランプを絞るほど閉ループが破綻し激しく振動</td></tr>
<tr><td>⑧</td><td>タスク完了後に無限リピート</td><td><span class="label sev-mid">中</span></td><td>実機が報酬0のため自動停止せず永久に反復</td></tr>
<tr><td>⑨</td><td>絶対上限値0.5度でタスク完了不能</td><td><span class="label sev-high">高</span></td><td>安全策のつもりが方策を分布外に押し出し停滞</td></tr>
<tr><td>⑩</td><td>起動直後の急加速（バン！）</td><td><span class="label sev-critical">重大</span></td><td>MoveJ直後に古い観測を使い14度の単発指令</td></tr>
<tr><td>⑪</td><td>グリッパ警告のスパム</td><td><span class="label sev-info">軽微</span></td><td>起動前のコンソールが警告で埋まる（実害なし）</td></tr>
</table>

<hr class="divider">

<h2 id="bug1">① 状態(state)入力バグ ― 方策に絶対座標が渡っていた</h2>
<div class="bug-block">
<span class="label sev-critical">重大</span>
<span class="file-badge">policy/diffusion_policy/RolloutDiffusionPolicy.py</span>
<span class="file-badge">他6ファイル</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
学習時と同じカメラ画像を入力しても、実機での動きは「対象物の位置に関わらず同じ前後動作・グリッパ開閉を反復する」だけで、
コップに近づく動作にすら至らなかった。オフライン検証（記録済みデータでの再現テスト）ではモデルは正しく軌道を再現できていたため、モデル自体は正常と判断していた。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
学習データはUMI（ハンドヘルド収録リグ）由来で、<strong>ロールアウト開始時の手先姿勢を原点とする「エピソード相対座標」</strong>で記録されている。
実機側では <code>RolloutBase.get_measured_data_for_policy()</code> がこの相対変換（絶対座標 → 相対座標）を正しく行うようになっていたが、
DiffusionPolicyをはじめとする7つの方策クラスの独自の状態バッファ構築処理（<code>update_state_buf()</code> など）が、
この変換メソッドを経由せず <code>self.motion_manager.get_data(state_key, self.obs)</code> を直接呼び出しており、変換前の<strong>絶対座標</strong>をそのままモデルに渡していた。
</div>

<pre><code># 修正前（バグ）: 絶対座標をそのまま使用
convert_data_to_policy(
    self.motion_manager.get_data(state_key, self.obs), state_key
)

# 修正後: エピソード相対変換を経由
convert_data_to_policy(
    self.get_measured_data_for_policy(state_key), state_key
)</code></pre>

<p>dry_runで実測した数値がこの深刻さを裏付けている。</p>
<div class="metric-row">
  <div class="metric"><span class="k">修正前（絶対座標）の並進</span><span class="v">(0.012, -0.267, 0.355) m</span></div>
  <div class="metric"><span class="k">修正後（相対座標）の並進</span><span class="v">(0, 0, 0) m</span></div>
</div>
<p>
回転についても、修正前はクォータニオン (0.010, -0.003, -0.702, 0.713) と<strong>90度以上傾いた値</strong>が渡っていたのに対し、
学習データの状態統計は回転がほぼ単位クォータニオン付近（<code>qw≈0.88〜1.0</code>）でしか変化しない。
つまりモデルは学習時に一度も見たことのない、分布から大きく外れた入力を毎回与えられていたことになる。
</p>

<div class="callout fix">
<span class="ctitle">修正</span>
影響を受けていた <code>RolloutDiffusionPolicy</code>／<code>RolloutDiffusionPolicy3d</code>／<code>RolloutFlowPolicy</code>／<code>RolloutManiFlowPolicy</code>／
<code>RolloutMlp</code>／<code>RolloutPi0</code>／<code>RolloutGr00t</code> の7クラス全てで、状態読み取りを <code>get_measured_data_for_policy()</code> 経由に統一した。
なお <code>RolloutAct</code>／<code>RolloutMtAct</code>／<code>RolloutSarnn</code> は独自の状態バッファを持たず、元々正しい経路（<code>RolloutBase.get_state()</code>）を使っていたため影響を受けていなかった。
</div>
</div>

<h2 id="bug2">② IK内部状態の巻き込み(ワインドアップ)</h2>
<div class="bug-block">
<span class="label sev-critical">重大</span>
<span class="file-badge">common/body/ArmManager.py</span>
<span class="file-badge">common/manager/MotionManager.py</span>
<span class="file-badge">common/base/RolloutBase.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
速度制限（<code>joint_vel_limit_scale</code>）を緩めるほど、揺れが<strong>悪化</strong>するという逆説的な現象が発生した。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
逆運動学（IK）を担う <code>ArmManager.inverse_kinematics()</code> は、毎ティック自分自身の内部状態 <code>self.arm_joint_pos</code> を起点に
1歩ずつ目標へ近づく計算を行い、その結果でまた <code>self.arm_joint_pos</code> を更新する。
しかし<strong>この内部状態は実機の実測関節角と一度も同期されていなかった</strong>。IKは「指令通り完璧に動けた」前提で計算を進めるが、
実際には安全機構の速度クランプで動きが制限されるため、内部モデルと実機の差は際限なく拡大していく（積分のワインドアップ）。
</p>
<p>
差が拡大すると、方策は「実機が大きく遅れた状態」を観測して<strong>全く別の軌道を再計画</strong>し、目標が逆方向へ飛ぶ。
速度制限を緩めると、実機がこの「暴走した目標」により強く反応できてしまうため、症状が悪化していた。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
<code>ArmManager.sync_to_measured()</code> を新設し、実測関節角へ内部状態を強制的に合わせる仕組みを追加。
<code>MotionManager.sync_arm_to_measured(obs)</code> が全アームに対してこれを呼び出し、<code>RolloutBase.run()</code> のメインループが
毎ティック・指令計算の直前にこれを実行するようにした。以後、IKは常に「実機が本当にいる場所」を起点に計算される。
</div>
</div>

<h2 id="bug3">③ 速度クランプの経過時間が固定値だった</h2>
<div class="bug-block">
<span class="label sev-high">高</span>
<span class="file-badge">envs/real/RealEnvBase.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
②の修正後もなお、関節角指令が跳ねる現象（1ティックで最大43度）が残っていた。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
安全機構（速度クランプ、<code>overwrite_command_for_safety</code>）は「今回の指令でどれだけ動いてよいか」を
<code>速度上限 × duration</code> で決めるが、この <code>duration</code> には<strong>常に固定値 <code>self.dt = 0.02秒</code></strong> が渡されていた。
実際の制御周期は拡散モデルの推論（実測最大281ms）やカメラ読み出し・XML-RPC通信の影響で26ms〜472msと大きくばらついており、
実時間で0.47秒経過したティックでも「0.02秒分（=1/23）しか動いてはいけない」と制限されていた。
その結果、方策の目標に対する遅れが蓄積し、次に短い周期のティックが来た瞬間に一気に放出される「詰まる→跳ねる」を繰り返していた。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
<code>RealEnvBase.step()</code> で実際の経過時間を実測し（<code>_clamp_duration</code>）、速度クランプの計算にはその実測値を使うよう変更。
ServoJ自体の <code>cmdT</code>（送信間隔）は元から実測経過時間を使う設計だったため、安全クランプ側もそれに合わせた形になる。
</div>
</div>

<h2 id="bug4">④ ループのペーシングが正のフィードバックで暴走</h2>
<div class="bug-block">
<span class="label sev-high">高</span>
<span class="file-badge">envs/real/RealEnvBase.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
③の修正直後、動きは滑らかになったが、<strong>ループ全体が極端に遅くなった</strong>（2.0Hz、実測の59%のティックでアームが静止）。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
③の修正で使った <code>duration</code> という同一の変数が、実は速度クランプの計算だけでなく、
<strong>ループのペーシング（<code>wait=True</code> のときの <code>sleep()</code> 時間の目標値）にも流用されていた</strong>。
実測した経過時間をそのまま次のペーシング目標として渡してしまうと、「前回の周期と同じだけ待つ」という正のフィードバックが生まれ、
周期はどんどん伸びて上限の0.5秒に張り付いてしまう。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
2つの用途を分離。<code>_set_action</code> に渡す <code>duration</code>（ペーシング目標）は公称値 <code>dt=0.02</code> に戻し、
速度クランプの計算だけが専用の <code>_clamp_duration</code>（実測値）を参照するようにした。
</div>
</div>

<h2 id="bug5">⑤ カメラのブロッキング待ちで制御ループが10Hzに律速</h2>
<div class="bug-block">
<span class="label sev-high">高</span>
<span class="file-badge">envs/real/RealEnvBase.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
④の修正後もなお、多少ましになった程度でカクつきが残っていた。制御周期の中央値がぴったり<strong>0.100秒</strong>という、
外部クロックに同期したかのような不自然な値になっていた。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
<code>get_pointcloud_camera_data()</code> が、毎ステップ<strong>新しいカメラフレームが届くまでブロック</strong>して待っていた。
Orbbecカメラの実際のフレームレートは10FPS（0.1秒間隔）であり、制御ループ全体がこれに完全に律速されていた。
指令レートが10Hzしかないため、1回の指令でカバーしなければならない移動量が学習時（50Hz想定）の約5倍に膨らみ、これが飛びとして現れていた。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
カメラ待ちをノンブロッキング化。キューを最新フレームまでドレインしてから使用し、新フレームが無ければ直近のフレームを再利用して即座に返すようにした。
実際にカメラが無応答になった場合（<code>RECONNECT_TIMEOUT_SEC</code>=2秒超）のみ、従来通りフォールト扱いで自動リコネクトする。
これにより制御ループはカメラのFPSから解放され、dry_run環境で約37〜48Hzまで高速化した。
</div>
</div>

<h2 id="bug6">⑥ 速度クランプの上限緩和が招いた起動時の急加速</h2>
<div class="bug-block">
<span class="label sev-critical">重大</span>
<span class="file-badge">envs/real/RealEnvBase.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
⑤でループが高速化した直後の実機テストで、<strong>「バン！」という急加速</strong>が発生した。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
③で導入した速度クランプの上限 <code>STEP_DURATION_MAX_SEC = 0.5秒</code> が緩すぎた。
ループが10Hzだった頃は目立たなかったが、⑤の修正で37Hzまで高速化したことで、
「推論スパイクで0.31秒かかったティックの直後に、0.5秒分の移動量（速度30度/秒として約9.4度）を一度に許可してしまう」という危険な組み合わせが発生した。
実測ログでも「実測との乖離が9.77度」というスパイクが確認された。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
<code>STEP_DURATION_MAX_SEC</code> を0.5秒から<strong>0.1秒</strong>に短縮し、単発の許可量を最悪でも3度程度に抑えた。
あわせて、指令と実測の乖離が<strong>30度を超えたら送信を停止して例外を送出するハードアボート</strong>を新設し、
何らかの理由で乖離が拡大した場合にアームがその場で保持されるようにした（従来は15度で警告を出すのみで、動作は継続していた）。
</div>
</div>

<h2 id="bug7">⑦ 速度クランプでの減速が閉ループを破綻させる（振動）</h2>
<div class="bug-block">
<span class="label sev-critical">重大</span>
<span class="file-badge">envs/real/RealEnvBase.py</span>
<span class="file-badge">envs/operation/OperationRealFairino5Demo.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
「動きを遅くしたい」という要望に対し <code>joint_vel_limit_scale</code>（速度クランプの倍率）を下げて対応したところ、
<strong>高速でガクガク振動する</strong>という、意図と正反対の結果になった。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
速度クランプは「安全網」であって「速度調整の手段」ではなかった。実測波形が動かぬ証拠となった。
</p>
<pre><code>方策の目標（手首関節）: 63.7度 → 55.6度 → 48.1度 → 43.5度（急降下）
実測位置:               62.8度 → 63.0度 → 62.5度 → 57.3度（追従できず20度乖離）
次の目標:               53.2度（+9.7度、逆方向へジャンプ）</code></pre>
<p>
方策は最大75度/秒での動きを要求していたが、クランプは15度/秒しか許可しない。結果、①アームが目標から大きく遅れる
→ ②方策が「遅れた状態」を観測して<strong>別の軌道を再計画</strong> → ③目標が逆方向に飛ぶ、という閉ループの破綻が起きていた。
クランプを絞れば絞るほど乖離が拡大し、振動が悪化する。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
<code>time_scale</code> パラメータを新設。制御周期そのものを引き伸ばし、<strong>観測間隔とアーム速度を同じ比率で一緒に遅くする</strong>方式に変更した。
これにより方策が2つの観測間で条件付けている「空間的な変化量」が学習時と一致したまま保たれ、閉ループを壊さずに減速できる。
これは実機で動作確認済みの <code>ReplayUmiOnFairino5.py --time_scale</code> と同じ考え方である。
速度クランプ（<code>joint_vel_limit_scale</code>）は本来の役割である安全網に戻し、余裕を持たせた値（2.0）とした。
</div>
</div>

<h2 id="bug8">⑧ タスク完了後に無限に同じ動作を繰り返す</h2>
<div class="bug-block">
<span class="label sev-mid">中</span>
<span class="file-badge">common/base/RolloutBase.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
タスクが完了しても方策が停止せず、<strong>何度も同じ動作を繰り返してしまう</strong>。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
学習データは「開始姿勢を離れる→作業→開始姿勢へ戻る」という<strong>往復構造</strong>で記録されている。
エピソード相対座標では、開始状態と終了状態がどちらも原点（単位姿勢）となり<strong>数学的に区別がつかない</strong>。
模倣学習の方策自体には「完了」という概念がなく、実機では報酬(reward)が常に0のため既存の <code>auto_exit</code> 判定も機能しない。
結果として、方策は「完了状態」を「新しいエピソードの開始」と誤認し、永久にタスクを繰り返す。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
<code>check_task_finished()</code> を新設。学習データ自身の統計（状態の並進レンジ）からしきい値を算出し、
<strong>「まず一定距離以上離れたことを確認してから、一定距離以内に戻ったことを検出する」</strong>方式で往復完了を判定する。
先に離脱を確認することで、「未開始」と「完了」の曖昧さを解消している。<code>--no_auto_finish</code> で無効化も可能。
</div>
</div>

<h2 id="bug9">⑨ 絶対上限値0.5度でタスクが完了不能に</h2>
<div class="bug-block">
<span class="label sev-high">高</span>
<span class="file-badge">envs/real/fairino5/RealFairino5EnvBase.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
「一気に大きく動くと物が壊れる」という要望を受け、ServoJ送信直前に<strong>1指令あたりの絶対移動量上限</strong>（<code>max_joint_pos_delta_deg</code>）を新設し、
安全側の値として0.5度を設定したところ、⑧の完了検出が働かず、タスクが<strong>永久に終わらなくなった</strong>。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
上限を厳しくしすぎると、指令が常に大きく遅れた状態になり、方策は「学習時には存在しなかった、目標から大きく取り残された状態」を
観測し続けることになる。そのような状態からの正しい行動を学習していないため、<strong>アームはその場で停滞</strong>してしまう。
実測では、目標地点（変位0.275m）まで到達した後、1499ステップ経過してもなお0.268m地点に留まり続けた
（開始位置に戻るための帰還しきい値は0.057m）。安全のための制限が、方策を学習分布の外へ押し出していた。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
方策の実際の要求速度を計測（<code>time_scale=2.0</code>、クランプ無効時）した上で数値を再検討。
<pre><code>1ティックあたりの要求角度: 95%点=0.20度 / 99%点=3.43度 / 最大=13.68度（稀なスパイク）</code></pre>
最大値を <strong>2.0度</strong> に設定。通常動作（0.2度）の10倍の余裕を持たせつつ、危険な13.7度スパイクは約7指令に分散させる。
この設定でdry_run検証を行い、学習時（85ステップ）とほぼ同等の<strong>65ステップで完走</strong>することを確認した。
</div>
</div>

<h2 id="bug10">⑩ 起動直後の急加速（バン！）― 観測の同期漏れ</h2>
<div class="bug-block">
<span class="label sev-critical">重大</span>
<span class="file-badge">envs/operation/OperationRealFairino5Demo.py</span>
<span class="file-badge">envs/operation/OperationRealFairino3Demo.py</span>
<span class="file-badge">envs/operation/OperationRealFairinoDualDemo.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
②の修正（IK内部状態の同期）を入れた後の実機テストで、<strong>ロールアウト開始直後にアームが急加速</strong>する現象が発生した。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
これは②の修正が新たに生んだ副作用であった。<code>MoveToInitPhase.start()</code> 内の <code>move_to_init_pose()</code> は
<strong>ブロッキングのMoveJでアームを大きく動かす</strong>が、これは <code>env.step()</code> ループの<strong>外側</strong>で実行される。
そのため <code>self.obs</code>（観測値）はMoveJ<strong>実行前</strong>の姿勢を保持したままになる。
②で追加した毎ティックの同期処理がこの古い <code>obs</code> を使ってしまい、IKの内部状態をMoveJ前の姿勢へ巻き戻してしまう結果、
最初の指令が「MoveJ前の位置へ戻れ」という<strong>14度も離れた目標</strong>になっていた。
これを速度クランプの上限いっぱい（60度/秒）で実行しようとしたため、静止状態からの急加速となった。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
<code>move_to_init_pose()</code> の直後に <code>self.op.obs = self.op.env.unwrapped._get_obs()</code> を追加し、観測値を強制的に更新するようにした。
同一構造を持つ3つのoperationクラス（Fairino5／Fairino3／FairinoDual）すべてに同じ修正を適用した。
検証の結果、起動時の乖離は <strong>13.82度 → 0.000度</strong> に解消された。
</div>
</div>

<h2 id="bug11">⑪ グリッパ状態読み取り警告のスパム</h2>
<div class="bug-block">
<span class="label sev-info">軽微</span>
<span class="file-badge">envs/real/fairino5/RealFairino5EnvBase.py</span>

<div class="callout symptom">
<span class="ctitle">症状</span>
ロールアウト起動後、<code>n</code> キーを押す前の待機中に
<code>Failed to read gripper DO state: no gripper command sent yet.</code> という警告が延々と流れ続けていた。
</div>

<div class="callout cause">
<span class="ctitle">原因</span>
<code>env.step()</code> はプロセス開始直後から毎ティック呼ばれるが、グリッパへの最初の指令は <code>move_to_init_pose()</code>／<code>GraspPhase</code>
（<code>n</code> キー押下後）まで送られない。この「まだ指令を送っていない」という<strong>正常な起動時の状態</strong>を例外として <code>raise</code> し、
それを毎ティック <code>print</code> していたため、実害のない警告が大量に表示されていた。
</div>

<div class="callout fix">
<span class="ctitle">修正</span>
「未指令」の状態を通常の分岐として扱うように変更し、警告を出さずに最後の既知値（デフォルト50%）を静かに使うようにした。
実際の異常（グリッパ読み取り失敗など）を知らせる別の警告文には影響していない。
</div>
</div>

<hr class="divider">

<h2 id="tested">検証したが効果がなかった対策</h2>
<p>
「動きがガクガクする」という残存課題に対して、以下の2つの対策を実装・検証したが、<strong>いずれも測定の結果、効果が確認できなかった</strong>ため、
参考情報として記録する。
</p>

<h3>方策の目標間を線形補間する</h3>
<p>
方策の目標が <code>skip</code> ティックごとにしか更新されず、その間は目標が一定に保持されるため、
「動いて止まる」の階段状の動きになっているのではという仮説のもと、目標間を滑らかに補間する処理（<code>get_interpolated_policy_action()</code>）を実装した。
</p>
<div class="metric-row">
  <div class="metric"><span class="k">動作中の停止ティック</span><span class="v">0.0%</span></div>
  <div class="metric"><span class="k">段差比(peak/median) 補間前</span><span class="v">3.2</span></div>
  <div class="metric"><span class="k">段差比(peak/median) 補間後</span><span class="v">3.1</span></div>
</div>
<p>
実測の結果、そもそもアームは止まっておらず、段差比もほぼ変化しなかった。<code>max_joint_pos_delta_deg</code> による分散が既に目標を複数指令に
分けていたため、補間の余地がなかったと考えられる。遅延が増えるだけなので、デフォルトでは無効化し（<code>--action_interp</code> で有効化可能）。
</p>

<h3>EMA平滑化（<code>command_smoothing_alpha</code>）の強化</h3>
<p>
指令に対する低域通過フィルタを強めれば滑らかになるのではという仮説のもと、係数を下げて検証した。
</p>
<table>
<tr><th>alpha</th><th>タスク完了</th><th>段差ばらつき (std/mean)</th></tr>
<tr><td>0.30（既定）</td><td>○ 完了</td><td>0.77</td></tr>
<tr><td>0.12（強め）</td><td>✗ 完了せず</td><td>0.75（改善なし）</td></tr>
</table>
<p>
平滑化を強めても改善は見られず、むしろ⑨と同種の「遅延しすぎて方策が分布外の状態を観測し停滞する」問題を引き起こした。
残るムラは<strong>方策自身が出力する軌道に内在するスパイク</strong>であり、指令の後段のフィルタでは除去できないという結論に至った。
既定値0.3のまま据え置いている。
</p>

<h2 id="tools">新規に作成した診断ツール</h2>

<div class="newfile">
<h4><code>misc/CheckDiffusionPolicyPrediction.py</code></h4>
<p>
学習に使ったエピソードの記録済み画像・状態を、学習時と全く同じ前処理でモデルに入力し、予測アクションと記録アクションを比較するオフライン検証スクリプト。
実機を一切動かさずにチェックポイント自体の妥当性を検証できる。<code>--image_mode all</code> を指定すると、
画像を「実画像／最初のフレームで凍結／真っ黒」の3パターンに差し替えて予測を比較でき、
<strong>モデルが本当に画像に反応しているか</strong>（状態だけのショートカット学習に陥っていないか）を検証できる。
</p>
</div>

<div class="newfile">
<h4><code>misc/RolloutWithRecordedImages.py</code></h4>
<p>
<code>bin/Rollout.py</code> と全く同じ実機制御ループ（状態は実機から読み、指令は実機のServoJへ送る＝閉ループ）を使いつつ、
ハンドカメラの画像だけを学習エピソードの記録映像に差し替えて動かすスクリプト。
「モデルは正常なのに実機の挙動がおかしい」場合に、原因が<strong>カメラ側のパイプライン</strong>にあるのか
<strong>実機の制御ループ</strong>にあるのかを切り分けるために作成した。本書の②〜⑩の発見は、主にこのツールによるdry_run検証で行われた。
</p>
</div>

<h2 id="final">最終的な推奨設定</h2>
<p><code>envs/configs/RealFairino5DemoEnv_UmiPolicy.yaml</code> の最終的な設定値と、それぞれの役割をまとめる。</p>

<table>
<tr><th>設定項目</th><th>値</th><th>役割</th></tr>
<tr>
  <td><code>time_scale</code></td>
  <td>2.0<br><span style="font-weight:400;color:#666;">(コマンドラインの<code>--time_scale</code>で上書き可)</span></td>
  <td><strong>速度調整はここで行う。</strong>観測周期とアーム速度を一緒に伸ばすので閉ループが壊れない。実運用では8.0まで上げて滑らかさを確認済み。</td>
</tr>
<tr>
  <td><code>joint_vel_limit_scale</code></td>
  <td>2.0</td>
  <td><strong>安全網。速度調整には使わない。</strong>絞ると閉ループが破綻し振動する（⑦参照）。余裕を持たせておき、<code>hit_hard_clip</code>がほぼ0であることを確認する。</td>
</tr>
<tr>
  <td><code>max_joint_pos_delta_deg</code></td>
  <td>2.0</td>
  <td>1指令あたりの絶対移動量上限。「一気に大きく動く」ことを防ぐ最終防衛線。厳しくしすぎるとタスクが完了不能になる（⑨参照）ため、下げる場合は必ずdry_runで完走確認する。</td>
</tr>
<tr>
  <td><code>command_smoothing_alpha</code></td>
  <td>0.3</td>
  <td>EMA平滑化。既定値のまま。強めても効果がないことを確認済み。</td>
</tr>
<tr>
  <td>追従誤差アボート</td>
  <td>30度</td>
  <td>指令と実測の乖離がこれを超えたら送信を停止するハードストップ（コード内定数）。</td>
</tr>
<tr>
  <td><code>--max_policy_steps</code></td>
  <td>既定: 学習エピソード最長×15</td>
  <td>タスク完了検出（⑧）が主。これは暴走時の保険。</td>
</tr>
</table>

<p class="footer-note">
本報告書は、実機に対する複数回のdry_run検証および実機テストのログ解析に基づいて作成した。
数値は特に断りのない限り、チェックポイント <code>RealUMIDemo_20260815_160937/policy_best.ckpt</code> を用いた検証結果である。
</p>

</body>
</html>

"""


def find_chrome():
    for name in ("google-chrome", "chromium-browser", "chromium"):
        path = shutil.which(name)
        if path is not None:
            return path
    raise RuntimeError(
        "No Chrome/Chromium binary found (tried google-chrome, "
        "chromium-browser, chromium). Install one to build this PDF."
    )


def main():
    html_path = os.path.join(os.path.dirname(__file__), "_rollout_bugfix_explained.tmp.html")
    with open(html_path, "w") as f:
        f.write(HTML)

    try:
        subprocess.run(
            [
                find_chrome(),
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--print-to-pdf={OUT_PDF}",
                "--print-to-pdf-no-header",
                "--no-pdf-header-footer",
                "--virtual-time-budget=10000",
                f"file://{html_path}",
            ],
            check=True,
        )
    finally:
        os.remove(html_path)

    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
