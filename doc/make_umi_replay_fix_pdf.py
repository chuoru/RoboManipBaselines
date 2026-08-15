"""Build the beginner-friendly PDF write-up of the UMI->FR5 replay fix."""

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    KeepTogether,
    Frame,
    Image,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

OUT = "/tmp/claude-1000/-home-sandbox-Work-RoboManipBaselines/08bf81df-4778-439c-a023-940e39849f8b/scratchpad"
PDF = "/home/sandbox/Work/RoboManipBaselines/doc/umi_replay_fix_explained.pdf"

# Noto Sans CJK ships only as .ttc collections with PostScript/CFF outlines,
# which reportlab cannot embed, and the one plain-TrueType Japanese font on
# this box (Droid Sans Fallback) has no Latin/digit glyphs at all -- it
# silently dropped every "UMI", "FR5" and number from the first build. So the
# JP face is extracted from the collection and its CFF outlines converted to
# TrueType glyf up front (see the conversion step alongside this script);
# that font has both Japanese and Latin coverage and embeds cleanly.
_JP_R = OUT + "/NotoJP.ttf"
_JP_B = OUT + "/NotoJP-Bold.ttf"
pdfmetrics.registerFont(TTFont("NotoJP", _JP_R))
pdfmetrics.registerFont(TTFont("NotoJP-B", _JP_B))

ss = getSampleStyleSheet()
BODY = ParagraphStyle(
    "body", parent=ss["Normal"], fontName="NotoJP", fontSize=10.2, leading=17,
    spaceAfter=7, alignment=TA_LEFT,
)
H1 = ParagraphStyle(
    "h1", parent=BODY, fontName="NotoJP-B", fontSize=19, leading=26,
    spaceAfter=4, textColor=colors.HexColor("#1a3a5c"),
)
SUB = ParagraphStyle(
    "sub", parent=BODY, fontSize=10.5, textColor=colors.HexColor("#5a6b7a"), spaceAfter=16,
)
H2 = ParagraphStyle(
    "h2", parent=BODY, fontName="NotoJP-B", fontSize=13.5, leading=20,
    spaceBefore=13, spaceAfter=6, textColor=colors.HexColor("#1a3a5c"),
)
H3 = ParagraphStyle(
    "h3", parent=BODY, fontName="NotoJP-B", fontSize=11.2, leading=17,
    spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#22516e"),
)
CAP = ParagraphStyle(
    "cap", parent=BODY, fontSize=8.8, leading=13,
    textColor=colors.HexColor("#5a6b7a"), spaceBefore=3, spaceAfter=11,
)
NOTE = ParagraphStyle(
    "note", parent=BODY, fontSize=9.8, leading=16,
    leftIndent=9, borderPadding=8, backColor=colors.HexColor("#f2f6fa"),
    borderColor=colors.HexColor("#c8d8e6"), borderWidth=0.8, spaceBefore=5, spaceAfter=11,
)


def p(t):
    return Paragraph(t, BODY)


def b(t):
    return f'<font name="NotoJP-B" color="#0b3d62">{t}</font>'


def img(path, w_mm):
    from PIL import Image as PILImage

    iw, ih = PILImage.open(path).size
    w = w_mm * mm
    return Image(path, width=w, height=w * ih / iw)


def table(rows, widths, header=True):
    t = Table(rows, colWidths=[w * mm for w in widths], hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "NotoJP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.3),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d8e6")),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "NotoJP-B"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff6")),
        ]
    t.setStyle(TableStyle(style))
    return t


story = []
A = story.append

A(Paragraph("UMIで記録した動きを<br/>ロボットアームで再生できなかった理由", H1))
A(Paragraph(
    "MuJoCoシミュレーション上のFairino FR5アームでの調査記録　／　"
    "対象ファイル: misc/ReplayUmiOnFairino5.py", SUB))

A(Paragraph("1. 何をしたかったのか", H2))
A(p(
    "この作業のゴールはシンプルです。"
    + b("「人が手で持つ装置(UMI)を動かして、その動きを記録し、あとからロボットアームに同じ動きをさせる」")
    + "。これができれば、ロボットに教えたい動作を、ロボット本体を直接動かさずに人の手で自然に記録できます。"))
A(p(
    "記録のときは、UMIの動きに合わせてシミュレーション内のアームも一緒に動かして、画面で確認しながら進めました。"
    "このときは何の問題もなく、アームはきれいに追従していました。"))

A(Paragraph("2. ところが再生すると全く違う動きになった", H2))
A(p(
    "同じデータを読み込んで再生(リプレイ)させると、アームは記録とは似ても似つかない動きをし、"
    "途中で自分の腕同士がぶつかって(自己干渉)止まってしまいました。"))
A(Paragraph(
    b("これは非常に奇妙な状況です。") +
    "記録のときに同じシミュレーションの中で問題なく動けていたのですから、"
    "その動き自体が「無理な動き」であるはずがありません。"
    "つまり原因はロボットや環境ではなく、" + b("再生する側のプログラムにある") + "はずです。", NOTE))

A(Paragraph("3. 原因は3つありました", H2))
A(p("調べた結果、独立した3つのバグが重なっていました。順に説明します。"))

A(Paragraph("原因① 読み込むデータの種類を間違えていた", H3))
A(p(
    "記録ファイルには、よく似た2種類の位置データが保存されています。"))
A(table([
    ["データ名", "意味"],
    ["command_eef_pose", "「ここに動いてほしい」という指令の値"],
    ["measured_eef_pose", "「実際にこうなった」という結果の値"],
], [45, 105]))
A(Spacer(1, 5))
A(p(
    "記録時のプログラムが使っていたのは" + b("指令の値") + "でした。"
    "ところが再生プログラムは" + b("結果の値") + "を読んでいました。"
    "この2つは実データで" + b("最大17cmもズレて") + "いました。"
    "出発点が違うのですから、違う動きになるのは当然です。"))

A(Paragraph("原因② 重力による「たわみ」を誤解していた（最大の原因）", H3))
A(p(
    "これが一番わかりにくく、一番影響が大きい原因でした。"))
A(p(
    "シミュレーション内のロボットの関節は、本物のモーターと同じように"
    + b("重力で少したわみます") + "。"
    "「この角度になって」と指令しても、腕の重さで実際にはほんの少し足りない角度で止まります。"
    "実際に測ったところ、指令を出し続けて400回シミュレーションを進めても、"
    "肩の関節には" + b("2.9度のズレが永久に残り続けました") + "。"
    "しかもこれは、衝突判定を全部オフにしても同じでした。つまり衝突とは無関係で、"
    "そもそも消せない性質のズレです。"))
A(p(
    "記録時のプログラムは実際の位置を見ていなかったので、このたわみと共存していました。"
    "ところが再生プログラムは毎回実際の位置を確認し、このたわみを"
    + b("「追従が遅れている！」と誤解") + "して、"
    "目標を通り越して強く指令するようになりました。"
    "しかしたわみは絶対に消えないので補正は永久に終わらず、指令はどんどん過剰になり、"
    "記録された経路から外れて、最後は" + b("記録時には起きなかった自己干渉") + "に突っ込んでいました。"))

A(Paragraph("この誤解を図にすると", H3))
A(KeepTogether([
    img(OUT + "/fig_loop.png", 165),
    Paragraph(
        "左（修正前）：実際の位置を見て補正しようとするため、絶対に消えないたわみと戦い続けて暴走する。"
        "右（修正後）：記録時と全く同じ計算だけを行い、実際の位置は見ない。"
        "たわみも記録時と同じだけ発生するので、結果として同じ動きになる。", CAP),
]))

A(Paragraph("原因③ データを勝手に加工していた", H3))
A(p(
    "再生プログラムには、記録の最初の数秒を切り捨てたり、"
    "センサーのノイズを取り除いたりする機能が入っていました。"
    "これ自体は本物のロボットを安全に動かすためには有用ですが、"
    "「記録と同じ動きを再現する」目的では、"
    + b("加工した時点で記録とは別のデータになってしまいます") + "。"
    "特に最初の数秒を切り捨てると、動きの基準となる出発点そのものが変わってしまいます。"))

A(Paragraph("4. どう直したか", K := H2))
A(table([
    ["原因", "修正内容"],
    ["① データの種類", "指令の値(command_eef_pose)を読むように変更"],
    ["② たわみの誤解", "実際の位置を見ない方式(開ループ)に変更。\n記録時と全く同じ計算をそのまま再演する"],
    ["③ 勝手な加工", "--mirror_exact オプションで加工を無効化できるようにした"],
], [32, 118]))

A(Paragraph("5. 本当に直ったのか確かめる", H2))
A(p(
    "「直ったつもり」で終わらせないため、"
    + b("記録時の動画と、再生した動画を1コマずつ画像として比較") + "しました。"))
A(KeepTogether([
    img(OUT + "/fig_result.png", 130),
    Paragraph(
        "動画は圧縮して保存されるため、たとえ完全に同じ映像でも差はゼロにはなりません。"
        "その「圧縮だけによる差(1.34)」が実質的な下限です。修正後の2.25はそれにかなり近い値です。", CAP),
]))
A(p(
    "さらにアームの位置そのものを画像から計算したところ、"
    "640×480ピクセルの画面上で、記録と再生のズレは"
    + b("平均0.73ピクセル、最大でも1.57ピクセル") + "でした。ほぼ完全な一致です。"))

A(KeepTogether([
    Paragraph("6. 実際の映像での比較", H2),
    p("左が記録時、右が再生時です。どの時点でもアームの姿勢が一致しています。"),
    img(OUT + "/fig_frames.png", 150),
    Paragraph("左: 記録時のシミュレーション映像 ／ 右: 修正後の再生映像", CAP),
]))

A(Paragraph("7. この件から学べること", H2))
A(p(
    b("「賢く補正する」ことが正解とは限りません。") +
    "今回の再生プログラムは、実際の位置を確認して誤差を補正するという、"
    "一見とても丁寧で正しそうな作りになっていました。"
    "しかし目的が「記録と同じ動きを再現すること」である以上、"
    "正解は" + b("記録時と全く同じ計算をそのまま繰り返すこと") + "でした。"
    "補正機能はむしろ邪魔をしていたのです。"))
A(p(
    b("「動いているように見える」は証拠になりません。") +
    "記録時のプログラムは実際の位置を確認していなかったため、"
    "実は同じたわみも同じ接触も起きていたのに、それに気づけませんでした。"
    "問題が「再生時だけ」に見えていたのは、"
    "単に再生側だけが正直に現実を確認していたからです。"))
A(p(
    b("最後は必ず目で確かめる。") +
    "今回は数値だけでなく、記録と再生の動画を実際に1コマずつ比較して初めて"
    "「本当に一致した」と判断できました。"))

A(Paragraph("8. 使い方", H2))
A(Paragraph(
    'python ./misc/ReplayUmiOnFairino5.py \\<br/>'
    '&nbsp;&nbsp;./dataset/&lt;DATASET_DIR&gt;/&lt;FILE&gt;.rmb \\<br/>'
    '&nbsp;&nbsp;--vive_config ./teleop/configs/ViveUMI.yaml \\<br/>'
    '&nbsp;&nbsp;--mirror_exact --sim',
    ParagraphStyle("code", parent=BODY, fontName="NotoJP", fontSize=9,
                   leading=14, backColor=colors.HexColor("#f2f6fa"),
                   borderColor=colors.HexColor("#c8d8e6"), borderWidth=0.8,
                   borderPadding=8, spaceBefore=4, spaceAfter=9)))
A(p(
    "実行すると再生の映像が _replay_sim.mp4 として保存され、"
    "記録時の _mujoco_mirror.mp4 とそのまま見比べられます。"))
A(Paragraph(
    b("注意: ") + "pos_scale は動きの大きさ(並進)にだけ掛かり、"
    "回転には掛かりません。記録時と違う値を指定すると再生は一致しなくなります。"
    "また、本物のロボットでの動作確認はまだ行っていません。", NOTE))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("NotoJP", 8)
    canvas.setFillColor(colors.HexColor("#8a99a8"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, str(doc.page))
    canvas.restoreState()


doc = BaseDocTemplate(
    PDF, pagesize=A4,
    leftMargin=20 * mm, rightMargin=20 * mm,
    topMargin=18 * mm, bottomMargin=18 * mm,
    title="UMIで記録した動きをロボットアームで再生できなかった理由",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(story)
print("PDF written:", PDF)
