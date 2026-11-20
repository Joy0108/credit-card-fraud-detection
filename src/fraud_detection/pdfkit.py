"""Small layout toolkit on top of ReportLab Platypus for the project report."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

# ---- colours (same system as the figures) ---------------------------------- #
INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
RULE = colors.HexColor("#e1e0d9")
ACCENT = colors.HexColor("#2a78d6")
ACCENT_DARK = colors.HexColor("#1c5cab")
ACCENT_WASH = colors.HexColor("#eef4fc")
FRAUD = colors.HexColor("#eb6834")
FRAUD_WASH = colors.HexColor("#fdf1ec")
NEUTRAL_WASH = colors.HexColor("#f4f4f2")
SURFACE = colors.HexColor("#fcfcfb")

PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X


# ---- fonts ------------------------------------------------------------------ #
def _register_fonts() -> dict:
    win = Path("C:/Windows/Fonts")
    candidates = {
        "UI": win / "segoeui.ttf", "UI-Bold": win / "segoeuib.ttf",
        "UI-Italic": win / "segoeuii.ttf", "UI-Semibold": win / "seguisb.ttf",
        "UI-Light": win / "segoeuil.ttf",
    }
    try:
        import matplotlib
        dv = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    except Exception:  # pragma: no cover
        dv = None
    fallback = {
        "UI": dv / "DejaVuSans.ttf" if dv else None, "UI-Bold": dv / "DejaVuSans-Bold.ttf" if dv else None,
        "UI-Italic": dv / "DejaVuSans-Oblique.ttf" if dv else None,
        "UI-Semibold": dv / "DejaVuSans-Bold.ttf" if dv else None, "UI-Light": dv / "DejaVuSans.ttf" if dv else None,
    }
    for name, path in candidates.items():
        src = path if path.exists() else fallback[name]
        pdfmetrics.registerFont(TTFont(name, str(src)))
    mono = win / "consola.ttf"
    pdfmetrics.registerFont(TTFont("Mono", str(mono if mono.exists() else dv / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("UI", normal="UI", bold="UI-Bold", italic="UI-Italic", boldItalic="UI-Bold")
    return candidates


_register_fonts()

# ---- paragraph styles ------------------------------------------------------- #
BODY = ParagraphStyle("Body", fontName="UI", fontSize=9.6, leading=14.4, textColor=INK,
                      spaceAfter=6, alignment=TA_LEFT)
BODY_TIGHT = ParagraphStyle("BodyTight", parent=BODY, spaceAfter=2)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.4, leading=11.6, textColor=INK_2, spaceAfter=3)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8.3, leading=11.4, textColor=INK_2,
                         spaceBefore=3, spaceAfter=12)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=13, bulletIndent=2, spaceAfter=3.5)
H1 = ParagraphStyle("H1", fontName="UI-Semibold", fontSize=17, leading=22, textColor=INK,
                    spaceBefore=4, spaceAfter=10)
H2 = ParagraphStyle("H2", fontName="UI-Semibold", fontSize=12.2, leading=16, textColor=INK,
                    spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("H3", fontName="UI-Bold", fontSize=9.8, leading=13.5, textColor=INK,
                    spaceBefore=6, spaceAfter=3)
CELL = ParagraphStyle("Cell", fontName="UI", fontSize=8.3, leading=10.8, textColor=INK)
CELL_B = ParagraphStyle("CellB", parent=CELL, fontName="UI-Semibold")
CELL_HEAD = ParagraphStyle("CellHead", parent=CELL, fontName="UI-Semibold", textColor=INK_2, fontSize=8)
CELL_C = ParagraphStyle("CellC", parent=CELL, alignment=TA_CENTER)
CODE = ParagraphStyle("Code", fontName="Mono", fontSize=7.8, leading=10.6, textColor=INK)
TOC1 = ParagraphStyle("TOC1", fontName="UI-Semibold", fontSize=10, leading=17, textColor=INK, leftIndent=0)
TOC2 = ParagraphStyle("TOC2", fontName="UI", fontSize=9, leading=13.5, textColor=INK_2, leftIndent=16)


# ---- document template ---------------------------------------------------- #
class ReportDoc(BaseDocTemplate):
    """A4 document with a cover page, running header/footer and a live TOC."""

    def __init__(self, filename: str, header_text: str, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                         topMargin=22 * mm, bottomMargin=18 * mm, **kw)
        self.header_text = header_text
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        cover = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="cover",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[cover], onPage=self._cover_page),
            PageTemplate(id="body", frames=[frame], onPage=self._body_page),
        ])

    def _cover_page(self, canv, doc):
        canv.saveState()
        canv.setFillColor(ACCENT)
        canv.rect(0, PAGE_H - 7 * mm, PAGE_W, 7 * mm, stroke=0, fill=1)
        canv.restoreState()

    def _body_page(self, canv, doc):
        canv.saveState()
        canv.setFont("UI", 7.8)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN_X, PAGE_H - 13 * mm, self.header_text)
        canv.drawRightString(PAGE_W - MARGIN_X, 10 * mm, f"{doc.page}")
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.6)
        canv.line(MARGIN_X, PAGE_H - 14.8 * mm, PAGE_W - MARGIN_X, PAGE_H - 14.8 * mm)
        canv.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and getattr(flowable, "toc_level", None) is not None:
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (flowable.toc_level, flowable.getPlainText(), self.page, key))
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=flowable.toc_level, closed=False)


def toc() -> TableOfContents:
    t = TableOfContents()
    t.levelStyles = [TOC1, TOC2]
    t.dotsMinLevel = 0
    return t


# ---- flowable helpers ------------------------------------------------------- #
def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def h1(text: str) -> list:
    para = Paragraph(text, H1)
    para.toc_level = 0
    return [CondPageBreak(60 * mm), para]


def h2(text: str) -> list:
    para = Paragraph(text, H2)
    para.toc_level = 1
    return [CondPageBreak(38 * mm), para]


def h3(text: str) -> Paragraph:
    return Paragraph(text, H3)


def bullets(items: list[str], style: ParagraphStyle = BULLET) -> list:
    return [Paragraph(t, style, bulletText="\u2022") for t in items]


def figure(path: Path, caption: str, width: float = CONTENT_W, max_height: float = 120 * mm) -> KeepTogether:
    iw, ih = ImageReader(str(path)).getSize()
    w = width
    h = w * ih / iw
    if h > max_height:
        h = max_height
        w = h * iw / ih
    img = Image(str(path), width=w, height=h)
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 4), img, Paragraph(caption, CAPTION)])


def table(rows: list[list], col_widths: list[float] | None = None, header: bool = True,
          zebra: bool = True, highlight_rows: list[int] | None = None, align_right_from: int | None = None,
          font_size: float | None = None) -> Table:
    """Data table with hairline rules; strings are wrapped as Paragraphs."""
    data = []
    for r, row in enumerate(rows):
        out = []
        for c, cell in enumerate(row):
            if isinstance(cell, str):
                st = CELL_HEAD if (header and r == 0) else CELL
                if align_right_from is not None and c >= align_right_from:
                    st = ParagraphStyle(f"r{st.name}", parent=st, alignment=2)
                if font_size:
                    st = ParagraphStyle(f"f{st.name}", parent=st, fontSize=font_size, leading=font_size * 1.3)
                out.append(Paragraph(cell, st))
            else:
                out.append(cell)
        data.append(out)
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
    ]
    if header:
        cmds += [("LINEBELOW", (0, 0), (-1, 0), 0.9, INK_2), ("BACKGROUND", (0, 0), (-1, 0), NEUTRAL_WASH)]
    start = 1 if header else 0
    for r in range(start, len(data)):
        if zebra and (r - start) % 2 == 1:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f9f9f7")))
        cmds.append(("LINEBELOW", (0, r), (-1, r), 0.4, RULE))
    for r in highlight_rows or []:
        cmds.append(("BACKGROUND", (0, r), (-1, r), ACCENT_WASH))
    t.setStyle(TableStyle(cmds))
    return t


def callout(text: str, title: str | None = None, tone: str = "accent") -> Table:
    bar, wash = (ACCENT, ACCENT_WASH) if tone == "accent" else (FRAUD, FRAUD_WASH) if tone == "fraud" \
        else (MUTED, NEUTRAL_WASH)
    body = []
    if title:
        body.append(Paragraph(title, ParagraphStyle("ct", parent=BODY, fontName="UI-Semibold", spaceAfter=2)))
    if isinstance(text, list):
        body.extend(text)
    else:
        body.append(Paragraph(text, ParagraphStyle("cb", parent=BODY, spaceAfter=0)))
    t = Table([[body]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), wash),
        ("LINEBEFORE", (0, 0), (0, -1), 2.6, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return KeepTogether([t, Spacer(1, 8)])


def kpi_tiles(items: list[tuple[str, str]], cols: int | None = None, accent_first: int = 0,
              value_size: float = 19) -> Table:
    """A row of stat tiles: (value, label)."""
    cols = cols or len(items)
    w = CONTENT_W / cols
    val = ParagraphStyle("kv", fontName="UI-Semibold", fontSize=value_size, leading=value_size * 1.15,
                         textColor=INK)
    lab = ParagraphStyle("kl", fontName="UI", fontSize=7.8, leading=10, textColor=INK_2)
    cells = [[Paragraph(v, val), Spacer(1, 2), Paragraph(l, lab)] for v, l in items]
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    t = Table(rows, colWidths=[w] * cols)
    cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]
    for r in range(len(rows)):
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(items):
                continue
            cmds.append(("BACKGROUND", (c, r), (c, r), ACCENT_WASH if idx < accent_first else NEUTRAL_WASH))
            cmds.append(("LINEABOVE", (c, r), (c, r), 2, ACCENT if idx < accent_first else RULE))
            cmds.append(("LINEAFTER", (c, r), (c, r), 3, colors.white))
    t.setStyle(TableStyle(cmds))
    return t


def code_block(text: str) -> Table:
    lines = [Paragraph(l.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;") or "&nbsp;", CODE)
             for l in text.strip("\n").split("\n")]
    t = Table([[lines]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NEUTRAL_WASH),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t
