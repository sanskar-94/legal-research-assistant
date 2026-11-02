"""Build the final report: the original report.pdf, unchanged, plus the Nyaya-Gen
section (IV-E), one closing paragraph in the Conclusion and references [15]-[18].

Pages 1-6 are copied as they are. Page 7 of the original is copied in three
clipped pieces (end of Section IV-D, Conclusion, References) so its text is
never retyped; the new material is laid out between them in the same
two-column style.
"""
import json
import sys
from pathlib import Path

import pymupdf as fitz

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LOG = Path(__file__).resolve().parent.parent / "data" / "generator" / "training_log.json"

COLS = [(45, 302), (315, 570)]
TOP, BOTTOM = 54, 714
CSS = """
* {font-family: 'Times New Roman', Times, serif;}
p {font-size: 10pt; line-height: 11.5pt; text-align: justify; margin: 0; text-indent: 10pt;}
p.c {text-indent: 0;}
p.h {text-indent: 0; text-align: left; font-style: italic;}
p.sec {text-indent: 0; text-align: center; font-variant: small-caps;}
p.cap {font-size: 8pt; line-height: 9.5pt; text-indent: 0; text-align: center;}
p.cell {font-size: 8pt; line-height: 9.2pt; text-indent: 0; text-align: left;}
p.eq {text-indent: 0; text-align: center; font-size: 10pt;}
p.ref {font-size: 8pt; line-height: 9.5pt; text-indent: 0; text-align: justify;}
"""

src = fitz.open(SRC)
out = fitz.open()
out.insert_pdf(src, from_page=0, to_page=5)
W, H = src[0].rect.width, src[0].rect.height

scratch = fitz.open()


def html_height(html, width):
    page = scratch.new_page(width=width + 10, height=3000)
    spare, _ = page.insert_htmlbox(fitz.Rect(0, 0, width, 3000), html, css=CSS, scale_low=1)
    scratch.delete_page(-1)
    return 3000 - spare


def draw_html(page, x, y, width, html):
    h = html_height(html, width)
    page.insert_htmlbox(fitz.Rect(x, y, x + width, y + h + 2), html, css=CSS, scale_low=1)
    return h


# Clipped pieces of original page 7, with everything outside the piece redacted
def clip_piece(rect):
    doc = fitz.open()
    doc.insert_pdf(src, from_page=6, to_page=6)
    pg = doc[0]
    for r in (fitz.Rect(0, 0, W, rect.y0), fitz.Rect(0, rect.y1, W, H),
              fitz.Rect(0, 0, rect.x0, H), fitz.Rect(rect.x1, 0, W, H)):
        if not r.is_empty:
            pg.add_redact_annot(r)
    pg.apply_redactions(images=0, graphics=1)
    return doc


def piece_block(rect, gap_after=0):
    doc = clip_piece(rect)

    def draw(page, x, y, width):
        target = fitz.Rect(x - 5, y, x - 5 + rect.width, y + rect.height)  # pieces start 5 pt left of their column
        page.show_pdf_page(target, doc, 0, clip=rect)
        return rect.height
    return ("draw", rect.height + gap_after, draw)


# Tables
def table_block(caption, header, rows, widths, bold=()):
    def layout(width):
        xs = [0]
        for w in widths:
            xs.append(xs[-1] + w * width)
        return xs

    def row_h(cells, width):
        xs = layout(width)
        return max(html_height(f'<p class="cell">{c}</p>', xs[i + 1] - xs[i] - 3) for i, c in enumerate(cells)) + 1

    def total(width):
        cap = html_height(f'<p class="cap">{caption}</p>', width)
        return cap + 3 + row_h(header, width) + sum(row_h(r, width) for r in rows) + 4

    def draw(page, x, y, width):
        xs = layout(width)
        cap = draw_html(page, x, y, width, f'<p class="cap">{caption}</p>')
        y += cap + 3
        page.draw_line((x, y), (x + width, y), width=0.8)
        for k, cells in enumerate([header] + rows):
            h = row_h(cells, width)
            for i, c in enumerate(cells):
                text = f"<b>{c}</b>" if (k - 1, i) in bold else c
                page.insert_htmlbox(fitz.Rect(x + xs[i] + 1, y + 0.5, x + xs[i + 1] - 2, y + h + 2),
                                    f'<p class="cell">{text}</p>', css=CSS, scale_low=1)
            y += h
            if k == 0:
                page.draw_line((x, y), (x + width, y), width=0.5)
        page.draw_line((x, y), (x + width, y), width=0.8)
        return total(width)
    return ("table", total, draw)


# Fig. 2: training and dev loss
def figure_block():
    log = json.loads(LOG.read_text())
    steps = [(s["update"], s["loss"]) for s in log["steps"]]
    evals = [(e["update"], e["dev_loss"]) for e in log["evals"]]
    caption = ("<b>Fig. 2.</b> Training loss (line) and dev loss (dots) of Nyaya-Gen for each update. "
               "The run stopped at about update 80; the checkpoint at update 50 is kept.")
    plot_h = 110

    def total(width):
        return plot_h + 22 + html_height(f'<p class="cap">{caption}</p>', width) + 6

    def draw(page, x, y, width):
        left, right = x + 26, x + width - 6
        top, bot = y + 4, y + plot_h
        xmax, ymax = 85, 1.0
        px = lambda u: left + (right - left) * u / xmax
        py = lambda v: bot - (bot - top) * min(v, ymax) / ymax
        page.draw_line((left, bot), (right, bot), width=0.6)
        page.draw_line((left, bot), (left, top), width=0.6)
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            page.draw_line((left - 2, py(v)), (left, py(v)), width=0.5)
            page.insert_text((x + 4, py(v) + 2.5), f"{v:.2f}", fontname="Times-Roman", fontsize=6.5)
        for u in (0, 20, 40, 60, 80):
            page.draw_line((px(u), bot), (px(u), bot + 2), width=0.5)
            page.insert_text((px(u) - 3, bot + 9), str(u), fontname="Times-Roman", fontsize=6.5)
        page.insert_text(((left + right) / 2 - 12, bot + 18), "update", fontname="Times-Roman", fontsize=7)
        page.insert_text((left + 4, top + 7), "loss", fontname="Times-Roman", fontsize=7)
        page.draw_polyline([(px(u), py(v)) for u, v in steps], width=0.8)
        for u, v in evals:
            page.draw_circle((px(u), py(v)), 1.8, fill=(0, 0, 0))
        page.draw_line((px(50), top), (px(50), bot), width=0.4, dashes="[2 2] 0")
        page.insert_text((px(50) + 2, top + 8), "kept", fontname="Times-Italic", fontsize=6.5)
        draw_html(page, x, bot + 22, width, f'<p class="cap">{caption}</p>')
        return total(width)
    return ("table", total, draw)


def refs_block(refs):
    def one(ref, width):
        return html_height(f'<p class="ref">{ref[1]}</p>', width - 18)

    def total(width):
        return sum(one(r, width) for r in refs) + 2

    def draw(page, x, y, width):
        for num, text in refs:
            page.insert_htmlbox(fitz.Rect(x, y, x + 18, y + 12), f'<p class="ref">[{num}]</p>', css=CSS, scale_low=1)
            y += draw_html(page, x + 18, y, width - 18, f'<p class="ref">{text}</p>')
        return total(width)
    return ("table", total, draw)


def para(text, indent=True):
    return ("para", text, indent)


def html(h, gap_before=0, gap_after=0):
    return ("html", h, gap_before, gap_after)


exec((Path(__file__).parent / "nyaya_gen_section.py").read_text())  # defines CONTENT_*  # noqa: S102

# Lay out the blocks over the columns
p7 = out.new_page(width=W, height=H)
top_piece = piece_block(fitz.Rect(40, 34, 305, 256))
top_piece[2](p7, 40, 34, 265)
columns = [(p7, COLS[0], 262), (p7, COLS[1], TOP)]
pages = [p7]


def next_column(i):
    while i >= len(columns):
        pg = out.new_page(width=W, height=H)
        pages.append(pg)
        columns.extend([(pg, COLS[0], TOP), (pg, COLS[1], TOP)])
    return columns[i]


def flow(blocks):
    ci, (page, (x0, x1), y) = 0, next_column(0)
    width = x1 - x0

    def advance():
        nonlocal ci, page, x0, x1, y, width
        ci += 1
        page, (x0, x1), y = next_column(ci)
        width = x1 - x0

    for b in blocks:
        kind = b[0]
        if kind == "gap":
            y += b[1]
            continue
        if kind == "html":
            _, h_, gb, ga = b
            need = gb + html_height(h_, width)
            if y + need > BOTTOM:
                advance(); gb = 0
            y += gb
            y += draw_html(page, x0, y, width, h_) + ga
            continue
        if kind in ("table", "draw"):
            need = b[1](width) if kind == "table" else b[1]
            if y + need > BOTTOM:
                advance()
            y += 4 if kind == "table" else 0
            if kind == "draw":
                b[2](page, x0, y, width); y += b[1]
            else:
                y += b[2](page, x0, y, width) + 4
            continue
        _, text, indent = b
        words = text.split(" ")
        first = True
        while words:
            cls = "" if (indent and first) else ' class="c"'
            avail = BOTTOM - y
            full = f"<p{cls}>{' '.join(words)}</p>"
            h_ = html_height(full, width)
            if h_ <= avail + 0.5:
                y += draw_html(page, x0, y, width, full)
                break
            lo, hi = 0, len(words)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if html_height(f"<p{cls}>{' '.join(words[:mid])}</p>", width) <= avail + 0.5:
                    lo = mid
                else:
                    hi = mid - 1
            if lo:
                draw_html(page, x0, y, width, f"<p{cls}>{' '.join(words[:lo])}</p>")
            words = words[lo:]
            first = False if lo else first
            advance()
    return ci, y


flow(BLOCKS)  # noqa: F821  (from nyaya_gen_section.py)
for pg in pages:
    pass
out.save(OUT, garbage=4, deflate=True)
print(f"{OUT}: {len(out)} pages")
