"""Burned-in bilingual subtitles (Chinese + English), drawn after post-processing."""
from __future__ import annotations

import skia

from .core import H, W, clamp, col, draw_text, font, smoothstep, FONT_SANS

ZH = None
EN = None


def draw(c, tl, T, enabled=True):
    global ZH, EN
    if not enabled:
        return
    ln = None
    for cand in tl.lines:
        if cand['start'] - 0.12 <= T <= cand['end'] + 0.35:
            ln = cand
            break
    if ln is None:
        return
    if ZH is None:
        ZH = font(46, FONT_SANS, 2)
        EN = font(27, FONT_SANS, 2)
    a = smoothstep(ln['start'] - 0.12, ln['start'] + 0.05, T) * (1 - smoothstep(ln['end'] + 0.1, ln['end'] + 0.35, T))
    if a <= 0:
        return
    zh = ln['text'].replace('……', '…')
    en = ln.get('en', '')
    y_zh = H - 108
    y_en = H - 62
    shadow = skia.Paint(AntiAlias=True, Color=col('#000010', 0.85 * a))
    shadow.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 5))
    tint = {'deng': '#ffe7b8', 'guang': '#fff6d6', 'whale': '#d6ecff', 'narrator': '#ffffff'}.get(ln['speaker'], '#ffffff')
    draw_text(c, zh, W / 2, y_zh + 2, ZH, shadow)
    draw_text(c, zh, W / 2, y_zh, ZH, skia.Paint(AntiAlias=True, Color=col(tint, a)))
    if en:
        draw_text(c, en, W / 2, y_en + 2, EN, shadow)
        draw_text(c, en, W / 2, y_en, EN, skia.Paint(AntiAlias=True, Color=col('#c9d3e8', 0.9 * a)))
