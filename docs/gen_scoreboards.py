# Generates the scoreboard theme variants. Same structure as the CRT original
# (title, subtitle, progress bar, HIGH SCORES, RANK/AGENT/SCORE table, footer);
# only the theme changes.
import os
SC = os.path.dirname(os.path.abspath(__file__))

ROWS = [
    ('1ST', 'OPINE-WORLD (OURS)', '78.4'),
    ('2ND', 'GPT-5.6 SOL (MAX)',  '7.8'),
    ('3RD', 'OPUS 4.8 (HIGH)',    '1.5'),
    ('4TH', 'GEMINI 3.1 PRO',     '0.4'),
    ('5TH', 'GROK 4.20',          '0.1'),
]
TITLE = 'ARC-AGI-3 2026'
SUBTITLE = 'COMMUNITY LEADERBOARD'
FOOT = '* Current rank 1 solution wrt current verified and community leaderboard &middot; July 2026'

BASE = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap" rel="stylesheet">
<style>
html,body{{margin:0;padding:0;overflow:hidden;background:{page_bg}}}
.stage{{position:relative;width:1600px;height:{stage_h}px;overflow:hidden;background:{stage_bg};font-family:{font}}}
{deco_css}
.inner{{position:absolute;inset:0;padding:60px 150px 110px;display:flex;flex-direction:column;justify-content:center}}
.title{{text-align:center;font-size:{title_size}px;transform:scaleX(1.14);transform-origin:center;{title_css}}}
.subtitle{{text-align:center;font-size:26px;letter-spacing:10px;margin-top:34px;{subtitle_css}}}
.barwrap{{margin:60px 10px 0;height:64px;position:relative;{barwrap_css}}}
.fill{{position:absolute;left:5px;top:5px;bottom:5px;width:78.4%;{fill_css}}}
.barlab{{position:absolute;top:50%;transform:translateY(-50%);font-size:26px;left:24px;z-index:2;font-family:{mono};{barlab_css}}}
.barlab.right{{left:auto;right:24px;{barlabr_css}}}
.hs{{text-align:center;font-size:37px;letter-spacing:12px;margin-top:72px;{hs_css}}}
table{{width:100%;margin-top:84px;border-collapse:collapse}}
td{{font-size:30px;padding:27px 0;letter-spacing:2px;font-weight:700}}
td.rank{{width:270px;font-family:{mono}}}
td.score{{text-align:right;width:330px;font-family:{mono};font-size:36px}}
{row_css}
.foot{{position:absolute;left:0;right:0;bottom:34px;text-align:center;font-size:15px;letter-spacing:2px;{foot_css}}}
</style>
</head>
<body>
<div class="stage">
{deco_html}
  <div class="inner">
    <div class="title">{title}</div>
    <div class="subtitle">{subtitle}</div>
{bar_html}
    <table>
      <tr class="hdr"><td>AGENT</td><td class="score" style="font-size:30px">SCORE</td></tr>
{rows}
    </table>
    <div class="foot">{foot}</div>
  </div>
{overlay_html}
</div>
</body>
</html>
'''

def rows_html():
    return '\n'.join(
        f'      <tr class="r{i+1}"><td>{n}</td><td class="score">{s}</td></tr>'
        for i, (r, n, s) in enumerate(ROWS))

THEMES = {}

FLAT = ['#5df06a', '#f4d63a', '#4aa3ff', '#ff5ec4', '#ff9a2e']  # CRT palette

def flat_rows(colors, hdr, border=None):
    return (f'.hdr td{{color:{hdr};font-weight:600}}\n' +
            '\n'.join(f'.r{i+1} td{{color:{c}}}' for i, c in enumerate(colors)))

# ---- clean: Inter, flat colors on black ----
THEMES['clean'] = dict(
    page_bg='#000', stage_bg='#000',
    font="'Inter',system-ui,sans-serif", mono="'JetBrains Mono',monospace",
    deco_css='', deco_html='  <div class="grid"></div>', overlay_html='',
    title_size=72, title_css='font-weight:800;letter-spacing:1px;color:#b48cf2',
    subtitle_css='color:#8a8f98;font-weight:600',
    barwrap_css='border:1px solid #3a3d45;background:#0d0e11;border-radius:6px',
    fill_css='background:linear-gradient(90deg,#f7b04a,#d8571c);border-radius:3px',
    barlab_css='color:#000;font-weight:800', barlabr_css='color:#e8e8e8;font-weight:700',
    hs_css='color:#f2f2f2;font-weight:800',
    row_css=flat_rows(FLAT, '#69dbff', '#1f2126'),
    foot_css='color:#6a6f78',
)

# ---- bold: heavy type, larger first row, flat colors on black ----
THEMES['bold'] = dict(
    page_bg='#000', stage_bg='#000',
    font="'Inter',system-ui,sans-serif", mono="'JetBrains Mono',monospace",
    deco_css='', deco_html='  <div class="grid"></div>', overlay_html='',
    title_size=84, title_css='font-weight:900;letter-spacing:-1px;color:#b48cf2',
    subtitle_css='color:#9a9fa8;font-weight:700',
    barwrap_css='border:2px solid #43464e;background:#0d0e11;border-radius:10px',
    fill_css='background:linear-gradient(90deg,#f7b04a,#d8571c);border-radius:6px',
    barlab_css='color:#000;font-weight:800', barlabr_css='color:#fff;font-weight:800',
    hs_css='color:#fff;font-weight:900',
    row_css=flat_rows(FLAT, '#69dbff', '#222429') +
        '\n.r1 td{font-size:31px}\n.r1 td.score{font-size:42px}',
    foot_css='color:#6a6f78',
)

# ---- mono: JetBrains Mono everywhere, flat colors on black ----
THEMES['mono'] = dict(
    page_bg='#000', stage_bg='#000',
    font="'JetBrains Mono',monospace", mono="'JetBrains Mono',monospace",
    deco_css='', deco_html='  <div class="grid"></div>', overlay_html='',
    title_size=66, title_css='font-weight:800;letter-spacing:2px;color:#b48cf2',
    subtitle_css='color:#8a8f98;font-weight:500',
    barwrap_css='border:1px solid #3a3d45;background:#0d0e11',
    fill_css='background:linear-gradient(90deg,#f7b04a,#d8571c)',
    barlab_css='color:#000;font-weight:800', barlabr_css='color:#e8e8e8;font-weight:700',
    hs_css='color:#f2f2f2;font-weight:800',
    row_css=flat_rows(FLAT, '#69dbff', '#1f2126'),
    foot_css='color:#6a6f78',
)


def svg_grid(h, w=1600, wt=95, wb=185, op=0.07, sw=2):
    # pseudo-perspective grid: straight verticals converge to a vanishing point;
    # horizontal spacing grows top-to-bottom to match cell width at each height,
    # so every cell is locally square while still reading as a tilted floor
    lines = []
    y = -20.0
    while y < h + 60:
        s = wt + (wb - wt) * max(0.0, min(1.0, y / h))
        lines.append(f'<line x1="0" y1="{y:.0f}" x2="{w}" y2="{y:.0f}"/>')
        y += s
    cx = w / 2
    k = wt / wb
    x = cx % wb - wb * 12
    while x < w + wb * 12:
        xt = cx + (x - cx) * k
        lines.append(f'<line x1="{x:.0f}" y1="{h}" x2="{xt:.0f}" y2="0"/>')
        x += wb
    return (f'  <svg class="grid" width="{w}" height="{h}" '
            f'style="position:absolute;inset:0" stroke="rgba(255,255,255,{op})" '
            f'stroke-width="{sw}">' + ''.join(lines) + '</svg>')

BAR = '    <div class="barwrap"><div class="fill"></div><div class="barlab">78.4%</div><div class="barlab right">100%</div></div>'
import copy
THEMES['nobar'] = copy.deepcopy(THEMES['clean'])
THEMES['nobar']['stage_h'] = 960
THEMES['nobar']['title_size'] = 64
for name, t in THEMES.items():
    t.setdefault('bar_html', '' if name == 'nobar' else BAR)
    t.setdefault('stage_h', 1240)
    t['deco_html'] = svg_grid(t['stage_h'])
    html = BASE.format(title=TITLE, subtitle=SUBTITLE, foot=FOOT, rows=rows_html(), **t)
    open(f'{SC}/scoreboard_{name}.html', 'w').write(html)
    print('wrote', name)
