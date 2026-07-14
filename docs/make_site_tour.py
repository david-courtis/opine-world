# Scroll-tour GIF of the results blog. Needs a full-page screenshot first:
#   python3 -m http.server -d docs 8013 &
#   google-chrome --headless=new --disable-gpu --window-size=1400,14000 \
#     --virtual-time-budget=10000 --screenshot=page.png http://localhost:8013/index.html
# Section stops are page-y offsets in that 1400px-wide capture.
import math, os, sys
from PIL import Image, ImageOps

SC = os.path.dirname(os.path.abspath(__file__))
PAGE = sys.argv[1] if len(sys.argv) > 1 else f'{SC}/page.png'
OUT = f'{SC}/opine_world_site_tour.gif'
VP = 900
CROP_X = (90, 1310)     # content column, 1220px -> 2:1 to 610
STOPS = [0, 790, 1620, 4730, 5930, 6730, 8950, 9830, 10880, 11632, 12261]
HOLD_MS, STEP_MS = 560, 50

page = Image.open(PAGE).convert('RGB')
OW = (CROP_X[1]-CROP_X[0])//2
OH = VP//2

def shot(y):
    c = page.crop((CROP_X[0], int(y), CROP_X[1], int(y)+VP))
    return c.resize((OW, OH), Image.LANCZOS)

def ease(u):
    return u*u*(3-2*u)

frames, durs = [], []
for i, y in enumerate(STOPS):
    frames.append(shot(y)); durs.append(HOLD_MS)
    if i+1 < len(STOPS):
        d = STOPS[i+1]-y
        n = max(9, min(18, round(6+math.sqrt(d)/2.8)))
        for k in range(1, n):
            frames.append(shot(y+d*ease(k/n))); durs.append(STEP_MS)

q = [ImageOps.posterize(f, 4).quantize(colors=64, dither=Image.NONE) for f in frames]
q[0].save(OUT, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True)
print(len(frames), 'frames', f'{OW}x{OH}', f'{os.path.getsize(OUT)/1e6:.1f} MB, ~{sum(durs)/1000:.1f}s')
