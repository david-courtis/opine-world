# Builds the X-post GIFs: our replays vs GPT-5.6 Sol max-effort full sessions.
#   opine_world_vs_sol.gif       one screen, 5 games (5 columns x 2 rows)
#   opine_world_vs_sol_20.gif    all 20 wins, four screens of 5 in sequence
#   opine_world_vs_sol_1v1.gif   one game at a time, ours left / Sol right,
#                                15 games (wins where our move count is not
#                                significantly above Sol's)
# Timing rule: within a screen all OPINE runs are stretched to end together;
# Sol plays at the same moves/sec as the OPINE run of its game, freezing early
# if its session was shorter and running past the OPINE finish if longer. The
# screen advances when every run has ended. If Sol outlasts the OPINE run,
# its remaining moves fast-forward so it finishes within ~2 seconds. The
# fractional position inside a move indexes that move's animation frames,
# so intermediate ticks play.
#
# Sol recordings are NDJSON, one line per action with all animation frames:
#   https://arcprize.org/api/recordings/<game_id-version>/<session_id>
# saved next to this script as sol_<game>.rec; session ids resolve from
# POST https://arcprize.org/api/models {"game_id": ...} at config
# openai-gpt-5-6-sol-max (list mirrored in sol_sessions.txt).
import gzip, json, os
from PIL import Image, ImageDraw, ImageFont

DOCS = '/home/david/git/opine-world/docs'
SC = os.path.dirname(os.path.abspath(__file__))
GP = ['#FFFFFF','#CCCCCC','#999999','#666666','#333333','#000000','#E53AA3','#FF7BCC',
      '#F93C31','#1E93FF','#88D8F1','#FFDC00','#FF851B','#921231','#4FCC30','#A356D6']
FLATPAL = sum([[int(c[i:i+2],16) for i in (1,3,5)] for c in GP], []) + [0]*(768-48)

TL = {'su15':9,'cd82':6,'m0r0':6,'tr87':6,'tu93':9,'wa30':9,'sb26':8,'r11l':6,
      're86':8,'g50t':7,'ls20':7,'tn36':7,'sp80':6,'vc33':7,'dc22':6,'ar25':8,
      'lp85':8,'sc25':6,'cn04':6,'ft09':6}

V5 = [['tu93','sb26','tr87','tn36','vc33']]
V20 = [['su15','cd82','m0r0','tr87','tu93'],
       ['wa30','sb26','r11l','re86','g50t'],
       ['ls20','tn36','sp80','vc33','dc22'],
       ['ar25','lp85','sc25','cn04','ft09']]
# wins minus the 5 where our move count is far above Sol's
# (dc22 3.5x, cd82 3.0x, su15 3.0x, g50t 1.9x, ls20 1.5x), by score gap
V1V1 = ['m0r0','tr87','tu93','wa30','sb26','r11l','re86','tn36','sp80','vc33',
        'ar25','lp85','sc25','cn04','ft09']

GREEN = (23,138,61)
RED = (196,60,46)
GRAY = (130,130,130)
BLACK = (26,26,26)
FLASH = 6
STOP_RATIO = 0.5

def font(name, sz):
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{name}.ttf', sz)
F, FB, FS = font('DejaVuSans',13), font('DejaVuSans-Bold',14), font('DejaVuSans',12)
FM = font('DejaVuSansMono-Bold', 15)

def load_ours(gid):
    b = json.load(gzip.open(f'{DOCS}/replay_data/{gid}.json.gz'))
    steps, grid, frame_step = [], None, []
    for f in b['frames']:
        if 'g' in f:
            grid = [row[:] for row in f['g']]
        elif 'd' in f:
            for x, y, v in f['d']:
                grid[y][x] = v
        frame_step.append(f['s'])
        s = f['s']
        while len(steps) <= s:
            steps.append([])
        steps[s].append(bytes(v for row in grid for v in row))
    for i in range(len(steps)):
        if not steps[i]:
            steps[i] = [steps[i-1][-1]]
    lvs = sorted({frame_step[i] for i in b['level_steps']})
    levels = [sum(1 for L in lvs if L <= s) for s in range(len(steps))]
    return steps, levels

def cap_sub(frames, k=10):
    if len(frames) <= k:
        return frames
    idx = sorted({round(i*(len(frames)-1)/(k-1)) for i in range(k)})
    return [frames[i] for i in idx]

def load_sol(gid):
    steps, levels = [], []
    for ln in open(f'{SC}/sol_{gid}.rec'):
        d = json.loads(ln)['data']
        steps.append(cap_sub([bytes(v for row in sub for v in row) for sub in d['frame']]))
        levels.append(d['levels_completed'])
    return steps, levels

CACHE = {}
def get(gid):
    if gid not in CACHE:
        CACHE[gid] = (load_ours(gid), load_sol(gid))
    return CACHE[gid]

def grid_img(gbytes, cell):
    im = Image.frombytes('P', (64,64), gbytes)
    im.putpalette(FLATPAL)
    return im.convert('RGB').resize((cell,cell), Image.NEAREST)

def mk_pos(n, rate, t_our_end=None, n_ours=None, fast_ticks=50):
    # position (in moves) at tick t; after the OPINE run ends, a longer Sol
    # session fast-forwards its remaining moves over fast_ticks
    def pos(t):
        t = max(0, t)
        if t_our_end is None or t <= t_our_end:
            return min(t*rate, n-1)
        rem = (n-1) - (n_ours-1)
        if rem <= 0:
            return min(t*rate, n-1)
        fr = max(rate, rem/fast_ticks)
        return min((n_ours-1) + (t-t_our_end)*fr, n-1)
    return pos

def draw_panel(im, dr, x0, y0, cell, steps, levels, t, pos, tl, ours, our_n=None, fast=False):
    n = len(steps)
    p = pos(t)
    s = min(n-1, int(p))
    end = p >= n-1
    if end:
        g = steps[n-1][-1]
        s = n-1
    else:
        sub = steps[s]
        g = sub[min(len(sub)-1, int((p-s)*len(sub)))]
    lv = levels[s]
    prev = levels[min(n-1, int(pos(t-FLASH)))]
    flash = (not end) and lv > prev
    won = lv >= tl and end
    im.paste(grid_img(g, cell), (x0, y0))
    if won or flash:
        bc = GREEN if (ours or won) else (120,120,120)
        dr.rectangle([x0-3, y0-3, x0+cell+2, y0+cell+2], outline=bc, width=3)
    else:
        dr.rectangle([x0-1, y0-1, x0+cell, y0+cell], outline=(225,225,225), width=1)
    ly = y0 + cell + 6
    mv = ('\u00bb\u00bb ' if fast and not end else '') + f'move {s}'
    dr.text((x0+cell-dr.textlength(mv, font=FS), ly+1), mv, font=FS, fill=GRAY)
    if end and won:
        dr.text((x0, ly), f'WIN · {lv}/{tl} levels', font=FB, fill=GREEN)
    elif end and not ours:
        dr.text((x0, ly), f'FAIL · {lv}/{tl} levels', font=FB, fill=RED)
        if our_n and n < STOP_RATIO*our_n:
            dr.text((x0, ly+17), '(stopped, no progress)', font=FS, fill=RED)
    else:
        dr.text((x0, ly), f'{lv}/{tl} levels', font=FB if ours else F,
                fill=BLACK if ours else GRAY)

def save(frames, durations, out, w, h, shrink=None, colors=64):
    if shrink and shrink < w:
        nh = round(h*shrink/w)
        frames = [f.resize((shrink, nh), Image.LANCZOS) for f in frames]
        w, h = shrink, nh
    print(f'{len(frames)} frames, {w}x{h}')
    q = [f.quantize(colors=colors, dither=Image.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations,
              loop=0, optimize=True)
    print(out, f'{os.path.getsize(out)/1e6:.1f} MB, ~{sum(durations)/1000:.1f}s')

def build_grid(acts, out, our_ticks, step_ms, scale=3, colors=64):
    SCALE = scale; CELL = int(64*SCALE); GAP = 12; MARG = 16; GUT = 140
    COLH = 24; SUB1 = 24; SUB2 = 38; ROWGAP = 10
    W = MARG + GUT + 5*CELL + 4*GAP + MARG
    H = MARG + COLH + CELL + SUB1 + ROWGAP + CELL + SUB2 + MARG
    y1 = MARG + COLH
    y2 = y1 + CELL + SUB1 + ROWGAP
    frames, durations = [], []
    for a, games in enumerate(acts):
        data = [get(g) for g in games]
        T = min(our_ticks, max(len(o[0]) for o, _ in data))
        FAST = round(2000/step_ms)
        cols, ticks = [], T
        for o, s in data:
            n_o, n_s = len(o[0]), len(s[0])
            rate = (n_o-1)/(T-1)
            over = n_s > n_o
            end_s = (T-1) + min((n_s-n_o)/rate, FAST) if over else (n_s-1)/rate
            end_s = round(end_s)
            ticks = max(ticks, end_s+1)
            cols.append(dict(
                po=mk_pos(n_o, rate),
                ps=mk_pos(n_s, rate, t_our_end=T-1, n_ours=n_o, fast_ticks=FAST),
                over=over))
        last_act = a == len(acts)-1
        for t in range(ticks):
            im = Image.new('RGB', (W,H), (255,255,255))
            dr = ImageDraw.Draw(im)
            for row, (l1, l2, c1) in enumerate([('OPINE-World','(ours)',BLACK),
                                                ('GPT-5.6 Sol','(max effort)',GRAY)]):
                cy = (y1 if row==0 else y2) + CELL//2
                dr.text((MARG, cy-16), l1, font=FB, fill=c1)
                dr.text((MARG, cy+3), l2, font=FS, fill=GRAY)
            for k, gid in enumerate(games):
                (og, olv), (sg, slv) = data[k]
                x0 = MARG + GUT + k*(CELL+GAP)
                dr.text((x0, MARG+2), gid, font=FM, fill=BLACK)
                c = cols[k]
                draw_panel(im, dr, x0, y1, CELL, og, olv, t, c['po'], TL[gid], True)
                draw_panel(im, dr, x0, y2, CELL, sg, slv, t, c['ps'], TL[gid], False,
                           our_n=len(og), fast=c['over'] and t > T-1)
            frames.append(im)
            durations.append((2600 if last_act else 1400) if t == ticks-1 else step_ms)
        CACHE.clear()
    save(frames, durations, out, W, H, colors=colors)

def build_1v1(games, out, seg_ticks=100, step_ms=40):
    SCALE = 4; CELL = 64*SCALE; GAP = 18; MARG = 18
    COLH = 26; LBL = 22; SUB = 26
    W = MARG + CELL + GAP + CELL + MARG
    H = MARG + COLH + LBL + CELL + SUB + MARG
    y0 = MARG + COLH + LBL
    frames, durations = [], []
    FAST = round(2000/step_ms)
    for gi, gid in enumerate(games):
        (og, olv), (sg, slv) = get(gid)
        n_o, n_s = len(og), len(sg)
        maxn = max(n_o, n_s)
        T = min(seg_ticks, maxn)
        rate = (maxn-1)/(T-1)
        t_our_end = (n_o-1)/rate
        over = n_s > n_o
        ticks = round(t_our_end + min((n_s-n_o)/rate, FAST))+1 if over else T
        po = mk_pos(n_o, rate)
        ps = mk_pos(n_s, rate, t_our_end=t_our_end, n_ours=n_o, fast_ticks=FAST)
        for t in range(ticks):
            im = Image.new('RGB', (W,H), (255,255,255))
            dr = ImageDraw.Draw(im)
            dr.text((MARG, MARG), gid, font=FM, fill=BLACK)
            dr.text((MARG, MARG+COLH+2), 'OPINE-World (ours)', font=FB, fill=BLACK)
            dr.text((MARG+CELL+GAP, MARG+COLH+2), 'GPT-5.6 Sol (max effort)', font=F, fill=GRAY)
            draw_panel(im, dr, MARG, y0, CELL, og, olv, t, po, TL[gid], True)
            draw_panel(im, dr, MARG+CELL+GAP, y0, CELL, sg, slv, t, ps, TL[gid], False,
                       our_n=n_o, fast=over and t > t_our_end)
            frames.append(im)
            last = gi == len(games)-1 and t == ticks-1
            durations.append((2400 if last else 1100) if t == ticks-1 else step_ms)
        CACHE.clear()
    save(frames, durations, out, W, H)

if __name__ == '__main__':
    print('v5...')
    build_grid(V5, os.path.join(SC, 'opine_world_vs_sol.gif'), our_ticks=426, step_ms=40)
    print('v20...')
    build_grid(V20, os.path.join(SC, 'opine_world_vs_sol_20.gif'), our_ticks=120, step_ms=50, scale=2.5)
    print('1v1...')
    build_1v1(V1V1, os.path.join(SC, 'opine_world_vs_sol_1v1.gif'))
