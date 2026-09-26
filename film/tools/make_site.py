"""Cut the finished film into per-scene chapter files for the web page, grab posters,
and write build/site/index.html.

    python3 tools/make_site.py [--vbr 1850k]
"""
import argparse
import json
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build')
SITE = os.path.join(BUILD, 'site')

CHAPTERS = {
    's01': ('开场', 'Opening'),
    's02': ('阿灯的夜晚', "Deng's evening"),
    's03': ('流星', 'The falling star'),
    's04': ('相遇', 'The meeting'),
    's05': ('想办法', 'Trying'),
    's06': ('灯塔的光', 'The beam'),
    's07': ('回答', 'The answer'),
    's08': ('尾声', 'Epilogue'),
}
POSTER_AT = {'s01': 5.5, 's02': 10.2, 's03': 16.0, 's04': 27.0, 's05': 22.3, 's06': 35.5, 's07': 31.0, 's08': 9.4}


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(BUILD, 'film.mp4'))
    ap.add_argument('--vbr', default='1850k')
    ap.add_argument('--skip-video', action='store_true')
    ap.add_argument('--only', default='', help='comma list of scene ids to (re)encode; others are kept')
    a = ap.parse_args()
    os.makedirs(SITE, exist_ok=True)
    tl = json.load(open(os.path.join(BUILD, 'timeline.json')))
    chapters = []
    for sc in tl['scenes']:
        sid, s0, s1 = sc['id'], sc['start'], sc['end']
        out = os.path.join(SITE, f'{sid}.mp4')
        poster = os.path.join(SITE, f'{sid}.jpg')
        if not a.skip_video and (not a.only or sid in a.only.split(',')):
            common = ['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{s0:.3f}', '-t', f'{s1 - s0:.3f}', '-i', a.src]
            venc = ['-c:v', 'libx264', '-preset', 'slow', '-b:v', a.vbr, '-maxrate', '3500k', '-bufsize', '6000k',
                    '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-g', '48']
            run(common + venc + ['-pass', '1', '-passlogfile', os.path.join(SITE, 'x264'), '-an', '-f', 'mp4', '/dev/null'])
            run(common + venc + ['-pass', '2', '-passlogfile', os.path.join(SITE, 'x264'),
                                 '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', out])
            run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{s0 + POSTER_AT[sid]:.3f}', '-i', a.src,
                 '-frames:v', '1', '-vf', 'scale=960:-2', '-q:v', '4', poster])
        zh, en = CHAPTERS[sid]
        chapters.append(dict(id=sid, zh=zh, en=en, start=s0, dur=round(s1 - s0, 3), src=f'{sid}.mp4',
                             poster=f'{sid}.jpg', mb=round(os.path.getsize(out) / 1e6, 2)))
        print(sid, chapters[-1]['mb'], 'MB')
    json.dump(chapters, open(os.path.join(SITE, 'chapters.json'), 'w'), ensure_ascii=False, indent=1)
    print('total', round(sum(c['mb'] for c in chapters), 1), 'MB')


if __name__ == '__main__':
    main()
