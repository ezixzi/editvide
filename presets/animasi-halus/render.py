"""Render an Animasi Halus episode from its own scene module and timed VO assets."""

import argparse
from bisect import bisect_right
import importlib.util
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np

import cast
from stage_gl import view_matrix
from studio import Studio, group, norm
from typography import Titles


PRESET=Path(__file__).resolve().parent
PROFILE=json.loads((PRESET/'profile.json').read_text())
VIDEO=PROFILE['video']
FPS=VIDEO['fps']


class Film:
    def __init__(self,episode,width,height):
        self.w,self.h=width,height
        cfg=json.loads((episode/'timeline.json').read_text())
        self.shots=cfg['shots']
        self.starts=[s['start'] for s in self.shots]
        cue_data=json.loads((episode/'captions.json').read_text())
        validate_timing(self.shots,cue_data,cfg['frames'])
        spec=importlib.util.spec_from_file_location('episode_scene',episode/'scene.py')
        self.scene=importlib.util.module_from_spec(spec)
        sys.path.insert(0,str(episode))
        spec.loader.exec_module(self.scene)
        atlas,rects=self.scene.atlas()
        self.stage=Studio(width,height,atlas,rects)
        cast.install(self.stage)
        self.scene.install(self.stage)
        self.titles=Titles(width,height,PROFILE['caption'],cue_data,cfg.get('keywords',()))
        self.corner=np.asarray(list(product((-1,1),repeat=3)))
        self.reset()

    def reset(self):
        self.checks={}
        self.errors=[]
        self.previous={}

    def world(self,name,maker):
        if name not in self.stage.groups:
            self.stage.set(name,maker())

    def clearance(self,label,value,minimum=0.):
        gap=float(value)-minimum
        if not np.isfinite(gap):
            raise ValueError((label,value))
        stats=self.checks.setdefault(label,{'samples':0,'margin':1e6,'frame':None})
        stats['samples']+=1
        if gap<stats['margin']:
            stats.update(margin=gap,frame=self.f)
        if gap < -1e-6:
            self.errors.append({'frame':self.f,'label':label,'margin':round(gap,5)})

    def smooth(self,label,at,limit):
        old=self.previous.get(label)
        if old is not None and old[0]==self.f-1:
            self.clearance(label,limit-np.linalg.norm(at-old[1]))
        self.previous[label]=(self.f,at.copy())

    def track(self,name,position,half_extent=(.199,.198,.234)):
        self.anchors.append((name,np.asarray(position),np.asarray(half_extent)))

    def frame(self,f,review=False,plain=False):
        self.f=f
        self.anchors=[]
        shot=self.shots[bisect_right(self.starts,f)-1]
        t=f/FPS
        p=(f-shot['start'])/(shot['end']-shot['start'])
        g,shadow,air=group(detail=shot.get('detail',True) and not review),group(),group()
        world,cam,light,focus=self.scene.draw(self,shot,g,shadow,air,t,p)
        eye,aim,fov=cam
        for name,center,extent in self.anchors:
            matrix,_=view_matrix(eye,aim,fov,self.w/self.h)
            vertices=np.column_stack((center+extent*self.corner,np.ones(len(self.corner))))
            corners=vertices@matrix.T
            if np.min(corners[:,3])<=0:
                self.clearance(f'Framing {name}',-1.)
                continue
            projected=corners[:,:2]/corners[:,3,None]
            self.clearance(f'Framing {name}',1.016-np.max(np.abs(projected)))
        if review:
            return None
        self.stage.render_view(eye,aim,fov,light,self.stage.groups[world],
                               self.stage.group(g,'subjects'),self.stage.group(shadow,'contact'),
                               self.stage.group(air,'air'),t)
        mask=self.stage.focus_mask(eye,aim,focus) if focus is not None and PROFILE['render']['depth_focus'] else None
        labels=self.scene.labels(shot,t,p)
        return self.titles.frame(self.stage.rgb(),t,labels,shot.get('caption_y'),mask,plain)


def validate_timing(shots,cues,frames):
    edge=0
    ids=set()
    for shot in shots:
        if shot['start']!=edge or shot['start']>=shot['end'] or shot['id'] in ids:
            raise ValueError(f"Invalid shot timeline: {shot['id']}")
        ids.add(shot['id'])
        edge=shot['end']
    if edge!=frames:
        raise ValueError(f'Shot frame count {edge} differs from final duration {frames}')
    previous=0.
    for cue in cues:
        if not previous<=cue['start']<cue['end']<=frames/FPS+.001:
            raise ValueError(f'Invalid caption range: {cue}')
        previous=cue['end']
        onset=cue['start']
        for line in cue['lines']:
            for word in line:
                if not onset<=word['onset']<cue['end'] or not word['text'].strip():
                    raise ValueError(f'Invalid caption onset: {word}')
                onset=word['onset']


def write_shot(film,i,out):
    shot=film.shots[i]
    name=shot['id']
    begin=monotonic()
    temporary=out/f'{i:02d}.{os.getpid()}.render.mp4'
    dest=out/f'{i:02d}-{name}.mp4'
    command=['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{film.w}x{film.h}','-r',str(FPS),'-i','pipe:0','-an',
             '-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p',
             '-c:v','libx264','-crf','17','-preset','veryfast','-threads','2',
             '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
             '-video_track_timescale','30000',str(temporary)]
    film.reset()
    try:
        with subprocess.Popen(command,stdin=subprocess.PIPE) as enc:
            for f in range(shot['start'],shot['end']):
                enc.stdin.write(film.frame(f).tobytes())
            enc.stdin.close()
            if enc.wait():
                raise RuntimeError(f'Cannot encode shot {name}')
        if film.errors:
            raise ValueError((name,film.errors[:8]))
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    report={'shot':name,'frames':shot['end']-shot['start'],'checks':film.checks,'errors':film.errors}
    (out/f'{i:02d}-review.json').write_text(json.dumps(report,indent=2))
    print(f'{i:02d} {name} · {monotonic()-begin:.1f}s · {len(film.errors)} flags',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--shots',type=int,nargs='+')
    parser.add_argument('--stills',type=int,nargs='+')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    preview=PROFILE['render']
    w,h=(preview['review_w'],preview['review_h']) if args.preview else (VIDEO['width'],VIDEO['height'])
    film=Film(args.episode.resolve(),w,h)
    if args.stills:
        for f in args.stills:
            film.frame(f).save(args.out/f'{f:04d}.png')
        if film.errors:
            print(json.dumps(film.errors),flush=True)
        return
    errors=[]
    for i in args.shots or range(len(film.shots)):
        if args.audit:
            film.reset()
            shot=film.shots[i]
            for f in range(shot['start'],shot['end']):
                film.frame(f,review=True)
            errors.extend(film.errors)
            print(json.dumps({'shot':shot['id'],'checks':film.checks,'errors':film.errors[:6]}),flush=True)
        else:
            write_shot(film,i,args.out)
    if errors:
        raise SystemExit(f'{len(errors)} QA flags in reviewed shot trajectories')


if __name__=='__main__':
    main()
