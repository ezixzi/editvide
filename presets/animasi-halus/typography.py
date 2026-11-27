"""Portrait typography, word reveals and optical finishing for Animasi Halus."""

from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont

from studio import E


@lru_cache(maxsize=120)
def font(size):
    return ImageFont.truetype(str(Path(__file__).with_name('Caption.ttf')),max(8,size))


class Titles:
    def __init__(self,width,height,profile,cues,keywords=()):
        self.s=width/1080
        self.p=profile
        self.cues=cues
        self.starts=[x['start'] for x in cues]
        self.keywords={w.casefold() for w in keywords}
        self.cream=ImageColor.getrgb(profile['primary'])
        self.accent=ImageColor.getrgb(profile['accent'])
        y,x=np.mgrid[0:height,0:width].astype(np.float32)
        radial=((x-width*.46)/(.85*width))**2+((y-height*.43)/(.76*height))**2
        grade=1-.22*np.clip(radial-.18,0,1)
        self.veil=Image.fromarray((grade*255).astype(np.uint8)).convert('RGB')

    def text(self,draw,text,x,y,size,opacity=1.,*,accent=False,anchor='la',stroke=2):
        color=self.accent if accent else self.cream
        opacity=min(1.,max(0.,opacity))
        draw.text((round(x*self.s),round(y*self.s)),text,anchor=anchor,
                  font=font(round(size*self.s)),fill=(*color,round(255*opacity)),
                  stroke_width=max(1,round(stroke*self.s)),stroke_fill=(16,22,28,round(225*opacity)))

    def words(self,draw,t,baseline):
        i=bisect_right(self.starts,t)-1
        if i<0:
            return
        cue=self.cues[i]
        if t>=cue['end']:
            return
        p=self.p
        baseline=baseline if baseline is not None else p['baseline_y']
        base=font(round(self.s*p['font_px']))
        max_width=p['right']-p['left']-14
        phrases=cue['lines']
        assert 1<=len(phrases)<=2
        for j,line in enumerate(phrases):
            widths=[base.getlength(w['text'])/self.s for w in line]
            span=sum(widths)+12*max(0,len(line)-1)
            scale=min(1.,max_width/max(span,1.))
            x=p['center_x']-span*scale/2
            y=baseline+j*p['line_px']-(len(phrases)-1)*p['line_px']
            for w,length in zip(line,widths):
                fade=E(t-w['onset']+.017,0,p['word_fade_in_s'])*E(cue['end']-t,0,p['phrase_fade_out_s'])
                token=w['text'].casefold().strip('!?,.\u2026')
                size=p['font_px']*scale*(.92+.08*fade)
                if fade>.008:
                    self.text(draw,w['text'],x+length*scale/2,y+6*(1.-fade),size,fade,
                              accent=token in self.keywords,anchor='mm',stroke=2.8)
                x+=(length+12)*scale

    def frame(self,data,t,labels,baseline,mask,plain=False):
        image=Image.fromarray(data)
        if mask is not None:
            matte=Image.fromarray(mask).resize(image.size,Image.Resampling.BILINEAR)
            matte=matte.filter(ImageFilter.GaussianBlur(self.s*2.6))
            blurred=image.filter(ImageFilter.GaussianBlur(self.s*11.4))
            image=Image.composite(blurred,image,matte)
        image=ImageChops.multiply(image,self.veil)
        if plain:
            return image
        overlay=Image.new('RGBA',image.size)
        draw=ImageDraw.Draw(overlay)
        for line in labels:
            self.text(draw,*line[:5],accent=line[5])
        self.words(draw,t,baseline)
        return Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB')
