"""Mix original narration with an original instrumental bed and timed procedural Foley."""

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy import signal
from scipy.io import wavfile

PROFILE=json.loads(Path(__file__).with_name('profile.json').read_text())
RATE=PROFILE['audio']['sample_rate']
TAU=2*math.pi


def ramps(time,start,end):
    x=np.clip((time-start)/(end-start),0.,1.)
    return x*x*(3-2*x)


def times(seconds):
    return np.arange(round(seconds*RATE),dtype=np.float32)/RATE


def noise(rng,n,low,high):
    sos=signal.butter(2,(low,high),fs=RATE,btype='bandpass',output='sos')
    part=signal.sosfilt(sos,rng.normal(size=n))
    return (part/max(.0001,np.std(part))).astype(np.float32)


def add(track,y,when,volume,pan=0.):
    at=round(when*RATE)
    count=min(len(y),len(track)-at)
    if count<=0:
        return
    angle=(pan+1)*math.pi/4
    track[at:at+count,0]+=y[:count]*volume*math.cos(angle)
    track[at:at+count,1]+=y[:count]*volume*math.sin(angle)


def rms(y):
    return float(np.sqrt(np.mean(np.square(y,dtype=np.float64))))


def midi(note):
    return 440.*2**((note-69)/12)


def tone(t,n,phase=.0):
    f=midi(n)
    phi=TAU*f*t+.065*np.sin(TAU*4.4*t)+phase
    return .78*np.sin(phi)+.15*np.sin(2*phi+.16)+.07*np.sin(3*phi+.20)


def arrange(count):
    music=np.zeros((count,2),np.float32)
    progression=[(45,52,57,60),(43,50,55,59),(48,55,60,64),(47,54,59,62),
                 (50,57,62,65),(45,52,57,60),(53,60,64,69),(48,55,64,67)]
    for i,at in enumerate(np.arange(0,count/RATE,3.86)):
        notes=progression[i%len(progression)]
        t=times(4.43)
        env=ramps(t,0,.51)*(1-ramps(t,3.66,4.43))
        for j,n in enumerate(notes[:3]):
            swell=.83+.17*np.sin(t*2.37+j*1.27)**2
            add(music,tone(t,n,j*.72)*env*swell,float(at),(.152,.096,.077)[j],(-.48,.25,.57)[j])
        cell=times(.92)
        for j,start in enumerate((.32,1.0,1.79,2.63,3.28)):
            attack=ramps(cell,0,.026)*np.exp(-cell*5.22)*(1-ramps(cell,.73,.92))
            note=notes[(0,2,1,3,2)[j]]
            pluck=(tone(cell,note)+.09*tone(cell,note+12,.27))*attack
            add(music,pluck,float(at+start),.070 if j%2==0 else .042,(-1)**j*.40)
    return music


def effect(rng,name,seconds):
    t=times(seconds)
    envelope=ramps(t,0,min(.08,seconds*.16))*(1-ramps(t,seconds*.73,seconds))
    sin=lambda f: np.sin(TAU*f*t)
    modal=lambda f,d: sin(f)*np.exp(-d*t)
    if name in ('wind','breeze','market','room','party','water'):
        low,high={'wind':(75,381),'breeze':(122,510),'market':(145,625),'room':(93,342),
                  'party':(174,551),'water':(421,1332)}[name]
        pulse=.76+.19*np.sin(t*(4.6 if name=='market' else 1.9))+.05*np.sin(t*7.2)
        y=noise(rng,len(t),low,high)*pulse+.19*noise(rng,len(t),low*1.30,high*1.9)
        envelope=ramps(t,0,min(.56,seconds*.19))*(1-ramps(t,seconds*.75,seconds))
    elif name in ('paper','quill','fabric','leaf'):
        low,high={'paper':(282,2800),'quill':(597,3280),'fabric':(126,964),'leaf':(235,1860)}[name]
        flutter=.60+.40*np.sin(TAU*(17*t+7*t*t))**2
        y=noise(rng,len(t),low,high)*flutter*np.sin(np.pi*t/seconds)**1.2
    elif name=='bird':
        phase=TAU*(1800*t+1230*t*t/seconds)+.25*np.sin(t*TAU*17)
        y=(.70*np.sin(phase)+.24*np.sin(phase*.554+.22))*(.55+.45*np.cos(TAU*11*t)**2)
    elif name in ('hoof','step','wheel','tick','tap'):
        hz={'hoof':157,'step':103,'wheel':203,'tick':738,'tap':256}[name]
        high={'hoof':1083,'step':552,'wheel':881,'tick':2720,'tap':1441}[name]
        y=.53*modal(hz,27)+.17*modal(hz*2.49,40)+.39*noise(rng,len(t),104,high)*np.exp(-29*t)
        envelope=ramps(t,0,.0032)*(1-ramps(t,seconds*.58,seconds))
    elif name in ('chime','pluck','toast','shimmer','notify'):
        hz={'chime':610,'pluck':461,'toast':566,'shimmer':767,'notify':725}[name]
        y=.65*modal(hz,7.4)+.23*modal(hz*1.503,12)+.085*modal(hz*2.402,14)
    elif name in ('pop','impact'):
        z=TAU*(116*t+134*.065*(1-np.exp(-t/.065)))
        dip=np.sin(z)*np.exp(-t*(17 if name=='pop' else 7.5))
        y=.78*dip+.28*noise(rng,len(t),113,1464)*np.exp(-t*19)
        envelope=ramps(t,0,.004)
    else:
        center=94 if name=='risewind' else 168
        y=(.29*np.sin(TAU*(center*t+40*t*t/seconds))+.65*noise(rng,len(t),90,830))*np.sin(np.pi*t/seconds)**1.8
    y=y*envelope
    return (y/max(.0001,np.max(np.abs(y)))).astype(np.float32)


def measure(path):
    cmd=['ffmpeg','-nostdin','-hide_banner','-i',str(path),'-af',
         'loudnorm=I=-14:TP=-1.65:LRA=14:print_format=json','-f','null','-']
    log=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stderr
    return json.loads(log[log.rindex('{'):log.rindex('}')+1])


def master(voice,timing,out):
    audio=PROFILE['audio']
    cmd=['ffmpeg','-v','error','-nostdin','-i',str(voice),'-ar',str(RATE),'-ac','1',
         '-f','f32le','-c:a','pcm_f32le','-']
    narration=np.frombuffer(subprocess.check_output(cmd),dtype='<f4')
    assert abs(len(narration)/RATE-timing['source_duration'])<1/RATE+.000001
    fps=PROFILE['video']['fps']
    count=timing['frames']*RATE//fps
    if not 0<=count-len(narration)<RATE/fps+1:
        raise ValueError('Render must contain the full, untrimmed narration')
    vo=np.pad(narration,(0,count-len(narration)))
    block=480
    aligned=np.pad(vo,(0,-count%block)).reshape(-1,block)
    power=np.sqrt(np.mean(aligned**2,axis=1))
    on=ramps(power,.0065,.047)
    envelope=np.zeros_like(on)
    smooth=0.
    for i in range(len(envelope)):
        target=on[min(i+2,len(on)-1)]
        smooth+=(target-smooth)*(.63 if target>smooth else .094)
        envelope[i]=smooth
    gate=np.interp(np.arange(count),np.arange(len(on))*block,envelope).astype(np.float32)
    voice_mask=np.repeat(power>.020,block)[:count]
    if not np.any(voice_mask):
        raise ValueError('Audio has no audible narration')
    vr=rms(vo[voice_mask])
    music=arrange(count)
    music*=10**(-4.8*gate[:,None]/20)
    music*=vr*10**(-audio['music_below_voice_db']/20)/rms(music[voice_mask])
    effects=np.zeros((count,2),np.float32)
    rng=np.random.default_rng(34107)
    for at,name,seconds,amount,pan in timing.get('effects',()):
        if not 0<=at<count/RATE or seconds<=0 or not -1<=pan<=1:
            raise ValueError(f'Invalid Foley event: {name} at {at}')
        add(effects,effect(rng,name,seconds),at,amount,pan)
    effects*=10**(-4.3*gate[:,None]/20)
    effect_rms=rms(effects[voice_mask])
    fx_target=vr*10**(-audio['sfx_below_voice_db']/20)
    if effect_rms>0:
        effects*=min(fx_target/effect_rms,1.5)
    end=times(count/RATE)
    fade=(1-ramps(end,count/RATE-.45,count/RATE))[:,None]
    music*=fade
    effects*=fade
    premix=np.column_stack((vo,vo))+music+effects
    before=out/'mix-unmastered.wav'
    after=out/'mix-master.wav'
    wavfile.write(before,RATE,premix)
    gain=audio['loudness_lufs']-float(measure(before)['input_i'])
    amplified=premix*10**(gain/20)
    ceiling=10**(-1.80/20)
    peaks=signal.resample_poly(amplified,4,1,axis=0)
    peaks=np.max(np.abs(peaks.reshape(count,4,2)),axis=(1,2))
    limiter=np.ones(count,np.float32)
    for peak in np.flatnonzero(peaks>ceiling):
        lo=max(0,peak-193)
        hi=min(count,peak+1585)
        t=np.arange(lo,hi)
        pulse=ramps(t,peak-193,peak)*(1-ramps(t,peak,peak+1585))
        reduction=1-(1-ceiling/peaks[peak])*pulse
        limiter[lo:hi]=np.minimum(limiter[lo:hi],reduction)
    result=amplified*limiter[:,None]
    wavfile.write(after,RATE,result)
    fx=rms(effects[voice_mask])
    report={
        'voice_input_samples_at_48k':len(narration),'duration':len(narration)/RATE,
        'end_padding_samples':count-len(narration),'music_below_voice_db':round(20*math.log10(vr/rms(music[voice_mask])),2),
        'sfx_below_voice_db':round(20*math.log10(vr/fx),2) if fx else None,
        'voice_dry_correlation':round(float(np.corrcoef(vo,result.mean(axis=1))[0,1]),6),
        'master':measure(after)
    }
    (out/'mix-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voice',type=Path,required=True)
    parser.add_argument('--timeline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    master(args.voice,json.loads(args.timeline.read_text()),args.out)


if __name__=='__main__':
    main()
