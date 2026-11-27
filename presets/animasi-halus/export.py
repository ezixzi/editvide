"""Join audited native shots, attach the original-VO mix and verify the MP4."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from render import FPS,PROFILE,VIDEO


def call(cmd):
    result=subprocess.run(cmd,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-3500:])
    return result


def probe(path):
    return json.loads(call(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]).stdout)


def render_shape(stream):
    return (stream['codec_name'],stream['width'],stream['height'],stream['avg_frame_rate'],
            stream['pix_fmt'],stream['color_space'])


def join(episode,plates,audio,output,work):
    timing=json.loads((episode/'timeline.json').read_text())
    wanted=('h264',VIDEO['width'],VIDEO['height'],f'{FPS}/1',VIDEO['pixel_format'],VIDEO['color'])
    concat=['ffconcat version 1.0']
    checks={}
    for i,shot in enumerate(timing['shots']):
        plate=plates/f'{i:02d}-{shot["id"]}.mp4'
        review=json.loads((plates/f'{i:02d}-review.json').read_text())
        if review['errors'] or review['shot']!=shot['id']:
            raise ValueError((shot['id'],review))
        for name,record in review['checks'].items():
            checks[name]=checks.get(name,0)+record['samples']
        v=probe(plate)['streams'][0]
        n=shot['end']-shot['start']
        assert render_shape(v)==wanted and int(v['nb_frames'])==n and review['frames']==n,plate
        source=str(plate.resolve()).replace("'","'\\''")
        concat.extend((f"file '{source}'",f'duration {n/FPS:.12f}'))
    list_path=work/'timeline.ffconcat'
    list_path.write_text('\n'.join(concat)+'\n')
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.export-') as directory:
        staged=Path(directory)/'joined.mp4'
        call(['ffmpeg','-v','error','-xerror','-nostdin','-y','-threads','3','-f','concat','-safe','0',
              '-i',str(list_path),'-i',str(audio),'-map','0:v','-map','1:a',
              '-c:v','copy','-c:a','aac','-b:a','224k','-video_track_timescale','30000',
              '-bsf:v','h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1',
              '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
              '-movflags','+faststart','-metadata','title='+timing['title'],str(staged)])
        measurements=verify(staged,timing,wanted)
        staged.replace(output)
    with output.open('rb') as fp:
        checksum=hashlib.file_digest(fp,'sha256').hexdigest()
    result={**measurements,'shot_count':len(timing['shots']),'checks':checks,
            'video_sha256':checksum,
            'size_bytes':output.stat().st_size}
    (work/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def verify(media,timing,shape):
    v,a=probe(media)['streams']
    frames=timing['frames']
    duration=frames/FPS
    assert render_shape(v)==shape and int(v['nb_frames'])==frames
    assert all(v[k]==VIDEO['color'] for k in ('color_primaries','color_transfer'))
    assert (a['codec_name'],a['sample_rate'],a['channels'])==('aac','48000',2)
    assert float(a['start_time'])==float(v['start_time'])==0
    assert abs(float(v['duration'])-duration)<.000002
    assert abs(float(a['duration'])-duration)<.012
    pts=call(['ffprobe','-v','error','-select_streams','v:0','-show_entries','packet=pts',
              '-of','csv=p=0',str(media)]).stdout
    assert sorted(map(int,pts.split()))==list(range(0,frames*(30000//FPS),30000//FPS))
    log=call(['ffmpeg','-v','info','-nostdin','-hide_banner','-xerror','-threads','3','-i',str(media),
              '-vf','blackdetect=d=0.015:pix_th=0.054:pic_th=0.985,freezedetect=n=-60dB:d=0.50',
              '-af','ebur128=peak=true:framelog=verbose','-f','null','-']).stderr
    assert 'black_start:' not in log and 'freeze_start:' not in log
    summary=log.rsplit('Summary:',1)[-1]
    lufs=float(re.search(r'I:\s*(-?\d+\.\d+)\s*LUFS',summary)[1])
    peak=float(re.search(r'Peak:\s*(-?\d+\.\d+)\s*dBFS',summary)[1])
    assert abs(lufs-PROFILE['audio']['loudness_lufs'])<=1.0
    assert peak<=PROFILE['audio']['true_peak_limit_dbtp']
    return {'frames':frames,'duration':duration,'integrated_lufs':lufs,'true_peak_dbtp':peak}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--plates',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    out=join(a.episode,a.plates,a.audio,a.out,a.work)
    print(json.dumps({key:out[key] for key in ('frames','duration','integrated_lufs','true_peak_dbtp','size_bytes')}),flush=True)


if __name__=='__main__':
    main()
