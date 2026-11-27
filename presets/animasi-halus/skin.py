"""Smooth deformed hands and sewn sleeves used by the close-camera rig."""

from functools import lru_cache
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
AZIMUTH=np.arange(16,dtype=np.float32)*(2*np.pi/16)
CS=np.cos(AZIMUTH)[None,:,None]
SN=np.sin(AZIMUTH)[None,:,None]


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-7)


@lru_cache(maxsize=2)
def template(side):
    tag='R' if side>0 else 'L'
    with np.load(HERE/'hands.npz') as d:
        return d['points_'+tag],d['normals_'+tag],d['triangles_'+tag].reshape(-1),d['weights_'+tag]


def rotation_between(a,b):
    u=unit(a)
    v=unit(b)
    x,y,z=np.cross(u,v)
    hat=np.array(((0,-z,y),(z,0,-x),(-y,x,0)),np.float32)
    return np.eye(3,dtype=np.float32)+hat+(hat@hat)/max(.02,1+np.dot(u,v))


def deformed_hand(side,rest,pose):
    positions,normals,triangles,weights=template(side)
    rows=[np.eye(3,dtype=np.float32)]
    translations=[np.zeros(3,np.float32)]
    for a,b in zip(rest,pose):
        for start,end in ((0,1),(1,2)):
            r=rotation_between(a[end]-a[start],b[end]-b[start])
            rows.append(r)
            translations.append(b[start]-r@a[start])
    m=(weights@np.asarray(rows).reshape(-1,9)).reshape(-1,3,3)
    q=np.einsum('kij,kj->ki',m,positions)+weights@translations
    n=unit(np.einsum('kij,kj->ki',m,normals))
    return np.column_stack((q,n))[triangles]


@lru_cache(maxsize=6)
def joins(rings):
    n=len(AZIMUTH)
    triangles=[]
    for j in range(rings-1):
        for i in range(n):
            a=j*n+i
            b=j*n+(i+1)%n
            triangles.extend((a,b,b+n,a,b+n,a+n))
    return np.asarray(triangles)


def tailored_limb(top,knee,end,reference,width,depth,style):
    v=unit(knee-top)
    w=unit(end-knee)
    k0=knee-.078*v
    k1=knee+.068*w
    u=np.linspace(0,1,8,dtype=np.float32)[:,None]
    path=np.concatenate((np.linspace(top-.034*v,k0,6),
                         (1-u)**2*k0+2*u*(1-u)*knee+u*u*k1,
                         np.linspace(k1,end,8)[1:]),axis=0)
    path=np.delete(path,5,axis=0)
    speed=unit(np.gradient(path,axis=0))
    lateral=[]
    a=reference.copy()
    for tangent in speed:
        projected=a-tangent*np.dot(a,tangent)
        if np.linalg.norm(projected)<1e-5:
            axis=np.eye(3)[np.argmin(np.abs(tangent))]
            projected=axis-tangent*np.dot(axis,tangent)
        a=unit(projected)
        lateral.append(a)
    lateral=np.asarray(lateral)
    over=unit(np.cross(speed,lateral))
    progress=np.linspace(0,1,len(path),dtype=np.float32)
    radius=np.interp(progress,(0,.13,.35,.60,.78,.91,.95,.985,1.),
                     (1.04,1.10,1.017,.907,.796,.704,.668,.697,.678)).astype(np.float32)
    if style=='trousers':
        radius=np.interp(progress,(0,.22,.47,.64,.83,1.),(1.4,1.26,.94,.83,.76,.68)).astype(np.float32)
    if style=='arm':
        radius=np.interp(progress,(0,.22,.46,.61,.79,1.),(1.04,1.0,.87,.79,.61,.424)).astype(np.float32)
    weave=(1.+.032*np.cos(progress[:,None]*31+AZIMUTH[None,:]*3)
             +.029*np.sin(progress[:,None]*18+AZIMUTH[None,:]*5))[:,:,None]
    r1=lateral[:,None,:]*CS
    r2=over[:,None,:]*SN
    wall=(r1*width+r2*depth)*.5
    points=path[:,None,:]+wall*radius[:,None,None]*weave
    normals=unit(r1*depth+r2*width)
    attributes=np.concatenate((points,normals),axis=2).reshape(-1,6)
    return np.ascontiguousarray(attributes[joins(len(path))],np.float32)
