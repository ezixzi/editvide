"""World-space hand placement and pose checks against work surfaces."""

import numpy as np

from cast import arm_pose,basis,hand_vertices,posed_hand,wrist_control
from studio import V,norm


def directed(hip,r,side,target,forward,roll,*,bend=None,curl=.45,index=0.):
    local=r.T@norm(forward)
    pos=r.T@(np.asarray(target)-hip)
    pole=r.T@bend if bend is not None else None
    return wrist_control(*pos,*local,roll,curl,bend=pole,index=index)


def hand_at_height(hip,r,side,x,y,top,forward,roll,curl,index=0.):
    local=r.T@norm(forward)
    orient=r@basis(local,roll)
    v=hand_vertices(side,curl,index)[:,:3]@orient.T
    return directed(hip,r,side,V(x,y,top+.004-np.min(v[:,2])),forward,roll,curl=curl,index=index)


def table_check(film,role,hip,r,controls,bounds):
    x0,x1,y0,y1,height=bounds
    for side,ctl in zip((-1,1),controls):
        a,e,w,_=arm_pose(hip,r,side,ctl)
        film.smooth(f'Kecepatan siku {role} {side}',e,.081)
        vertices=posed_hand(hip,r,side,ctl)
        mask=((vertices[:,0]>x0)&(vertices[:,0]<x1)&(vertices[:,1]>y0)&(vertices[:,1]<y1))
        if np.any(mask):
            film.clearance(f'Telapak {role} terhadap meja',np.min(vertices[mask,2])-height)
        for a0,b0,rad in ((a,e,.047),(e,w,.034)):
            nodes=np.linspace(a0,b0,13)
            mask=((nodes[:,0]>x0+.009)&(nodes[:,0]<x1-.009)&(nodes[:,1]>y0+.009)&(nodes[:,1]<y1-.009))
            if np.any(mask):
                film.clearance(f'Lengan {role} terhadap meja',np.min(nodes[mask,2])-height-rad,minimum=-.005)
