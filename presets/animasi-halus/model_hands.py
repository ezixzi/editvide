"""Sculpt and bind five-digit hands in Blender, then export their skin weights."""

from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from cast import fingers


def oval(name,where,dims):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,location=where)
    part=bpy.context.object
    part.name=name
    part.scale=tuple(d*.5 for d in dims)
    return part


def joint(start,end,width,parts):
    vec=end-start
    middle=(start+end)*.5
    segment=oval('digit capsule',middle,(width,width,np.linalg.norm(vec)+width*.88))
    segment.rotation_euler=Vector(vec).to_track_quat('Z','Y').to_euler()
    parts.append(segment)


def bind_weights(vertices,digits):
    v=np.asarray(vertices)
    n=len(v)
    weights=np.zeros((n,11),np.float32)
    # Assign each vertex to the closest digit centerline, then blend into the palm
    # and across the knuckle. All baselines are in the same coordinates as the film.
    best=np.full(n,np.inf)
    winner=np.zeros(n,np.int64)
    fraction=np.zeros(n)
    for j,(a,b,c,d) in enumerate(digits):
        ray=c-a
        s=((v-a)@ray)/np.dot(ray,ray)
        point=a+np.clip(s,0,1)[:,None]*ray
        gap=np.linalg.norm(v-point,axis=1)-.23*d
        take=gap<best
        best[take]=gap[take]
        winner[take]=j
        fraction[take]=s[take]
    base=np.clip((fraction+.086)/.375,0,1)
    base=base*base*(3-2*base)
    hinge=np.clip((fraction-.392)/.272,0,1)
    hinge=hinge*hinge*(3-2*hinge)
    weights[:,0]=1-base
    weights[np.arange(n),1+2*winner]=base*(1-hinge)
    weights[np.arange(n),2+2*winner]=base*hinge
    return weights


def make_hand(side):
    digits=list(fingers(side,0.,0.))
    parts=[oval('palm',Vector((0,.027,-.003)),(.068,.081,.030)),
           oval('metacarpals',Vector((0,.048,-.001)),(.067,.028,.025)),
           oval('wrist',Vector((0,-.004,-.004)),(.044,.047,.027)),
           oval('thenar',Vector((-side*.021,.020,-.006)),(.034,.046,.028))]
    for a,b,c,w in digits:
        joint(a,b,w*1.09,parts)
        joint(b,c,w*.87,parts)
        parts.append(oval('rounded fingertip',c,(w*.84,)*3))
    bpy.ops.object.select_all(action='DESELECT')
    for p in parts:
        p.select_set(True)
    obj=parts[0]
    bpy.context.view_layer.objects.active=obj
    bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    obj.data.remesh_voxel_size=.00093
    bpy.ops.object.voxel_remesh()
    smooth=obj.modifiers.new('soften skin', 'SMOOTH')
    smooth.factor=.385
    smooth.iterations=7
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    dec=obj.modifiers.new('production topology','DECIMATE')
    dec.ratio=min(1.,3200/max(1,len(obj.data.polygons)))
    bpy.ops.object.modifier_apply(modifier=dec.name)
    obj.name='Hand left' if side<0 else 'Hand right'
    for p in obj.data.polygons:
        p.use_smooth=True
    bpy.context.view_layer.update()
    obj.data.calc_loop_triangles()
    point=np.array([v.co for v in obj.data.vertices],np.float32)
    normal=np.array([v.normal for v in obj.data.vertices],np.float32)
    triangles=np.array([list(f.vertices) for f in obj.data.loop_triangles],np.uint16)
    weights=bind_weights(point,digits)

    data=bpy.data.armatures.new(obj.name+' skeleton')
    rig=bpy.data.objects.new(data.name,data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    palm=rig.data.edit_bones.new('palm')
    palm.head=(0,-.020,-.004)
    palm.tail=(0,.039,.002)
    names=['palm']
    for j,(a,b,c,w) in enumerate(digits):
        parent=palm
        for part,begin,end in ((0,a,b),(1,b,c)):
            bone=rig.data.edit_bones.new(f'digit_{j+1}_{part+1}')
            bone.head=begin
            bone.tail=end
            bone.parent=parent
            names.append(bone.name)
            parent=bone
    bpy.ops.object.mode_set(mode='OBJECT')
    deform=obj.modifiers.new('finger skin','ARMATURE')
    deform.object=rig
    for j,name in enumerate(names):
        group=obj.vertex_groups.new(name=name)
        for vertex in np.flatnonzero(weights[:,j]>.001):
            group.add([int(vertex)],float(weights[vertex,j]),'REPLACE')
    obj.parent=rig
    obj.location.x=side*.103
    rig.location.x=side*.103
    obj.location.x=0
    return point,normal,triangles,weights


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    data={}
    for side,tag in ((-1,'L'),(1,'R')):
        p,n,f,w=make_hand(side)
        data.update({f'points_{tag}':p,f'normals_{tag}':n,f'triangles_{tag}':f,f'weights_{tag}':w})
        print(tag,len(p),len(f),float(np.max(np.abs(w.sum(axis=1)-1))),flush=True)
    np.savez_compressed(HERE/'hands.npz',**data)
    bpy.ops.wm.save_as_mainfile(filepath=str(HERE/'hands.blend'))


if __name__=='__main__':
    main()
