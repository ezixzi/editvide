"""Geometry, lighting and the offscreen camera for the Renaissance film."""

from collections import defaultdict
import ctypes
import math

import numpy as np

from stage_gl import Stage, MESH_VERTEX, api, beam, lathe, mesh, normalized, row


TAU=math.tau
PI=math.pi

VERTEX=MESH_VERTEX.replace('uniform mat4 view;', '''
layout(location=7) in vec2 textureUV;
out vec2 uv;
uniform mat4 view;
''').replace('mat=material;', 'mat=material; uv=textureUV;')

FRAGMENT=r"""#version 330 core
in vec3 pos, local, normal;
in vec2 uv;
flat in vec4 mat;
out vec4 color;
uniform vec3 eye, horizon, keyLight, lightTint, cold, lampPos;
uniform float fog, exposure, time, lampGain;
uniform mat4 lightView;
uniform sampler2D shadowMap, atlas;

float hash(vec2 p) {
    p=fract(p*vec2(123.34,437.81));
    p+=dot(p,p+17.91);
    return fract(p.x*p.y);
}
float n2(vec2 p) {
    vec2 i=floor(p), f=fract(p);
    f=f*f*(3.-2.*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),
               mix(hash(i+vec2(0,1)),hash(i+1.),f.x),f.y);
}
float softShadow(vec3 n) {
    vec3 p=(lightView*vec4(pos,1)).xyz*.5+.5;
    if(any(lessThan(p,vec3(.006)))||any(greaterThan(p,vec3(.994)))) return 1.;
    float z=p.z-.00105-.00135*(1.-max(.0,dot(n,keyLight)));
    vec2 u=vec2(.0011,.0009);
    float s=step(z,texture(shadowMap,p.xy+u).r);
    s+=step(z,texture(shadowMap,p.xy-u).r);
    s+=step(z,texture(shadowMap,p.xy+u.yx*vec2(-2.1,1.55)).r);
    s+=step(z,texture(shadowMap,p.xy+u.yx*vec2(2.1,-1.55)).r);
    s+=step(z,texture(shadowMap,p.xy+u*vec2(.3,-2.1)).r);
    s+=step(z,texture(shadowMap,p.xy+u*vec2(-.3,2.1)).r);
    return .27+.1217*s;
}
void main() {
    float kind=mat.a;
    vec3 p=mat.rgb, n=normalize(normal), v=normalize(eye-pos);
    float distanceToLens=length(eye-pos);
    if(kind>7.5 && kind<8.5) {
        float taper=(1.-smoothstep(.13,.5,abs(local.x)))*(1.-smoothstep(.20,.5,abs(local.y)));
        color=vec4(p,.077*taper);
        return;
    }
    if(kind>6.5 && kind<7.5) {
        float a=pow(1.-smoothstep(.01,.5,length(local.xy)),1.35);
        color=vec4(.077,.053,.081,.34*a);
        return;
    }
    if(kind>5.5 && kind<6.5) {
        color=vec4(p,1.);
        return;
    }
    if(kind>11.5 && kind<12.5) {
        vec2 a=vec2(atan(local.y,local.x),local.z);
        float detail=n2(a*vec2(26,39))+n2(a*vec2(73,103))*.6;
        float limb=pow(max(.001,dot(n,v)),.55);
        color=vec4(mix(vec3(.95,.274,.045),vec3(1.,.83,.45),.37*detail+.34*limb),1.);
        return;
    }
    if(kind>12.5 && kind<13.5) {
        vec3 ink=texture(atlas,uv).rgb;
        float lit=max(.05,dot(n,normalize(lampPos-pos)));
        float edge=pow(1.-max(.0,dot(n,v)),2.8);
        color=vec4(clamp(ink*(.53+.73*lit)+edge*vec3(.17,.32,.45),0.,1.),1.);
        return;
    }
    if(kind>9.5 && kind<10.5) {
        vec3 ink=texture(atlas,uv).rgb;
        float falloff=exp(-distanceToLens*fog*.48);
        vec3 grade=ink*(.73+.16*lightTint+.095*cold)*clamp(exposure,.76,1.13);
        color=vec4(mix(horizon,grade,falloff),1.);
        return;
    }
    vec2 onSurface=vec2(pos.x+pos.y*1.147,pos.z+pos.y*.37);
    float natural=n2(onSurface*14.5);
    float grit=.93+.13*natural;
    float specGain=.028, tight=34.;
    if(kind>1.5 && kind<2.5) {
        float grain=n2(vec2(pos.x*4.+pos.y*10.5,pos.z*13.+pos.y*98.));
        grit=.79+.37*grain+.08*natural;
    } else if(kind>2.5 && kind<3.5) {
        float fabric=n2(onSurface*138.);
        grit=.88+.15*fabric+.09*natural;
        specGain=.012;
    } else if(kind>3.5 && kind<4.5) {
        grit=.975+.037*natural;
        specGain=.045;
        tight=55.;
    } else if(kind>4.5 && kind<5.5) {
        specGain=.19;
        tight=53.;
        grit=.84+.18*natural;
    } else if(kind>13.5 && kind<14.5) {
        vec2 cheek=vec2((abs(local.x)-.098)/.048,(local.z+.064)/.062);
        float warm=exp(-dot(cheek,cheek))*smoothstep(.06,.132,local.y);
        p=mix(p,p*vec3(1.08,.88,.83),warm*.32);
        grit=.981+.034*natural;
        specGain=.027;
        tight=46.;
    } else if(kind>10.5 && kind<11.5) {
        float strands=n2(vec2(local.x*284.+local.z*25.,local.y*22.+local.z*15.));
        grit=.82+.25*strands;
        specGain=.12;
        tight=34.;
    }
    float direct=max(.0,dot(n,keyLight));
    float shade=direct>.016 ? softShadow(n) : 1.;
    vec3 light=lampPos-pos;
    float pool=lampGain*(.19+.81*max(.0,dot(n,normalize(light))))/(.61+2.6*dot(light,light));
    float bounce=.19+.19*max(n.z,.0)+.087*max(.0,dot(n,vec3(.26,-.8,.45)));
    float rim=pow(1.-max(.0,dot(n,v)),2.6)*max(.0,dot(n,normalize(vec3(-.39,.70,.52))));
    vec3 source=pow(clamp(p*grit,vec3(.0),vec3(1.)),vec3(2.05));
    vec3 outlit=source*(cold*bounce+lightTint*direct*shade*.94+vec3(1.35,.72,.29)*pool);
    outlit+=source*rim*vec3(.14,.26,.38);
    outlit+=specGain*pow(max(.0,dot(n,normalize(keyLight+v))),tight)*shade*lightTint;
    outlit*=exposure;
    outlit=outlit/(.94+.28*outlit);
    vec3 srgb=pow(max(vec3(.0),outlit),vec3(.475));
    color=vec4(mix(horizon,srgb,exp(-distanceToLens*fog)),1.);
}
"""


def C(code,material=1):
    return tuple(int(code[i:i+2],16)/255 for i in (0,2,4))+(float(material),)


def tint(base,f):
    return tuple(min(1.0,max(0,x*f)) for x in base[:3])+(base[3],)


def mix(a,b,p):
    return a+(b-a)*p


def blend(a,b,p):
    return tuple(mix(x,y,p) for x,y in zip(a,b))


def S(p):
    p=max(0.,min(1.,float(p)))
    return p*p*(3.-2.*p)


def E(p,a,b):
    return S((p-a)/(b-a))


def V(x,y,z):
    return np.array((x,y,z),np.float32)


def norm(v):
    return np.asarray(v)/max(1e-8,np.linalg.norm(v))


def R(x=0.,y=0.,z=0.):
    a,b,c=math.cos(x),math.cos(y),math.cos(z)
    i,j,k=math.sin(x),math.sin(y),math.sin(z)
    return np.array(((c*b,c*j*i-k*a,c*j*a+k*i),
                     (k*b,k*j*i+c*a,k*j*a-c*i),(-j,b*i,b*a)),np.float32)


class PoseGroup(defaultdict):
    def __init__(self,detail):
        super().__init__(list)
        self.detail=detail
        self.surfaces={}


def group(detail=False):
    return PoseGroup(detail)


def surface(g,vertices,material,position=(0,0,0),rotation=None,scale=1.):
    key=f'pose-surface-{len(g.surfaces)}'
    g.surfaces[key]=vertices
    put(g,key,position,(scale,)*3,material,rotation)


def put(g,tag,p,d,mat,rot=None):
    g[tag].append(row(p,d,mat,orient=None if rot is None else tuple(rot.T)))


def cube(g,p,d,mat,rot=None):
    put(g,'bevel',p,d,mat,rot)


def orb(g,p,d,mat,rot=None):
    put(g,'ball',p,d,mat,rot)


def link(g,a,b,w,mat,d=None,tag='lathe'):
    if math.dist(a,b)>1e-6:
        g[tag].append(beam(a,b,w,w if d is None else d,mat))


def pipe(g,pts,w,mat):
    for a,b in zip(pts,pts[1:]):
        link(g,a,b,w,mat)
    for p in pts[1:-1]:
        orb(g,p,(w,)*3,mat)


def ring(g,p,diameter,mat,rot=None,w=.047):
    put(g,'ring',p,(diameter,diameter,diameter*w/.047),mat,rot)


def plate(g,p,d,mat,rot=None):
    put(g,'plane',p,(*d,1.),mat,rot)


def lens(eye,aim,fov,move=0.,dolly=(0,0,0),track=(0,0,0)):
    p=S(move)
    return (blend(eye,tuple(a+b for a,b in zip(eye,dolly)),p),
            blend(aim,tuple(a+b for a,b in zip(aim,track)),p),fov)


def lamp(h,z,key,col,fill,fog=.035,ev=1.20,pool=(0,-1,2),gain=.43,sun=.7):
    return dict(horizon=h,zenith=z,keyLight=normalized(key),lightTint=col,cold=fill,
                fog=fog,exposure=ev,lampPos=pool,lampGain=gain,glow=sun)


def crosslight(a,b,p):
    out={k:blend(v,b[k],p) if isinstance(v,tuple) else mix(v,b[k],p) for k,v in a.items()}
    out['keyLight']=normalized(out['keyLight'])
    return out


def bevel_geometry():
    rings=(-.5,-.472,-.442,.442,.472,.5)
    out=[]
    for axis in (0,1,2):
        a,b=(axis+1)%3,(axis+2)%3
        for sign in (-1,1):
            points=[]
            for y in rings:
                for x in rings:
                    p=np.zeros(3)
                    p[a],p[b],p[axis]=x,y,sign*.5
                    anchor=np.clip(p,-.442,.442)
                    n=norm(p-anchor)
                    points.append(np.r_[anchor+.058*n,n])
            width=len(rings)
            for j in range(width-1):
                for i in range(width-1):
                    k=j*width+i
                    face=(k,k+1,k+width+1,k,k+width+1,k+width)
                    if sign<0:
                        face=tuple(reversed(face))
                    out.extend(points[f] for f in face)
    return np.asarray(out,dtype=np.float32)


def sphere_geometry(sides=28,rings=17):
    return lathe([(math.sin(a)*.5,max(.0005,math.cos(a)*.5),max(.0005,math.cos(a)*.5),0)
                  for a in np.linspace(-PI/2,PI/2,rings)],sides,facets=.015)


def torus_geometry():
    vs,fs=[],[]
    aN,bN=60,10
    for i in range(aN):
        a=TAU*i/aN
        for j in range(bN):
            b=TAU*j/bN
            r=.476+.024*math.cos(b)
            vs.append((r*math.cos(a),r*math.sin(a),.024*math.sin(b)))
    for i in range(aN):
        for j in range(bN):
            k=i*bN+j
            a=i*bN+(j+1)%bN
            b=((i+1)%aN)*bN+(j+1)%bN
            c=((i+1)%aN)*bN+j
            fs.extend(((k,c,b),(k,b,a)))
    return mesh(vs,fs,.035)


def art_mesh(rect,vert):
    x,y,w,h=rect
    out=[]
    for a,b in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,-.5),(.5,.5),(-.5,.5)):
        p=(a,0,b,0,-1,0) if vert else (a,b,0,0,0,1)
        out.append((*p,x+w*(a+.5),y+h*(b+.5)))
    return np.float32(out)


def cloth_height(x,y,wave=.05,pillow=0.,edge=.055):
    z=.018+wave*(.55*math.sin(x*23.+5*y)**2+.23*math.cos(18*y+x*11.))
    return z-edge*abs(x*2)**7+pillow*math.exp(-(x/.26)**2-((y-.345)/.29)**2)


def cloth_surface(wave=.05,pillow=0.,edge=.055,fold=None):
    pts,faces=[],[]
    cols,rows=42,34
    for j in range(rows):
        y=j/(rows-1)-.5
        for i in range(cols):
            x=i/(cols-1)-.5
            z=cloth_height(x,y,wave,pillow,edge)
            pts.append((x,y,z if fold is None else z+fold(x,y)))
    for j in range(rows-1):
        for i in range(cols-1):
            k=j*cols+i
            faces.extend(((k,k+1,k+cols+1),(k,k+cols+1,k+cols)))
    return mesh(pts,faces,.08)


class Studio(Stage):
    def __init__(self,width,height,atlas,rects,samples=1):
        super().__init__(width,height,samples)
        self.program=self.compile_program(VERTEX,FRAGMENT)
        texture=ctypes.c_uint()
        self.GenTextures(1,ctypes.byref(texture))
        self.ActiveTexture(0x84C1)
        self.BindTexture(0x0DE1,texture.value)
        for key,value in ((0x2801,0x2703),(0x2800,0x2601),(0x2802,0x812F),(0x2803,0x812F)):
            self.TexParameteri(0x0DE1,key,value)
        data=np.ascontiguousarray(np.array(atlas)[::-1],np.uint8)
        self.TexImage2D(0x0DE1,0,0x8058,atlas.width,atlas.height,0,0x1908,0x1401,data.ctypes.data)
        api(ctypes.CDLL('libGL.so.1'),'glGenerateMipmap',None,ctypes.c_uint)(0x0DE1)
        self.UseProgram(self.program)
        self.Uniform1i(self.uniform(self.program,'atlas'),1)
        self.ActiveTexture(0x84C0)
        self.add_mesh('bevel',bevel_geometry())
        self.add_mesh('ball',sphere_geometry())
        self.add_mesh('distant',sphere_geometry(12,8))
        self.add_mesh('ring',torus_geometry())
        self.add_mesh('cloth',cloth_surface())
        self.add_mesh('quilt',cloth_surface(.041,.170,.030))
        self.add_mesh('curtain',cloth_surface(.12))
        self.add_mesh('lathe',lathe([(-.5,.449,.449,0),(-.462,.495,.495,0),(.462,.495,.495,0),
                                   (.5,.449,.449,0)],18,facets=.14))
        self.add_mesh('taper',lathe([(-.5,.41,.40,0),(-.39,.49,.47,0),(-.095,.51,.50,0),
                                   (.2,.43,.45,0),(.45,.31,.32,0),(.5,.26,.28,0)],18,facets=.045))
        self.add_mesh('boot',lathe([(-.50,.42,.43,.103),(-.43,.47,.479,.103),(-.17,.48,.47,.09),
                                  (.08,.42,.35,.018),(.33,.35,.265,-.16),(.49,.346,.245,-.172)],
                                 22,.68,.16))
        self.add_mesh('plane',art_mesh((0,0,1,1),False))
        for k,r in rects.items():
            self.add_mesh('up:'+k,art_mesh(r,True))
            self.add_mesh('flat:'+k,art_mesh(r,False))
        self.groups={}
        self.rects=rects

    def add_mesh(self,name,source):
        data=np.zeros((len(source),8),dtype=np.float32)
        data[:,:source.shape[1]]=source
        vao=self.vertex_array()
        self.BindVertexArray(vao)
        buf=self.buffer()
        self.mesh_buffers[name]=buf
        self.BindBuffer(0x8892,buf)
        self.BufferData(0x8892,data.nbytes,data.ctypes.data,0x88E4)
        for index,size,offset in ((0,3,0),(1,3,12),(7,2,24)):
            self.EnableVertexAttribArray(index)
            self.VertexAttribPointer(index,size,0x1406,0,32,offset)
        self.meshes[name]=(vao,len(data))

    def deform_mesh(self,name,source):
        if name not in self.meshes:
            self.add_mesh(name,source)
            return
        data=np.zeros((len(source),8),dtype=np.float32)
        data[:,:source.shape[1]]=source
        self.BindBuffer(0x8892,self.mesh_buffers[name])
        self.BufferData(0x8892,data.nbytes,data.ctypes.data,0x88E0)
        self.meshes[name]=(self.meshes[name][0],len(data))

    def group(self,geometries,stream=None):
        for name,vertices in geometries.surfaces.items():
            self.deform_mesh(name,vertices)
        return super().group(geometries,stream)

    def focus_mask(self,eye,aim,focus):
        if self.antialias:
            return None
        if not hasattr(self,'depth_pixels'):
            self.depth_pixels=np.empty((self.height,self.width),np.float32)
        pixels=self.depth_pixels
        self.ReadPixels(0,0,self.width,self.height,0x1902,0x1406,pixels.ctypes.data)
        step=max(2,self.width//270)
        d=pixels[::-step,::step]
        depth=1./((1.-d)/.085+d/135.)
        forward=norm(np.asarray(aim)-eye)
        focal=np.dot(np.asarray(focus)-eye,forward)
        far=(depth-focal-.40)/1.41
        near=(focal-depth-.44)/.68
        weight=np.clip(np.maximum(far,near),0.,1.)
        return np.array(212*weight*weight*(3.-2.*weight),np.uint8)

    def set(self,name,content):
        if name not in self.groups:
            self.groups[name]=self.group(content)


def art(g,tag,at,w,h,base='4b3125',tilt=None,flat=False,depth=.019):
    r=R() if tilt is None else tilt
    if flat:
        cube(g,at,(w+.013,h+.013,depth),C(base,2),r)
        p=np.asarray(at)+r[:,2]*(depth*.5+.002)
        put(g,'flat:'+tag,p,(w,h,1.),C('ffffff',10),r)
    else:
        cube(g,at,(w+.020,depth,h+.022),C(base,2),r)
        p=np.asarray(at)-r[:,1]*(depth*.53+.002)
        put(g,'up:'+tag,p,(w,1.,h),C('ffffff',10),r)


def ground_shadow(g,x,y,w=.83,h=.72,z=.007):
    plate(g,(x,y,z),(w,h),C('202028',7))
