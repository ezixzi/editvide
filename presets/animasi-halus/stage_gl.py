"""Offline OpenGL stage for the animated LAH miniature sets."""
from collections import defaultdict
import ctypes as C
import math

import numpy as np

FLOAT, INT, UINT, PTR = C.c_float, C.c_int, C.c_uint, C.c_void_p


def api(library, name, result, *arguments):
    f = getattr(library, name)
    f.restype = result
    f.argtypes = arguments
    return f


def integers(*items):
    return (INT * len(items))(*items)


def context(width, height, samples):
    egl = C.CDLL('libEGL.so.1')
    display = api(egl, 'eglGetPlatformDisplay', PTR, UINT, PTR, PTR)(0x31DD, None, None)
    if not api(egl, 'eglInitialize', UINT, PTR, PTR, PTR)(display, None, None):
        raise RuntimeError('EGL surfaceless display unavailable')
    api(egl, 'eglBindAPI', UINT, UINT)(0x30A2)
    config, count = PTR(), INT()
    attributes = [0x3024,8,0x3023,8,0x3022,8,0x3033,1,0x3040,8,0x3025,24]
    if samples > 1:
        attributes += [0x3032,1,0x3031,samples]
    choose = api(egl, 'eglChooseConfig', UINT, PTR, PTR, PTR, INT, PTR)
    choose(display, integers(*attributes,0x3038), C.byref(config),1,C.byref(count))
    if count.value < 1:
        raise RuntimeError('No compatible EGL color/depth buffer')
    surface = api(egl,'eglCreatePbufferSurface',PTR,PTR,PTR,PTR)(
        display,config,integers(0x3057,width,0x3056,height,0x3038))
    handle = api(egl,'eglCreateContext',PTR,PTR,PTR,PTR,PTR)(
        display,config,None,integers(0x3098,3,0x30FB,3,0x3038))
    if not api(egl,'eglMakeCurrent',UINT,PTR,PTR,PTR,PTR)(display,surface,surface,handle):
        raise RuntimeError('EGL could not bind the OpenGL context')
    return egl, display, surface, handle


MESH_VERTEX = '''#version 330 core
layout(location=0) in vec3 vertex;
layout(location=1) in vec3 surfaceNormal;
layout(location=2) in vec4 m0;
layout(location=3) in vec4 m1;
layout(location=4) in vec4 m2;
layout(location=5) in vec4 m3;
layout(location=6) in vec4 material;
uniform mat4 view;
out vec3 pos;
out vec3 local;
out vec3 normal;
flat out vec4 mat;
void main() {
    vec4 p=mat4(m0,m1,m2,m3)*vec4(vertex,1.);
    pos=p.xyz;
    local=vertex;
    normal=normalize(m0.xyz*surfaceNormal.x/dot(m0.xyz,m0.xyz)
        +m1.xyz*surfaceNormal.y/dot(m1.xyz,m1.xyz)
        +m2.xyz*surfaceNormal.z/dot(m2.xyz,m2.xyz));
    mat=material;
    gl_Position=view*p;
}
'''

MESH_FRAGMENT = '''#version 330 core
in vec3 pos;
in vec3 local;
in vec3 normal;
flat in vec4 mat;
out vec4 color;
uniform vec3 eye, horizon, keyLight, lightTint, cold, lampPos;
uniform float fog, exposure, time, lampGain;
uniform mat4 lightView;
uniform sampler2D shadowMap;
float hash(vec2 p) {
    vec3 v=fract(vec3(p.xyx)*.1031);
    v+=dot(v,v.yzx+33.33);
    return fract((v.x+v.y)*v.z);
}
float noise(vec2 p) {
    vec2 i=floor(p), f=fract(p);
    f=f*f*(3.-2.*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),
        mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}
float visibility(vec3 n) {
    vec3 projected=(lightView*vec4(pos,1.)).xyz*.5+.5;
    if (any(lessThan(projected,vec3(.01))) || any(greaterThan(projected,vec3(.99))))
        return 1.;
    float bias=.0013+.0015*(1.-max(dot(n,keyLight),0.));
    vec2 offset=vec2(.00153,.00078);
    float a=step(projected.z-bias,texture(shadowMap,projected.xy+offset).r);
    float b=step(projected.z-bias,texture(shadowMap,projected.xy-offset).r);
    float c=step(projected.z-bias,texture(shadowMap,projected.xy+offset.yx*vec2(-1,1)).r);
    float d=step(projected.z-bias,texture(shadowMap,projected.xy+offset.yx*vec2(1,-1)).r);
    return .41+.59*(a+b+c+d)*.25;
}
void main() {
    float d=distance(eye,pos);
    float groundMist=.47+.61*exp(-max(0.,pos.z)*.47);
    float fade=1.-exp(-pow(d*fog,1.22)*groundMist);
    if(mat.a==7.) {
        float a=pow(1.-smoothstep(.02,.50,length(local.xy)),1.4);
        color=vec4(vec3(.073,.078,.091),a*.53*(1.-fade));
        return;
    }
    if(mat.a==9.) {
        float f=1.-smoothstep(.02,.50,length(local.xy));
        color=vec4(mat.rgb,f*clamp(.62-d*.011,.10,.58));
        return;
    }
    vec3 n=normalize(normal);
    vec3 base=mat.rgb;
    float detail=1.;
    float gloss=0.;
    if (mat.a==1.) {
        float broad=noise(pos.xy*.78), medium=noise(pos.xy*3.8), fine=noise(pos.xy*24.);
        float edge=broad*.60+medium*.30+fine*.10;
        float frost=smoothstep(.54,.73,edge)*smoothstep(.05,.45,pos.z+.31);
        frost*=.25+.74*max(n.z,0.);
        base=mix(base*.76+vec3(.058,.040,.028)*broad,vec3(.64,.68,.67),frost);
        detail=.78+.34*medium+.16*fine;
        n=normalize(n+vec3((medium-.5)*.21,(fine-.5)*.22,0));
        gloss=.065*(1.-frost);
    } else if (mat.a==2.) {
        float grain=noise(vec2(local.x*37.+local.y*18.,local.z*3.3));
        detail=.74+.43*grain+.10*sin((local.x+local.y*.7)*66.+sin(local.z*10.));
    } else if (mat.a==3.) {
        vec3 q=local*8.;
        float weave=noise(q.xz*12.+q.yx*7.);
        detail=.90+.145*noise(q.xz+q.y)+.065*weave;
        gloss=.022;
    } else if (mat.a==4.) {
        detail=.983+.040*noise(local.xz*8.+local.y*8.);
        gloss=.024;
    } else if(mat.a==5.) {
        detail=.83+.20*noise(pos.xz*64.);
        gloss=.21;
    } else if (mat.a==6.) {
        color=vec4(mix(base*1.22,horizon,fade*.74),1);
        return;
    } else if(mat.a==8.) {
        float wave=noise(pos.xy*16.+vec2(time*.34,-time*.19));
        n=normalize(n+vec3(wave-.5,sin(pos.y*20.+time*.8)*.18,0)*.34);
        detail=.91+wave*.16;
        gloss=.48;
        base=mix(base,horizon,.23);
    }
    vec3 viewDir=normalize(eye-pos);
    float diffuse=pow(max(dot(keyLight,n),0.),.84);
    float bounce=.265+.31*(n.z*.5+.5);
    float shadow=diffuse>.025?visibility(n):1.;
    float edge=max(dot(n,vec3(.37,.72,.58)),0.);
    vec3 ray=lampPos-pos;
    float hot=lampGain*(.19+.81*max(dot(n,normalize(ray)),0.))/(1.+.83*dot(ray,ray));
    vec3 ember=vec3(1.0,.68,.42);
    float rim=pow(1.-max(dot(n,viewDir),0.),2.7)*edge*.115;
    float sparkle=pow(max(dot(n,normalize(keyLight+viewDir)),0.),22.)*gloss;
    vec3 lit=base*detail*(bounce*cold+diffuse*.76*shadow*lightTint+hot*ember)
        + sparkle*shadow*lightTint + rim*vec3(.46,.67,.91);
    color=vec4(mix(lit*exposure,horizon,fade),1.);
}
'''

DEPTH_VERTEX = '''#version 330 core
layout(location=0) in vec3 vertex;
layout(location=2) in vec4 m0;
layout(location=3) in vec4 m1;
layout(location=4) in vec4 m2;
layout(location=5) in vec4 m3;
uniform mat4 view;
void main() {
    gl_Position=view*mat4(m0,m1,m2,m3)*vec4(vertex,1.);
}
'''
DEPTH_FRAGMENT = '''#version 330 core
void main() {}
'''

SKY_VERTEX = '''#version 330 core
out vec2 xy;
void main() {
    vec2 p=vec2((gl_VertexID << 1) & 2,gl_VertexID & 2);
    xy=p*2.-1.;
    gl_Position=vec4(xy,0.,1.);
}
'''

SKY_FRAGMENT = '''#version 330 core
in vec2 xy;
out vec4 color;
uniform vec3 horizon, zenith;
uniform vec4 sky;
void main() {
    float altitude=xy.y*sky.z+sky.w;
    float layer=smoothstep(-.18,.84,altitude);
    vec3 p=mix(horizon,zenith,layer);
    vec2 at=vec2(xy.x+sky.x,altitude*4.1);
    float cloud=sin(at.x*2.7+at.y*1.6 + sin(at.x*1.24-at.y*1.3))
        *sin(at.x*.97-at.y*1.9+sin(at.x*2.9+at.y*2.));
    float cloudBank=smoothstep(-.35,.79,cloud)*smoothstep(-.3,.13,altitude)*(1.-smoothstep(.40,.91,altitude));
    p=mix(p,horizon,cloudBank*.51);
    vec2 q=vec2(at.x-.88,(altitude+.04)*2.1);
    float glow=exp(-dot(q,q)*1.7);
    color=vec4(p+glow*vec3(.117,.059,.025)*sky.y,1.);
}
'''


def view_matrix(eye,target,fov,aspect):
    eye,target=np.asarray(eye),np.asarray(target)
    ahead=target-eye
    ahead/=np.linalg.norm(ahead)
    right=np.cross(ahead,(0,0,1))
    right/=np.linalg.norm(right)
    up=np.cross(right,ahead)
    view=np.eye(4)
    view[:3,:3]=[right,up,-ahead]
    view[:3,3]=-view[:3,:3]@eye
    near,far=.085,135.
    focal=1/math.tan(math.radians(fov)/2)
    perspective=np.array([[focal/aspect,0,0,0],[0,focal,0,0],
        [0,0,(far+near)/(near-far),2*far*near/(near-far)],[0,0,-1,0]])
    return np.array(perspective@view,dtype=np.float32,order='F'),float(ahead[2])


def shadow_matrix(eye, target, direction):
    focus=np.asarray(target)
    direction=np.asarray(direction)
    ahead=-direction
    right=np.cross(ahead,(0.,0.,1.))
    right/=np.linalg.norm(right)
    up=np.cross(right,ahead)
    view=np.eye(4)
    view[:3,:3]=[right,up,direction]
    view[:3,3]=-view[:3,:3]@(focus+direction*19.)
    span=max(4.1,math.sqrt(math.dist(eye,target))*2.36)
    projection=np.diag((1/span,1/span,-2/40.,1.))
    projection[2,3]=-1.
    return np.array(projection@view,dtype=np.float32,order='F')


class Stage:
    def __init__(self,width,height,samples=1):
        self.width,self.height=width,height
        self.context=context(width,height,1)
        gl=C.CDLL('libGL.so.1')
        for name,res,args in (
            ('CreateShader',UINT,(UINT,)),('ShaderSource',None,(UINT,INT,PTR,PTR)),
            ('CompileShader',None,(UINT,)),('GetShaderiv',None,(UINT,UINT,PTR)),
            ('GetShaderInfoLog',None,(UINT,INT,PTR,PTR)),('CreateProgram',UINT,()),
            ('AttachShader',None,(UINT,UINT)),('LinkProgram',None,(UINT,)),
            ('GetProgramiv',None,(UINT,UINT,PTR)),('GetProgramInfoLog',None,(UINT,INT,PTR,PTR)),
            ('DeleteShader',None,(UINT,)),('UseProgram',None,(UINT,)),
            ('GenVertexArrays',None,(INT,PTR)),('BindVertexArray',None,(UINT,)),
            ('GenBuffers',None,(INT,PTR)),('BindBuffer',None,(UINT,UINT)),
            ('BufferData',None,(UINT,C.c_ssize_t,PTR,UINT)),('EnableVertexAttribArray',None,(UINT,)),
            ('VertexAttribPointer',None,(UINT,INT,UINT,C.c_ubyte,INT,PTR)),
            ('VertexAttribDivisor',None,(UINT,UINT)),('GetUniformLocation',INT,(UINT,C.c_char_p)),
            ('UniformMatrix4fv',None,(INT,INT,C.c_ubyte,PTR)),('Uniform1f',None,(INT,FLOAT)),
            ('Uniform1i',None,(INT,INT)),
            ('Uniform3f',None,(INT,FLOAT,FLOAT,FLOAT)),('Uniform4f',None,(INT,FLOAT,FLOAT,FLOAT,FLOAT)),
            ('Clear',None,(UINT,)),('Viewport',None,(INT,INT,INT,INT)),('Scissor',None,(INT,INT,INT,INT)),
            ('Enable',None,(UINT,)),('Disable',None,(UINT,)),('DepthMask',None,(C.c_ubyte,)),
            ('BlendFunc',None,(UINT,UINT)),('DrawArrays',None,(UINT,INT,INT)),
            ('DrawArraysInstanced',None,(UINT,INT,INT,INT)),
            ('ReadPixels',None,(INT,INT,INT,INT,UINT,UINT,PTR)),('PixelStorei',None,(UINT,INT)), ('GenFramebuffers',None,(INT,PTR)),
            ('BindFramebuffer',None,(UINT,UINT)), ('GenRenderbuffers',None,(INT,PTR)),
            ('BindRenderbuffer',None,(UINT,UINT)), ('RenderbufferStorageMultisample',None,(UINT,INT,UINT,INT,INT)),
            ('FramebufferRenderbuffer',None,(UINT,UINT,UINT,UINT)), ('CheckFramebufferStatus',UINT,(UINT,)),
            ('BlitFramebuffer',None,(INT,INT,INT,INT,INT,INT,INT,INT,UINT,UINT)),
            ('GenTextures',None,(INT,PTR)),('BindTexture',None,(UINT,UINT)),('ActiveTexture',None,(UINT,)),
            ('TexImage2D',None,(UINT,INT,INT,INT,INT,INT,UINT,UINT,PTR)),('TexParameteri',None,(UINT,UINT,INT)),
            ('FramebufferTexture2D',None,(UINT,UINT,UINT,UINT,INT)),('DrawBuffer',None,(UINT,)),('ReadBuffer',None,(UINT,)),
        ):
            setattr(self,name,api(gl,'gl'+name,res,*args))
        self.program=self.compile_program(MESH_VERTEX,MESH_FRAGMENT)
        self.sky=self.compile_program(SKY_VERTEX,SKY_FRAGMENT)
        self.depth=self.compile_program(DEPTH_VERTEX,DEPTH_FRAGMENT)
        self.sky_vao=self.vertex_array()
        self.meshes,self.locations,self.streams,self.mesh_buffers={},{},{},{}
        self.PixelStorei(0x0D05,1)
        self.BlendFunc(0x0302,0x0303)
        self.pixels=np.empty((height,width,3),np.uint8)
        self.antialias=0
        if samples>1:
            fb=UINT()
            self.GenFramebuffers(1,C.byref(fb))
            self.antialias=fb.value
            self.BindFramebuffer(0x8D40,self.antialias)
            for attachment,fmt in ((0x8CE0,0x8058),(0x8D00,0x81A6)):
                handle=UINT()
                self.GenRenderbuffers(1,C.byref(handle))
                self.BindRenderbuffer(0x8D41,handle.value)
                self.RenderbufferStorageMultisample(0x8D41,samples,fmt,width,height)
                self.FramebufferRenderbuffer(0x8D40,attachment,0x8D41,handle.value)
            if self.CheckFramebufferStatus(0x8D40)!=0x8CD5:
                raise RuntimeError('Multisample framebuffer setup failed')
        self.shadow_size=960
        self.shadow_fbo,self.shadow_tex=self.make_shadow_buffer()

    def make_shadow_buffer(self):
        tex=UINT()
        self.GenTextures(1,C.byref(tex))
        self.BindTexture(0x0DE1,tex.value)
        self.TexImage2D(0x0DE1,0,0x81A6,self.shadow_size,self.shadow_size,0,0x1902,0x1406,None)
        for setting,value in ((0x2801,0x2600),(0x2800,0x2600),(0x2802,0x812F),(0x2803,0x812F)):
            self.TexParameteri(0x0DE1,setting,value)
        fbo=UINT()
        self.GenFramebuffers(1,C.byref(fbo))
        self.BindFramebuffer(0x8D40,fbo.value)
        self.FramebufferTexture2D(0x8D40,0x8D00,0x0DE1,tex.value,0)
        self.DrawBuffer(0)
        self.ReadBuffer(0)
        if self.CheckFramebufferStatus(0x8D40)!=0x8CD5:
            raise RuntimeError("Depth framebuffer setup failed")
        self.BindFramebuffer(0x8D40,self.antialias)
        return fbo.value,tex.value

    def vertex_array(self):
        identifier=UINT()
        self.GenVertexArrays(1,C.byref(identifier))
        return identifier.value

    def buffer(self):
        identifier=UINT()
        self.GenBuffers(1,C.byref(identifier))
        return identifier.value

    def compile_program(self,vertex,fragment):
        program=self.CreateProgram()
        for kind,source in ((0x8B31,vertex),(0x8B30,fragment)):
            shader=self.CreateShader(kind)
            code=C.c_char_p(source.encode())
            self.ShaderSource(shader,1,C.byref(code),None)
            self.CompileShader(shader)
            result=INT()
            self.GetShaderiv(shader,0x8B81,C.byref(result))
            if not result.value:
                message=C.create_string_buffer(8192)
                self.GetShaderInfoLog(shader,len(message),None,message)
                raise RuntimeError(message.value.decode())
            self.AttachShader(program,shader)
            self.DeleteShader(shader)
        self.LinkProgram(program)
        result=INT()
        self.GetProgramiv(program,0x8B82,C.byref(result))
        if not result.value:
            message=C.create_string_buffer(8192)
            self.GetProgramInfoLog(program,len(message),None,message)
            raise RuntimeError(message.value.decode())
        return program

    def add_mesh(self,name,data):
        vertices=np.ascontiguousarray(data,np.float32)
        vao=self.vertex_array()
        self.BindVertexArray(vao)
        buffer=self.buffer()
        self.mesh_buffers[name]=buffer
        self.BindBuffer(0x8892,buffer)
        self.BufferData(0x8892,vertices.nbytes,vertices.ctypes.data,0x88E4)
        for index in (0,1):
            self.EnableVertexAttribArray(index)
            self.VertexAttribPointer(index,3,0x1406,0,24,index*12)
        self.meshes[name]=(vao,len(vertices))

    def deform_mesh(self,name,mesh_data):
        if name not in self.meshes:
            self.add_mesh(name,mesh_data)
            return
        vertices=np.ascontiguousarray(mesh_data,np.float32)
        self.BindBuffer(0x8892,self.mesh_buffers[name])
        self.BufferData(0x8892,vertices.nbytes,vertices.ctypes.data,0x88E0)
        self.meshes[name]=(self.meshes[name][0],len(vertices))

    def group(self,geometries,stream=None):
        group={}
        for name,items in geometries.items():
            if not items:
                continue
            key=(stream,name)
            pointer=self.streams.get(key) if stream else None
            if pointer is None:
                pointer=self.buffer()
                if stream:
                    self.streams[key]=pointer
            self.BindBuffer(0x8892,pointer)
            rows=np.ascontiguousarray(items,np.float32)
            self.BufferData(0x8892,rows.nbytes,rows.ctypes.data,0x88E0 if stream else 0x88E4)
            group[name]=(pointer,len(rows))
        return group

    def uniform(self,program,name):
        key=(program,name)
        if key not in self.locations:
            self.locations[key]=self.GetUniformLocation(program,name.encode())
        return self.locations[key]

    def draw(self,group):
        for name,(pointer,count) in group.items():
            vao,length=self.meshes[name]
            self.BindVertexArray(vao)
            self.BindBuffer(0x8892,pointer)
            for index in range(5):
                slot=index+2
                self.EnableVertexAttribArray(slot)
                self.VertexAttribPointer(slot,4,0x1406,0,80,index*16)
                self.VertexAttribDivisor(slot,1)
            self.DrawArraysInstanced(0x0004,0,length,count)

    def render_view(self,eye,target,fov,light,world,actors,shadow,particles,time,region=None):
        light_matrix=shadow_matrix(eye,target,light["keyLight"])
        self.BindFramebuffer(0x8D40,self.shadow_fbo)
        self.Disable(0x0C11)
        self.Disable(0x0BE2)
        self.Enable(0x0B71)
        self.DepthMask(1)
        self.Viewport(0,0,self.shadow_size,self.shadow_size)
        self.Clear(0x00000100)
        self.UseProgram(self.depth)
        self.UniformMatrix4fv(self.uniform(self.depth,"view"),1,0,light_matrix.ctypes.data)
        self.draw(world)
        self.draw(actors)
        self.BindFramebuffer(0x8D40,self.antialias)

        left,bottom,width,height=region or (0,0,self.width,self.height)
        self.Viewport(left,bottom,width,height)
        self.Enable(0x0C11)
        self.Scissor(left,bottom,width,height)
        self.DepthMask(1)
        self.Clear(0x00004000|0x00000100)
        self.Disable(0x0B71)
        self.Disable(0x0BE2)
        self.UseProgram(self.sky)
        self.BindVertexArray(self.sky_vao)
        matrix,elevation=view_matrix(eye,target,fov,width/height)
        self.Uniform3f(self.uniform(self.sky,'horizon'),*light['horizon'])
        self.Uniform3f(self.uniform(self.sky,'zenith'),*light['zenith'])
        self.Uniform4f(self.uniform(self.sky,'sky'),float(eye[0])*.052,light['glow'],1.,elevation*1.38)
        self.DrawArrays(0x0004,0,3)
        self.Enable(0x0B71)
        self.UseProgram(self.program)
        self.UniformMatrix4fv(self.uniform(self.program,'view'),1,0,matrix.ctypes.data)
        self.UniformMatrix4fv(self.uniform(self.program,"lightView"),1,0,light_matrix.ctypes.data)
        self.ActiveTexture(0x84C0)
        self.BindTexture(0x0DE1,self.shadow_tex)
        self.Uniform1i(self.uniform(self.program,"shadowMap"),0)
        for key in ('horizon','keyLight','lightTint','cold','lampPos'):
            self.Uniform3f(self.uniform(self.program,key),*light[key])
        for key in ('fog','exposure','lampGain'):
            self.Uniform1f(self.uniform(self.program,key),light[key])
        self.Uniform3f(self.uniform(self.program,'eye'),*eye)
        self.Uniform1f(self.uniform(self.program,'time'),time)
        self.draw(world)
        self.draw(actors)
        self.Enable(0x0BE2)
        self.DepthMask(0)
        self.draw(shadow)
        self.draw(particles)
        self.DepthMask(1)
        self.Disable(0x0BE2)
        self.Disable(0x0C11)

    def rgb(self):
        if self.antialias:
            self.BindFramebuffer(0x8CA8,self.antialias)
            self.BindFramebuffer(0x8CA9,0)
            rect=(0,0,self.width,self.height)
            self.BlitFramebuffer(*rect,*rect,0x00004000,0x2600)
            self.BindFramebuffer(0x8CA8,0)
        self.ReadPixels(0,0,self.width,self.height,0x1907,0x1401,self.pixels.ctypes.data)
        if self.antialias:
            self.BindFramebuffer(0x8D40,self.antialias)
        return np.ascontiguousarray(self.pixels[::-1])


def row(position,dimensions,material,rotation=0,orient=None):
    if orient is None:
        c,s=math.cos(rotation),math.sin(rotation)
        orient=((c,s,0),(-s,c,0),(0,0,1))
    return [v for axis,scale in zip(orient,dimensions) for v in (*[a*scale for a in axis],0)]+[*position,1,*material]


def normalized(v):
    length=math.sqrt(sum(x*x for x in v))
    return tuple(x/length for x in v)


def orient_along(a,b):
    d=tuple(y-x for x,y in zip(a,b))
    length=math.sqrt(sum(x*x for x in d))
    if length<1e-9:
        raise ValueError('Zero length axis')
    z=tuple(x/length for x in d)
    x=normalized((z[2],0,-z[0])) if abs(z[2])>.95 else normalized((-z[1],z[0],0))
    y=(z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0])
    return (x,y,z),length


def beam(a,b,width,depth,material):
    axes,length=orient_along(a,b)
    return row(tuple((x+y)*.5 for x,y in zip(a,b)),(width,depth,length),material,orient=axes)


def lathe(rings,sides,power=1,facets=.25,offset=0.):
    vertices=[]
    for z,rx,ry,dy in rings:
        angles=np.arange(sides)*2*np.pi/sides+offset
        xy=np.c_[np.sign(np.cos(angles))*np.abs(np.cos(angles))**power*rx,
                 np.sign(np.sin(angles))*np.abs(np.sin(angles))**power*ry+dy]
        vertices.extend(np.c_[xy,np.full(sides,z)])
    faces=[]
    for layer in range(len(rings)-1):
        for j in range(sides):
            a=layer*sides+j; b=layer*sides+(j+1)%sides
            faces.extend(((a,b,b+sides),(a,b+sides,a+sides)))
    for j in range(1,sides-1):
        u=(len(rings)-1)*sides
        faces.extend(((0,j+1,j),(u,u+j,u+j+1)))
    return mesh(vertices,faces,facets)


def head_mesh():
    rings = [
        (-.50,.12,.15,.062),(-.48,.27,.26,.053),(-.41,.35,.34,.046),
        (-.33,.41,.40,.026),(-.24,.456,.449,.016),(-.15,.478,.473,.010),
        (-.102,.483,.485,.008),(-.075,.485,.49,.005),(-.038,.487,.496,.003),
        (0,.490,.501,.001),(.085,.490,.503,0.),
        (.16,.493,.495,-.005),(.245,.483,.487,-.014),(.33,.472,.471,-.020),
        (.419,.422,.428,-.03),(.48,.322,.33,-.040),(.51,.109,.18,-.046),
    ]
    sides = 36
    vertices = []
    for z,width,depth,shift in rings:
        for angle in np.linspace(0,math.tau,sides,endpoint=False):
            x = math.cos(angle)*width
            front = max(0.,math.sin(angle))
            y = math.sin(angle)*depth+shift
            nose = .057*math.exp(-(x/.107)**2-((z-.03)/.177)**2)
            nose += .096*math.exp(-(x/.126)**2-((z+.066)/.086)**2)
            socket = .079*math.exp(-((abs(x)-.204)/.114)**2-((z-.099)/.084)**2)
            cheek = .037*math.exp(-((abs(x)-.245)/.139)**2-((z+.094)/.129)**2)
            y += (nose-socket+cheek)*front**5
            vertices.append((x,y,z))
    faces = []
    for layer in range(len(rings)-1):
        for side in range(sides):
            a = layer*sides+side
            b = layer*sides+(side+1)%sides
            faces.extend(((a,b,b+sides),(a,b+sides,a+sides)))
    return mesh(vertices,faces,.045)


def mesh(vertices,faces,facets=.15):
    vertices=np.asarray(vertices,dtype=float)
    faces=np.asarray(faces,dtype=np.int64)
    points=vertices[faces]
    raw=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
    lengths=np.linalg.norm(raw,axis=1)
    good=lengths>1e-10
    faces,points,raw,lengths=faces[good],points[good],raw[good],lengths[good]
    norms=raw/lengths[:,None]
    accumulated=np.zeros_like(vertices)
    for v in range(3):
        np.add.at(accumulated,faces[:,v],raw)
    accumulated/=np.maximum(1e-9,np.linalg.norm(accumulated,axis=1)[:,None])
    normals=accumulated[faces]*(1-facets)+norms[:,None,:]*facets
    normals/=np.maximum(1e-9,np.linalg.norm(normals,axis=2)[:,:,None])
    return np.concatenate((points,normals),axis=2).reshape(-1,6).astype(np.float32)


def primitives():
    return {
        'round':lathe([(math.sin(a)*.5,max(.002,math.cos(a)*.5),max(.002,math.cos(a)*.5),0)
                       for a in np.linspace(-math.pi/2,math.pi/2,9)],12,facets=.19),
        'stone':lathe([(-.50,.22,.22,0),(-.32,.47,.38,.035),(-.07,.51,.47,0),
                       (.20,.40,.46,-.03),(.47,.22,.26,.07),(.52,.06,.10,.04)],8,.87,.59,.17),
        'block':lathe([(-.50,.41,.41,0),(-.43,.50,.50,0),(.43,.50,.50,0),(.50,.41,.41,0)],12,.28,.39,.08),
        'bag':lathe([(-.51,.30,.27,0),(-.39,.45,.43,-.015),(-.06,.51,.51,0),(.37,.48,.47,.022),(.49,.35,.35,.027)],10,.54,.28,.23),
        'limb':lathe([(-.56,.30,.28,0),(-.44,.41,.39,0),(-.17,.57,.49,.018),(.24,.61,.52,-.018),(.51,.40,.36,-.015)],10,.88,.21,.2),
        'log':lathe([(-.5,.42,.43,0),(-.47,.5,.5,0),(.43,.46,.50,-.01),(.50,.42,.46,0)],8,1,.55,.10),
        'coat':lathe([(-.50,.44,.49,.01),(-.42,.47,.49,.014),(-.08,.41,.40,-.02),(.22,.52,.49,0),(.40,.56,.44,-.003),(.49,.36,.31,-.05),(.53,.20,.24,-.02)],14,.72,.22),
        'face':head_mesh(),
        'palm':lathe([(-.49,.28,.29,0),(-.34,.43,.37,0),(-.06,.50,.49,0),
                      (.26,.53,.49,.016),(.44,.46,.38,.014),(.50,.37,.33,.014)],16,.64,.11),
        'digit':lathe([(-.54,.22,.26,0),(-.39,.47,.44,0),(.06,.51,.48,0),
                       (.35,.46,.41,0),(.50,.29,.28,-.03),(.57,.09,.091,-.03)],10,.85,.08),
        'boot':lathe([(-.50,.39,.46,0),(-.42,.48,.50,.029),(-.20,.48,.49,.025),
                      (-.02,.45,.45,-.015),(.16,.39,.35,-.099),(.36,.36,.251,-.185),
                      (.50,.34,.235,-.202)],22,.70,.16),
        'sole':lathe([(-.50,.47,.469,.011),(-.23,.51,.505,.011),(.29,.50,.50,.011),
                      (.49,.46,.46,.011)],22,.65,.23),
        'cap':lathe([(-.50,.44,.44,.014),(-.32,.50,.50,.013),(.11,.54,.52,.0),(.39,.48,.45,-.033),(.50,.37,.37,-.041)],22,.96,.27),
        'peak':lathe([(-.50,.38,.42,0),(-.30,.48,.48,0),(.37,.50,.48,-.06),(.50,.45,.43,-.08)],16,.69,.30),
        'nose':mesh([(-.49,-.38,.32),(.49,-.38,.32),(-.38,-.32,-.47),(.38,-.32,-.47),(0,.46,-.28),(0,-.04,.50)],[(0,4,2),(4,1,3),(2,4,3),(0,5,4),(4,5,1),(0,2,3),(3,1,0)],.34),
        'plane':np.array([[-.5,-.5,0,0,0,1],[.5,-.5,0,0,0,1],[.5,.5,0,0,0,1],[-.5,-.5,0,0,0,1],[.5,.5,0,0,0,1],[-.5,.5,0,0,0,1]],dtype=np.float32),
    }


def batch():
    return defaultdict(list)
