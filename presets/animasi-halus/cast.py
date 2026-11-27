"""Fresh sculpted cast with a transform rig, two-bone arms and planted strides."""

import math

import numpy as np

from skin import deformed_hand, tailored_limb
from studio import C, E, PI, R, S, TAU, V, ground_shadow, lathe, link, mesh, mix, norm, orb, pipe, put, ring, surface, tint


CAST={
    'marco':('bf8c66','4d2e25','e6cb9e','8c5638','4f4236',1.0),
    'merchant':('ab7652','382921','d9b586','596758','555045',1.02),
    'scholar':('c09273','777067','c4ab81','53645b','55473e',1.06),
    'pope':('c69877','b1a38d','eddbb6','a44337','d9c6a1',1.04),
    'advisor':('a97f60','524038','c7a583','4d3e4c','473f47',1.03),
    'scribe':('c28c62','42312e','c7a27a','685d48','524b44',.96),
    'town':('b98863','5e4937','b29861','9e6445','575044',.96),
    'today':('a96e52','2e2526','81a59a','5d8e84','3b4c55',.99),
    'birthday':('c39474','46352c','9a6c52','9a6c52','535767',.97),
}

MODERN={}


def register_character(name,skin,hair,shirt,trousers,*,scale=1.,short_sleeves=True):
    """Register a contemporary costume while using the shared body and face rig."""
    CAST[name]=(skin,hair,shirt,shirt,trousers,scale)
    MODERN[name]=short_sleeves


BODY_RINGS=[(0,.133,.094,-.006),(.041,.153,.105,-.008),(.123,.154,.109,-.011),
            (.222,.164,.123,-.009),(.322,.185,.124,-.014),(.413,.213,.108,-.018),
            (.470,.175,.097,-.018),(.518,.128,.078,-.010),(.538,.074,.064,.0)]


FACE_TIERS=np.asarray([
    (-.199,.038,.042,.048),(-.181,.071,.072,.038),(-.156,.113,.100,.024),
    (-.121,.141,.125,.012),(-.076,.160,.139,.003),(-.025,.165,.142,-.003),
    (.027,.165,.138,-.008),(.081,.160,.138,-.014),(.132,.148,.126,-.024),
    (.174,.122,.107,-.029),(.206,.075,.076,-.033),(.222,.016,.018,-.035)])


def face_relief(x,z,front):
    bridge=.020*math.exp(-(x/.025)**2-((z-.018)/.068)**2)*front**6
    nose=.059*math.exp(-(x/.033)**2-((z+.039)/.042)**2)*front**6
    alae=.013*math.exp(-((abs(x)-.025)/.016)**2-((z+.054)/.020)**2)*front**7
    muzzle=.017*math.exp(-(x/.069)**2-((z+.100)/.038)**2)*front**5
    cheeks=.008*math.exp(-((abs(x)-.097)/.039)**2-((z+.043)/.053)**2)*front**4
    return bridge+nose+alae+muzzle+cheeks


def face_profile(z):
    return [float(np.interp(z,FACE_TIERS[:,0],FACE_TIERS[:,j])) for j in (1,2,3)]


def face_y(x,z):
    rx,ry,cy=face_profile(z)
    front=math.sqrt(max(0.,1.-(x/rx)**2))
    return cy+ry*front+face_relief(x,z,front)


def face_geometry(closed=False):
    tiers=FACE_TIERS
    higher=[]
    for a,b in zip(tiers,tiers[1:]):
        for j in range(3):
            higher.append(tuple(mix(i,k,j/3) for i,k in zip(a,b)))
    higher.append(tiers[-1])
    pts,faces=[],[]
    n=72
    for z,rx,ry,cy in higher:
        for i in range(n):
            a=TAU*i/n
            x=rx*math.cos(a)
            front=max(0,math.sin(a))
            y=cy+ry*math.sin(a)
            y+=face_relief(x,z,front)
            pts.append((x,y,z))
    for i in range(len(higher)-1):
        for j in range(n):
            a=i*n+j
            b=i*n+(j+1)%n
            faces.extend(((a,b,b+n),(a,b+n,a+n)))
    if closed:
        end=(len(higher)-1)*n
        for i in range(1,n-1):
            faces.extend(((0,i+1,i),(end,end+i,end+i+1)))
    return mesh(pts,faces,.027)


def hair_geometry():
    pts,faces=[],[]
    across,down=72,25
    for k in range(down):
        q=(down-1-k)/(down-1)
        for i in range(across):
            a=TAU*i/across
            front=E(math.sin(a),.02,.74)
            back=1-E(math.sin(a),-.55,.19)
            phi=q*(1.57+.67*back-.566*front+.143*front*math.sin(3.1*a-.43))
            ripple=1+.039*math.sin(10*a+q*2.17)+.025*math.cos(17*a-q*3.6)
            x=.176*math.sin(phi)*math.cos(a)*ripple-.013*(1-q)
            y=.157*math.sin(phi)*math.sin(a)*ripple-.035
            z=.019+.219*math.cos(phi)*ripple
            pts.append((x,y,z))
    for k in range(down-1):
        for i in range(across):
            u=k*across+i
            v=k*across+(i+1)%across
            faces.extend(((u,v,v+across),(u,v+across,u+across)))
    return mesh(pts,faces,.024)


def swept_hair():
    rows,around=31,96
    points,faces=[],[]
    for row in range(rows):
        u=row/(rows-1)
        for i in range(around):
            a=TAU*i/around
            f=math.sin(a)
            front=E(f,.09,.77)
            back=E(-f,.19,.88)
            hairline=-.018+.141*front-.083*back+.009*front*math.cos(3.7*a+.2)
            z=mix(hairline,.222,u)
            rx,ry,cy=face_profile(z)
            envelope=math.sin(PI*u)**.86
            ridges=(.00155*math.cos(13*a-2.1*u)+.00110*math.sin(26*a-3.4*u))*envelope
            wave=(.017+.025*math.exp(-((a-1.95)/.71)**2))*front*envelope
            thickness=.0045+.010*envelope+ridges
            x=(rx+thickness)*math.cos(a)-.014*front*envelope
            y=cy+(ry+thickness)*f+(.74*wave)
            if front:
                y+=face_relief(rx*math.cos(a),z,front)*1.12
            points.append((x,y,z+.62*wave+.008*u))
    for row in range(rows-1):
        for i in range(around):
            a=row*around+i
            b=row*around+(i+1)%around
            faces.extend(((a,b,b+around),(a,b+around,a+around)))
    peak=len(points)
    points.append((0.,-.035,.233))
    for i in range(around):
        faces.append((peak-around+i,peak-around+(i+1)%around,peak))
    return mesh(points,faces,.020)


def mouth(g,coords,scale,jaw,yawn,smile,worry):
    center=-.105-.013*jaw
    width=.088+.027*yawn-.033*jaw
    height=.066*jaw
    shape=lambda x,h: center+h+(.005+.019*smile-.021*worry)*(x/(width*.5))**2
    P=lambda x,z:coords(x,face_y(x,z)+.0042,z)
    if jaw>.06 and g.detail:
        around=24
        rings=5
        points=[P(0.,center)]
        for j in range(1,rings+1):
            q=j/rings
            for i in range(around):
                a=TAU*i/around
                x=width*.5*q*math.cos(a)
                points.append(P(x,shape(x,.5*height*q*math.sin(a))))
        faces=[]
        for i in range(around):
            faces.append((0,1+i,1+(i+1)%around))
        for j in range(rings-1):
            for i in range(around):
                a=1+around*j+i
                b=1+around*j+(i+1)%around
                faces.extend(((a,a+around,b+around),(a,b+around,b)))
        vertices=[]
        for a,b,c in faces:
            v,w,q=(points[i] for i in (a,b,c))
            normal=norm(np.cross(q-v,w-v))
            if np.dot(normal,coords(0,1,center)-coords(0,0,center))<0.:
                normal=-normal
            for vertex in (v,w,q):
                vertices.append((*vertex,*normal))
        surface(g,np.asarray(vertices,np.float32),C('472825',3))
    elif jaw>.06:
        orb(g,P(0.,center),(width*scale,.009*scale,height*scale),C('5c302b',4))
    else:
        line=[P(x,shape(x,0.)) for x in np.linspace(-width*.5,width*.5,11)]
        pipe(g,line,(.0039+.001*smile)*scale,C('855144',4))


def garment():
    pts,faces=[],[]
    n=42
    rings=np.asarray(BODY_RINGS)
    steps=16
    for side in (1,-1):
        start=len(pts)
        for k in range(steps):
            u=k/(steps-1)
            for i in range(n):
                edge=.137+.71*E(u,.58,1.)
                a=edge+(TAU*.5-.017-edge)*i/(n-1)
                zz=.013+(.484-.112*math.sin(a)**8)*u
                w,d,y=[np.interp(zz,rings[:,0],rings[:,j]) for j in (1,2,3)]
                xx=side*math.sin(a)**.83*(w+.0066)
                v=math.cos(a)
                yy=math.copysign(abs(v)**.83,v)*(d+.0066)+y
                pts.append((xx,yy,zz))
        for j in range(steps-1):
            for i in range(n-1):
                a=start+j*n+i
                f=((a,a+n,a+n+1),(a,a+n+1,a+1))
                faces.extend(f if side>0 else tuple(tuple(reversed(t)) for t in f))
    return mesh(pts,faces,.058)


def install(studio):
    studio.add_mesh('sculpt',face_geometry())
    studio.add_mesh('sculpt-bed',face_geometry(closed=True))
    studio.add_mesh('hair',hair_geometry())
    studio.add_mesh('swept-hair',swept_hair())
    studio.add_mesh('shirt',lathe(BODY_RINGS,44,power=.83,facets=.052))
    studio.add_mesh('vest',garment())
    studio.add_mesh('skirt',lathe([(-.52,.236,.184,-.021),(-.48,.243,.207,-.023),(-.37,.216,.174,-.019),
                               (-.15,.177,.139,-.011),(.15,.139,.109,-.012),(.49,.128,.101,-.016)],32,.90,.13))
    studio.add_mesh('knit',lathe([(-.5,.466,.441,0),(-.32,.492,.474,0),(.0,.498,.466,0),
                              (.3,.461,.439,0),(.5,.408,.392,0)],16,.92,.025))


def basis(forward,roll=0.):
    v=norm(forward)
    lateral=V(1,0,0) if abs(v[0])<.96 else V(0,0,1)
    across=norm(lateral-v*np.dot(lateral,v))
    top=np.cross(across,v)
    return np.column_stack((across,v,top))@R(y=roll)


def point(origin,angle,p,s=1.):
    return np.asarray(origin)+angle@np.asarray(p,dtype=np.float32)*s


def fingers(side,curl,index):
    index_slot=3 if side<0 else 0
    for j,(x,L) in enumerate(zip((-.027,-.0095,.0097,.0275),(.054,.069,.075,.062))):
        bend=mix(curl,.025 if j==index_slot else .95,index)
        yield (V(x,.055,.002),V(x*1.13,.055+L*.53,-.007-.025*bend),
               V(x*1.18,.055+L*(.96-.25*bend),-.013-.040*bend),.0158-.0014*(j==0 or j==3))
    inner=-side
    yield (V(inner*.029,.014,-.005),V(inner*.047,.041-.009*curl,-.006-.008*curl),
           V(inner*(.054-.012*curl),.066-.014*curl,-.010-.029*curl),.021)


def hand(g,p,r,side,col,curl,index,s=1.):
    orb(g,point(p,r,(0,.025,.002),s),(.070*s,.079*s,.032*s),col,r)
    for a,b,c,width in fingers(side,curl,index):
        start,hinge,tip=(point(p,r,q,s) for q in (a,b,c))
        w=width*s
        link(g,start,hinge,w,col,tag='knit')
        orb(g,hinge,(w,)*3,col)
        link(g,hinge,tip,w*.87,col,tag='knit')
        orb(g,tip,(w*.87,)*3,col)


def elbow_pos(a,b,side,guide=None):
    offset=b-a
    length=np.linalg.norm(offset)
    if length<1e-5:
        offset=V(side,0,0)
    v=norm(offset)
    d=float(np.clip(length,.102,.568))
    target=a+v*d
    k=(.297**2-.282**2+d*d)/(2*d)
    pole=V(side*.47,-.12,-.88) if guide is None else np.asarray(guide)
    lateral=pole-v*np.dot(v,pole)
    if np.linalg.norm(lateral)<.075:
        lateral=V(side,0,.42)-v*np.dot(v,V(side,0,.42))
    elbow=a+v*k+norm(lateral)*math.sqrt(max(0.,.297**2-k*k))
    return elbow,target


def wrist_control(x,y,z,fx,fy,fz,roll=0.,curl=.25,*,bend=None,index=0.):
    return (V(x,y,z),V(fx,fy,fz),roll,curl,bend,index)


def arm_pose(at,r,side,ctl,scale=1.):
    target,forward,roll,curl,guide,index=ctl
    shoulder=V(side*.184,-.025,.434)
    elbow,wrist=elbow_pos(shoulder,target,side,guide)
    A,E0,W=(point(at,r,p,scale) for p in (shoulder,elbow,wrist))
    return A,E0,W,r@basis(forward,roll)


def touch_control(at,r,side,tip,forward,roll=0.,scale=1.,bend=None):
    local=r.T@norm(forward)
    index_slot=3 if side<0 else 0
    finger=list(fingers(side,.74,1.))[index_slot]
    target=(r.T@(tip-at))/scale-basis(local,roll)@finger[2]
    return wrist_control(*target,*local,roll,.74,bend=bend,index=1.)


def hand_vertices(side,curl,index=0.):
    return deformed_hand(side,list(fingers(side,0.,0.)),list(fingers(side,curl,index)))


def posed_hand(at,rotation,side,control,scale=1.):
    """Return the world-space vertices of the same hand mesh used for rendering."""
    _,_,wrist,orientation=arm_pose(at,rotation,side,control,scale)
    return (hand_vertices(side,control[3],control[5])[:,:3]*scale)@orientation.T+wrist


def arm(g,at,r,side,ctl,skin,cloth,scale,short_sleeves=False):
    A,E0,W,rotation=arm_pose(at,r,side,ctl,scale)
    D=norm(W-E0)
    cuff=W-D*(.028*scale)
    if g.detail:
        if short_sleeves:
            axis=norm(E0-A)
            shoulder=A-axis*.061*scale
            seam=A+axis*.122*scale
            arm_skin=tailored_limb(seam+axis*.021*scale,E0,W,r[:,0],.091*scale,.089*scale,'arm')
            surface(g,arm_skin,skin)
            orb(g,A-r[:,0]*side*.024*scale,(.133*scale,.133*scale,.128*scale),cloth,r)
            link(g,shoulder,seam,.139*scale,cloth,.134*scale,'knit')
            link(g,seam-axis*.017*scale,seam+axis*.006*scale,.115*scale,tint(cloth,1.055))
        else:
            anchor=A-r[:,0]*side*.030*scale-r[:,2]*.021*scale
            sleeve=tailored_limb(anchor,E0,cuff,r[:,0],.111*scale,.111*scale,'sleeve')
            surface(g,sleeve,cloth)
            link(g,cuff-D*.008*scale,cuff+D*.009*scale,.071*scale,tint(cloth,1.10))
            link(g,cuff,W,.043*scale,skin,tag='knit')
        surface(g,hand_vertices(side,ctl[3],ctl[5]),skin,W,rotation,scale)
    else:
        orb(g,A,(.102*scale,.114*scale,.097*scale),cloth,r)
        link(g,A,E0,.109*scale,cloth,.113*scale,'taper')
        orb(g,E0,(.095*scale,.091*scale,.089*scale),cloth)
        link(g,E0,cuff,.086*scale,cloth,.081*scale,'taper')
        link(g,cuff-D*.019*scale,cuff+D*.008*scale,.060*scale,tint(cloth,1.13))
        link(g,cuff,W,.048*scale,skin,tag='knit')
        hand(g,W,rotation,side,skin,ctl[3],ctl[5],scale)
    return W


def head(g,tag,center,r,t,s=1.,blink=0.,surprise=0.,yawn=0.,smile=0.,question=0.,gaze=(0.,0.),mesh_name='sculpt'):
    skin0,hair0,*_=CAST[tag]
    skin=C(skin0,14)
    hair=C(hair0,11)
    s*=CAST[tag][5]
    coords=lambda x,y,z: point(center,r,(x,y,z),s)
    fit=lambda x,y,z:(x*s,y*s,z*s)
    put(g,mesh_name,center,(s,s,s),skin,r)
    for n in (-1,1):
        orb(g,coords(n*.159,-.022,-.061),fit(.072,.052,.114),skin,r@R(y=n*.194))
        orb(g,coords(n*.184,.002,-.052),fit(.031,.013,.066),tint(skin,.78),r@R(y=n*.21))
    natural_hair=tag in MODERN or tag in ('marco','today','birthday')
    put(g,'swept-hair' if natural_hair else 'hair',center,(s,s,s),hair,r)
    if tag=='pope':
        orb(g,coords(0,-.040,.170),fit(.267,.262,.173),C('e9d5ac',3),r)
        ring(g,coords(0,-.04,.161),.259*s,C('e0c28f',3),r)
    elif tag in ('scholar','merchant','advisor','scribe','town'):
        cap=C({'scholar':'4d4237','merchant':'7f563f','advisor':'544654','scribe':'6c5948','town':'665339'}[tag],3)
        orb(g,coords(0,-.049,.158),fit(.342,.305,.136),cap,r)
        orb(g,coords(0,-.040,.110),fit(.350,.328,.064),tint(cap,1.10),r)
    if tag=='scholar':
        for i in range(11):
            u=(i-5)/5
            orb(g,coords(.106*u,.081+.032*(1-u*u),-.151-.032*(1-u*u)),
                fit(.036,.054,.091-.034*abs(u)),tint(hair,1.22),r@R(y=u*.35))
    spontaneous=1-E(abs((t+.43)%3.62-.15),.016,.086)
    lid=max(blink,spontaneous*.94)
    lid*=1-.91*surprise
    for k in (-1,1):
        x=k*.0648
        f=.119
        eye_h=(.038+.009*surprise)*(1-lid)
        if eye_h>.008:
            eye_r=r@R(y=k*.058,z=-k*.365)
            socket=coords(x,f,.013)
            orb(g,socket,fit(.081,.018,eye_h+.004),tint(skin,.77),eye_r)
            white=socket+eye_r[:,1]*(.0024*s)
            orb(g,white,fit(.074,.019,eye_h),C('eee1ca',4),eye_r)
            gx=float(np.clip(gaze[0],-1.,1.))*.010
            gz=float(np.clip(gaze[1],-1.,1.))*.008*(1-lid)
            bulge=.0095*math.sqrt(max(0.,1-(gx/.037)**2-(gz/max(.004,eye_h*.5))**2))
            iris=point(white,eye_r,(gx,bulge+.0008,gz),s)
            orb(g,iris,fit(.026,.0064,.027*(1-lid)),C('725744' if tag in MODERN else '547271',5),eye_r)
            pup=iris+eye_r[:,1]*(.0032*s)
            orb(g,pup,fit(.014,.0035,.0185*(1-lid)),C('202a2c',5),eye_r)
            orb(g,point(pup,eye_r,(-.0052,.0024,.0064*(1-lid)),s),fit(.0048,.0025,.0055),C('fff0cb',6),eye_r)
        else:
            p=[coords(x+q,.125-.333*k*q-.014*(q/.041)**2,.010+.023*q*k-.012*(1-(q/.041)**2))
               for q in np.linspace(-.04,.04,7)]
            pipe(g,p,.006*s,C('714b42',4))
        z=.069+.028*surprise+.024*question*(1 if k<0 else -.35)
        strip=[]
        for q in np.linspace(-.047,.047,7):
            a=x+q
            b=z+.013*(1-(q/.051)**2)+q*k*(.09+question*.29)
            strip.append(coords(a,face_y(a,b)+.0025,b))
        pipe(g,strip,.0102*s,tint(hair,1.07))
    jaw=max(yawn*.83,surprise*.82,question*.15)
    mouth(g,coords,s,jaw,yawn,smile,question)
    return center


def upper(g,name,hip,r,t,*,controls=None,blink=0.,worry=0.,shock=0.,yawn=0.,smile=0.,look=(0.,0.),
          rotate=(0.,0.),head_offset=None,s=1.):
    sk,hr,li,ve,tr,_=CAST[name]
    fabric,skin=C(li,3),C(sk,4)
    breathing=1+.012*math.sin(2.94*t)
    P=lambda p:point(hip,r,p,s)
    put(g,'shirt',hip,(s,s*breathing,s),fabric,r)
    if name not in MODERN and name not in ('today','birthday'):
        put(g,'vest',hip,(s,s*breathing,s),C(ve,3),r)
        if name in ('marco','scribe','merchant'):
            for i in range(5):
                z=.095+i*.063
                y=.104+.014*math.sin(.50*i)
                orb(g,P((.024,y,z)),(.013*s,.014*s,.015*s),C('d3a161',5),r)
    neck_top=V(0,.005,.679)
    head_center=V(0,.007,.739+math.sin(t*2.61)*.004)
    head_r=R(x=rotate[0],z=rotate[1],y=.015*math.sin(t*1.92))
    if head_offset is not None:
        head_center+=np.asarray(head_offset,np.float32)
        neck_top=head_center+head_r@V(0,-.030,-.117)
    link(g,P((0,-.010,.479)),P(neck_top),.134*s,skin,tag='knit')
    ring(g,P((0,-.003,.538)),.151*s,tint(fabric,1.12),r)
    if name=='pope':
        for j in (-1,1):
            pipe(g,[P((j*.086,.067,.468)),P((j*.083,.108,.332)),P((j*.076,.109,.169)),P((j*.082,.090,.035))],
                 .036*s,C('a85040',3))
        orb(g,P((0,.116,.254)),(.043*s,.025*s,.044*s),C('b88b53',5),r)
    ctr=P(head_center)
    head(g,name,ctr,r@head_r,t,s,blink,shock,yawn,smile,worry,look,
         'sculpt-bed' if head_offset is not None else 'sculpt')
    defaults=(wrist_control(-.234,.027,-.006,-.14,.34,-.93,-.20,.34),wrist_control(.234,.027,-.006,.14,.34,-.93,.20,.34))
    if controls is None:
        controls=defaults
    wrist=[arm(g,hip,r,k,c,skin,fabric,s,MODERN.get(name,False)) for k,c in zip((-1,1),controls)]
    return ctr,wrist


def foot_cycle(distance,cycle,scale,side,weight):
    phase=(distance/(cycle*scale)+(side>0)*.5)%1.
    reach=.357*cycle
    if phase<.625:
        return (reach-cycle*phase)*weight,0.
    advance=(phase-.625)/.375
    return (mix(reach-.625*cycle,reach,S(advance))*weight,
            min(.132,.192*cycle)*math.sin(PI*advance)*weight)


def stride(g,name,at,yaw,t,shadow,*,distance=None,weight=1.,controls=None,rotate=(0,0),
           look=(0,0),smile=0.,worry=0.,shock=0.,robe=False,s=1.,cycle=.70):
    r=R(z=yaw)
    base=np.asarray(at,np.float32)
    tr=C(CAST[name][4],3)
    footcolor=C('4c3026' if name not in ('today','birthday') else '37434e',2)
    P=lambda x,y,z:point(base,r,(x,y,z),s)
    phase0=0 if distance is None else distance/(cycle*s)
    bob=(.016*math.cos(TAU*phase0*2.0)+.009)*weight if distance is not None else .004*math.sin(t*1.6)
    hip=P(.011*math.sin(TAU*phase0)*weight,0,.875+bob)
    if robe:
        put(g,'skirt',P(0,0,.598),(s,s,1.05*s),tr,r)
    arm_pose=[]
    for side in (-1,1):
        x=side*.100
        if distance is None:
            y=.028*side
            lift=0.
        else:
            y,lift=foot_cycle(distance,cycle,s,side,weight)
        if not robe:
            foot=P(x,y,.067+lift)
            center=P(x*.87,0,.890+bob)
            knee=P(x*1.03,.085+y*.53,.484+lift*.39+bob*.2)
            if g.detail:
                surface(g,tailored_limb(center,knee,foot,r[:,0],.146*s,.142*s,'trousers'),tr)
            else:
                link(g,center,knee,.181*s,tr,.172*s,'taper')
                orb(g,knee,(.143*s,.143*s,.129*s),tr,r)
                link(g,knee,foot,.113*s,tint(tr,.88),.113*s,'taper')
            put(g,'boot',P(x,y+.047,.095+lift),(.137*s,.258*s,.170*s),footcolor,r)
        else:
            put(g,'boot',P(x*.96,y+.079,.050),(.132*s,.227*s,.090*s),footcolor,r)
        arm_pose.append(wrist_control(side*.240,-.45*y,.004,side*.12,.31,-.94,side*.17,.40))
    ground_shadow(shadow,at[0],at[1],.87*s,.74*s)
    return upper(g,name,hip,r,t,controls=controls or arm_pose,rotate=rotate,look=look,
                 smile=smile,worry=worry,shock=shock,s=s)


def seated(g,name,at,yaw,t,torso_rotation=None,**options):
    r=R(z=yaw)
    p=np.asarray(at,np.float32)
    m=C(CAST[name][4],3)
    for k in (-1,1):
        a=point(p,r,(k*.100,.043,.028))
        b=point(p,r,(k*.112,.361,-.114))
        c=point(p,r,(k*.112,.315,-p[2]+.071))
        if g.detail:
            surface(g,tailored_limb(a,b,c,r[:,0],.141,.140,'trousers'),m)
        else:
            link(g,a,b,.187,m,tag='knit')
            orb(g,b,(.148,.15,.147),m)
            link(g,b,c,.116,tint(m,.91),tag='taper')
        put(g,'boot',c+r[:,1]*.063,(.14,.23,.137),C('452e27',2),r)
    return upper(g,name,p,r if torso_rotation is None else torso_rotation,t,**options)
