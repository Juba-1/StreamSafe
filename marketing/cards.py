import numpy as np, arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont, ImageFilter
W,H=1920,1080
logo=Image.open('/mnt/user-data/attachments/f132bfdd-8a6b-5a7b-ab8f-f40b225d8c5d/42e2e17f-image.png').convert('RGB')
def font(path,size,w):
    f=ImageFont.truetype(path,size)
    try: f.set_variation_by_axes([w] if 'mont' in path else [w,0])
    except Exception as e: print('var',e)
    return f
def ar(s): return s
def bg():
    y,x=np.mgrid[0:H,0:W]
    r=np.sqrt(((x-W/2)/W)**2+((y-H*0.45)/H)**2)
    g=np.clip(1-r*1.6,0,1)**2
    a=np.zeros((H,W,3),np.float32)
    a[...,0]=40*g; a[...,1]=6*g; a[...,2]=60*g
    return Image.fromarray(a.astype(np.uint8))
def glow_text(img,xy,txt,f,fill,glow,anchor='mm'):
    lay=Image.new('RGBA',img.size,(0,0,0,0)); d=ImageDraw.Draw(lay)
    d.text(xy,txt,font=f,fill=glow+(255,),anchor=anchor,direction=DIR(txt))
    lay=lay.filter(ImageFilter.GaussianBlur(14))
    img.paste(lay,(0,0),lay); img.paste(lay,(0,0),lay)
    ImageDraw.Draw(img).text(xy,txt,font=f,fill=fill,anchor=anchor,direction=DIR(txt))
def put_logo(img,h,cy):
    l=logo.resize((int(logo.width*h/logo.height),h),Image.LANCZOS)
    m=np.asarray(l).max(axis=2).astype(np.float32)
    mask=Image.fromarray(np.clip((m-8)*4,0,255).astype(np.uint8))
    img.paste(l,((W-l.width)//2,cy-h//2),mask)
DIR=lambda t:'rtl' if any('\u0600'<=c<='\u06ff' for c in t) else 'ltr'
CA='cairo.ttf'; MO='mont.ttf'
# intro
im=bg()
put_logo(im,430,330)
glow_text(im,(W//2,690),ar('نفس التريلر… بعد ستريم سيف'),font(CA,96,800),(255,255,255),(255,40,200))
glow_text(im,(W//2,800),'THE SAME TRAILER — PROTECTED LIVE',font(MO,46,700),(200,235,255),(40,160,255))
im.save('intro.png')
# outro
im=bg()
put_logo(im,560,360)
glow_text(im,(W//2,745),ar('يغطي المشاهد غير اللائقة قبل ما توصل لمشاهدينك'),font(CA,64,700),(255,255,255),(255,40,200))
glow_text(im,(W//2,840),'Live stream protection for GTA 6',font(MO,44,600),(200,235,255),(40,160,255))
glow_text(im,(W//2,940),ar('النسخة التجريبية للستريمرز — قريباً'),font(CA,50,700),(255,210,90),(255,120,0))
im.save('outro.png')
# watermark (small logo, transparent)
l=logo.resize((int(logo.width*150/logo.height),150),Image.LANCZOS).convert('RGBA')
a=np.asarray(l).copy(); m=a[...,:3].max(axis=2).astype(np.float32)
a[...,3]=np.clip((m-8)*4,0,255)*0.85; Image.fromarray(a).save('wm.png')
im=Image.open('intro.png'); im.resize((960,540)).save('prev_intro.jpg'); Image.open('outro.png').resize((960,540)).save('prev_outro.jpg')
