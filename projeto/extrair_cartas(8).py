#!/usr/bin/env python3
"""Coloque vídeos em videos/ e execute este arquivo com Python 3.8 ou superior.
Na primeira execução instala OpenCV e RapidOCR (internet necessária).
O OCR é local: nenhum vídeo ou imagem é enviado à internet. TCGdex fornece
apenas o catálogo público. Cache fica na pasta de cache do usuário.
Idioma 'Suíço' é reservado: nacionalidade não é um idioma e não é inferida.
Campos sem confirmação são DESCONHECIDA / DESCONHECIDO / Outro.
"""
from __future__ import annotations
import argparse
import collections
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request
from difflib import SequenceMatcher

BASE = Path(__file__).resolve().parent
EXTENSOES = {'.mov', '.mp4', '.m4v', '.avi', '.mkv', '.webm', '.mts', '.m2ts'}
DESCONHECIDA = {'nome': 'DESCONHECIDA', 'numero': 'DESCONHECIDO', 'idioma': 'Outro'}
SUSPEITA = {**DESCONHECIDA, 'suspeita': True}
CACHE = Path(os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'extrair_cartas'


def dependencias():
    pacotes = {'cv2': 'opencv-python-headless>=4.10,<5',
               'numpy': 'numpy>=1.26,<3',
               'rapidocr': 'rapidocr>=3.9.2,<4',
               'onnxruntime': 'onnxruntime>=1.19,<2'}
    faltantes = [p for m, p in pacotes.items() if importlib.util.find_spec(m) is None]
    if faltantes:
        print('Preparando OCR e análise de vídeo na primeira execução...', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *faltantes], check=True)
    global cv2, np
    import cv2
    import numpy as np
    cv2.setNumThreads(2)


def normalizar(s):
    s = unicodedata.normalize('NFKD', str(s)).lower()
    return ''.join(c for c in s if c.isalnum())


def numero_chave(s):
    return tuple(re.sub(r'(?<!\d)0+(?=\d)', '', p.upper()) for p in s.split('/'))


def nome_arquivo(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', s.replace('/', '-')).strip('_')[:100] or 'DESCONHECIDA'


def json_atomico(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def baixar_json(caminho):
    CACHE.mkdir(parents=True, exist_ok=True)
    arquivo = CACHE / (re.sub(r'[^a-zA-Z0-9_-]', '_', caminho) + '.json')
    antigo = None
    if arquivo.exists():
        try:
            antigo = json.loads(arquivo.read_text(encoding='utf-8'))
            if time.time() - arquivo.stat().st_mtime < 7 * 86400:
                return antigo
        except (ValueError, OSError):
            pass
    for tentativa in range(2):
        try:
            req = urllib.request.Request('https://api.tcgdex.net/v2/' + caminho,
                                         headers={'User-Agent': 'ExtrairCartas/1.0'})
            with urllib.request.urlopen(req, timeout=25) as resposta:
                obj = json.load(resposta)
            json_atomico(arquivo, obj)
            time.sleep(.15)
            return obj
        except (OSError, ValueError):
            if tentativa == 0:
                time.sleep(1)
    return antigo


def ordenar_quad(p):
    p = np.asarray(p, np.float32).reshape(4, 2)
    soma, dif = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return np.array([p[soma.argmin()], p[dif.argmin()], p[soma.argmax()], p[dif.argmax()]], np.float32)


def recortar(frame, quad):
    p = ordenar_quad(quad)
    w = max(np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[3]))
    h = max(np.linalg.norm(p[3]-p[0]), np.linalg.norm(p[2]-p[1]))
    if w > h:
        p = np.roll(p, -1, axis=0)
        w, h = h, w
    # A saída conserva a resolução disponível, sem interpolação para fingir detalhe.
    w, h = max(80, round(w)), max(112, round(h))
    destino = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(p, destino), (w, h))


def detectar(frame):
    escala = min(1., 900 / max(frame.shape[:2]))
    im = cv2.resize(frame, None, fx=escala, fy=escala)
    H, W = im.shape[:2]
    cinza = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    arestas = cv2.Canny(cinza, 40, 120)
    mapas = [cv2.morphologyEx(arestas, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))]
    for canal in cv2.split(im):
        for limiar in (80, 130, 175, 210, 235):
            _, mask = cv2.threshold(canal, limiar, 255, cv2.THRESH_BINARY)
            mapas.append(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)))
    candidatos = []
    for mapa in mapas:
        contornos = cv2.findContours(mapa, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in contornos:
            area = cv2.contourArea(c)
            if not .035 * H * W < area < .88 * H * W:
                continue
            p = cv2.approxPolyDP(c, .022 * cv2.arcLength(c, True), True)
            if len(p) != 4 or not cv2.isContourConvex(p):
                continue
            p = ordenar_quad(p)
            if p[:, 0].min() < 4 or p[:, 1].min() < 4 or p[:, 0].max() > W-5 or p[:, 1].max() > H-5:
                continue
            lados = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
            a, b = (lados[0]+lados[2])/2, (lados[1]+lados[3])/2
            razao = min(a, b)/max(a, b)
            if not .62 < razao < .80 or min(lados[0], lados[2])/max(lados[0], lados[2]) < .80 or min(lados[1], lados[3])/max(lados[1], lados[3]) < .80:
                continue
            angulos = []
            for k in range(4):
                u, v = p[(k-1)%4]-p[k], p[(k+1)%4]-p[k]
                angulos.append(abs(float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))))
            if max(angulos) > .30:
                continue
            # Preferência pelo contorno na proporção real de uma carta, não pela moldura externa.
            score = 1. - abs(razao - 63/88)*5 - np.mean(angulos)*.5
            centro = p.mean(axis=0)
            score -= .12 * np.linalg.norm((centro - [W/2,H/2])/[W,H])
            candidatos.append((float(score), p/escala, area))
    candidatos.sort(key=lambda v: v[2], reverse=True)
    unicos = []
    for score, p, area in candidatos:
        if any(cv2.pointPolygonTest(q.astype(np.float32), tuple(map(float,p.mean(0))), False)>=0 or cv2.pointPolygonTest(p.astype(np.float32), tuple(map(float,q.mean(0))), False)>=0 for _,q,_ in unicos):
            continue
        unicos.append((score,p,area))
    return unicos[:3]


def assinatura(im):
    g = cv2.cvtColor(cv2.resize(im,(96,132)), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g,(5,5),0)
    # Partes superior e inferior distinguem cartas com a mesma ilustração.
    return g.astype(np.float32)


def distancia(a, b):
    a, b = a.ravel(), b.ravel()
    aa, bb = a-a.mean(), b-b.mean()
    return 1 - float(np.dot(aa,bb) / (np.linalg.norm(aa)*np.linalg.norm(bb)+1e-6))


def qualidade(im):
    small = cv2.resize(im,(420,588))
    g = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    nitidez = float(cv2.Laplacian(g,cv2.CV_32F).var())
    hsv = cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    # Brilho sem textura penaliza reflexos; áreas brancas impressas não bastam isoladamente.
    brilho = ((hsv[:,:,2]>237)&(hsv[:,:,1]<45)).astype(np.uint8)
    reflexo = float(cv2.morphologyEx(brilho,cv2.MORPH_OPEN,np.ones((13,13),np.uint8)).mean())
    return math.log1p(nitidez)-4.5*reflexo, nitidez


def qualidade_ocr(im):
    """Pontua principalmente as áreas onde nome e numeração costumam aparecer.

    O objetivo aqui não é escolher a foto mais bonita, e sim o frame mais útil
    para OCR: texto nítido no topo/rodapé e pouco reflexo estourado.
    """
    small=cv2.resize(im,(420,588))
    g=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    h=g.shape[0]
    topo=g[:max(1,round(h*.25))]
    rodape=g[round(h*.70):]
    nitidez_geral=float(cv2.Laplacian(g,cv2.CV_32F).var())
    nitidez_topo=float(cv2.Laplacian(topo,cv2.CV_32F).var())
    nitidez_rodape=float(cv2.Laplacian(rodape,cv2.CV_32F).var())
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    brilho=((hsv[:,:,2]>237)&(hsv[:,:,1]<45)).astype(np.uint8)
    reflexo=float(cv2.morphologyEx(brilho,cv2.MORPH_OPEN,np.ones((13,13),np.uint8)).mean())
    score=(.30*math.log1p(nitidez_geral)+
           .42*math.log1p(nitidez_topo)+
           .28*math.log1p(nitidez_rodape)-5.0*reflexo)
    return score,nitidez_geral


def diferenca_visual(a,b):
    """Diferença barata entre dois recortes; alta = carta/mão em movimento."""
    a=cv2.cvtColor(cv2.resize(a,(160,224)),cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(cv2.resize(b,(160,224)),cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(a,b).mean()/255.0)


def ler_recorte_no_frame(cap,frame_idx,roi,total):
    if frame_idx<0 or (total and frame_idx>=total):return None
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_idx))
    ok,f=cap.read()
    if not ok:return None
    crop,_=recorte_amostra(f,roi)
    return crop


def detectar_linhas(frame, ancora):
    escala=min(1.,900/max(frame.shape[:2]))
    im=cv2.resize(frame,None,fx=escala,fy=escala)
    H,W=im.shape[:2]
    cinza=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cinza,35,100)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=110,maxLineGap=25)
    if lines is None:return []
    vert=[];tops=[]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        if abs(y2-y1)>110 and abs(x2-x1)<abs(y2-y1)*.14:
            k=(x2-x1)/(y2-y1);b=x1-k*y1
            vert.append((abs(y2-y1),k,b,min(y1,y2),max(y1,y2)))
        elif abs(x2-x1)>W*.3 and abs(y2-y1)<abs(x2-x1)*.10:
            tops.append((y1+y2)/2)
    vert.sort(reverse=True)
    vs=[]
    for v in vert:
        if not any(abs((v[1]*H*.6+v[2])-(q[1]*H*.6+q[2]))<9 for q in vs):vs.append(v)
        if len(vs)>=16:break
    dist=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    coords=np.linspace(.035,.965,60,dtype=np.float32)
    esperada=cv2.contourArea(ancora)*escala**2 if ancora is not None else H*W*.3
    candidatos=[]
    for ii,l in enumerate(vs):
        for r in vs[ii+1:]:
            if l[1]*H*.5+l[2]>r[1]*H*.5+r[2]:lft,rgt=r,l
            else:lft,rgt=l,r
            width=(rgt[1]-lft[1])*H*.55+rgt[2]-lft[2]
            height=width/(63/88)
            if not .65*esperada<width*height<1.7*esperada:continue
            ys=[max(lft[3],rgt[3]),min(lft[4],rgt[4])-height]
            ys+= [y for y in tops if H*.15<y<H*.65]
            for y in sorted(set(round(y/6)*6 for y in ys)):
                for factor in [.975,1.,1.025]:
                    y2=y+height*factor
                    p=np.float32([[lft[1]*y+lft[2],y],[rgt[1]*y+rgt[2],y],[rgt[1]*y2+rgt[2],y2],[lft[1]*y2+lft[2],y2]])
                    if p[:,0].min()<5 or p[:,0].max()>W-5 or y<5 or y2>H-5:continue
                    supports=[]
                    for k in range(4):
                        xy=p[k,None,:]*(1-coords[:,None])+p[(k+1)%4,None,:]*coords[:,None]
                        xy=np.rint(xy).astype(int)
                        supports.append(float(np.exp(-dist[xy[:,1],xy[:,0]]/3).mean()))
                    if min(supports)<.36 or np.mean(supports)<.65:continue
                    score=np.mean(supports)+.3*min(supports)
                    candidatos.append((score,p/escala,cv2.contourArea(p)))
    candidatos.sort(key=lambda x:x[0],reverse=True)
    if not candidatos:return []
    limite=candidatos[0][0]-.13
    candidatos=[c for c in candidatos if c[0]>=limite]
    candidatos.sort(key=lambda x:x[2],reverse=True)
    distintos=[]
    for c in candidatos:
        if any(np.mean(np.linalg.norm(c[1]-q[1],axis=1))<25 for q in distintos):continue
        distintos.append(c)
        if len(distintos)==7:break
    return distintos


def aprender_posicao(cap, total):
    # Aprende uma posição recorrente a partir de contornos reais. Isso recupera
    # bordas prateadas que somem contra o suporte branco em parte dos frames.
    amostras=[]
    for i in np.linspace(0,max(0,total-1),32).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
        ok,f=cap.read()
        if not ok:continue
        for score,p,area in detectar(f):
            if area < .12*(900/max(f.shape[:2]))**2*f.shape[0]*f.shape[1]:continue
            if np.linalg.norm(p[1]-p[0])>np.linalg.norm(p[3]-p[0]):continue
            amostras.append(p)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    if len(amostras)<4:return None
    clusters=[]
    for p in amostras:
        grupo=next((g for g in clusters if np.linalg.norm(p.mean(0)-g[0].mean(0))<.12*np.linalg.norm(g[0][2]-g[0][0]) and .75<cv2.contourArea(p)/cv2.contourArea(g[0])<1.33),None)
        if grupo is None:clusters.append([p])
        else:grupo.append(p)
    grupo=max(clusters,key=len)
    if len(grupo)<4:return None
    p=np.median(grupo,axis=0).astype(np.float32)
    return p.mean(0)+(p-p.mean(0))*1.035


def aprender_roi_video(cap, total):
    """Aprende a região em que as cartas mudam ao longo do vídeo.

    Em vez de exigir que as quatro bordas da carta estejam visíveis, usa a
    variação temporal dos pixels. Isso funciona bem em scanners/suportes em que
    a câmera fica parada e apenas a carta muda.
    """
    if not total:
        return None
    amostras=[]
    for i in np.linspace(0,max(0,total-1),24).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
        ok,f=cap.read()
        if not ok:continue
        escala=min(1.,512/f.shape[0],288/f.shape[1])
        pequeno=cv2.resize(f,None,fx=escala,fy=escala)
        amostras.append(pequeno.astype(np.float32))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    if len(amostras)<5:return None

    # Todas as amostras precisam ter o mesmo tamanho.
    h=min(x.shape[0] for x in amostras);w=min(x.shape[1] for x in amostras)
    pilha=np.stack([x[:h,:w] for x in amostras])
    variacao=pilha.std(axis=0).mean(axis=2)
    mask=(variacao>18).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    contornos=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contornos:return None

    # Une regiões grandes próximas; mão/reflexo isolado não deve definir o ROI.
    grandes=[c for c in contornos if cv2.contourArea(c)>.025*h*w]
    if not grandes:grandes=[max(contornos,key=cv2.contourArea)]
    pts=np.vstack(grandes)
    x,y,rw,rh=cv2.boundingRect(pts)
    margem_x=max(8,round(rw*.05));margem_y=max(8,round(rh*.04))
    x=max(0,x-margem_x);y=max(0,y-margem_y)
    x2=min(w,x+rw+2*margem_x);y2=min(h,y+rh+2*margem_y)

    # Volta para as coordenadas do vídeo original.
    escala_x=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/w
    escala_y=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/h
    X1=round(x*escala_x);Y1=round(y*escala_y)
    X2=round(x2*escala_x);Y2=round(y2*escala_y)
    if X2-X1<120 or Y2-Y1<170:return None
    return (X1,Y1,X2,Y2)


def recorte_amostra(frame, roi):
    """Obtém a carta sem descartar frames quando uma borda está fora da câmera."""
    H,W=frame.shape[:2]
    if roi is None:
        # Fallback conservador: região central/inferior, mantendo bastante margem.
        x1=0;y1=round(H*.18);x2=round(W*.90);y2=round(H*.98)
    else:
        x1,y1,x2,y2=roi
        x1=max(0,min(W-1,x1));x2=max(x1+1,min(W,x2))
        y1=max(0,min(H-1,y1));y2=max(y1+1,min(H,y2))

    base=frame[y1:y2,x1:x2]
    if base.size==0:base=frame;x1=y1=0;x2=W;y2=H

    # Se houver um contorno bom dentro do ROI, usa-o; senão mantém o ROI inteiro.
    candidatos=detectar(base)
    if candidatos:
        _,q,_=max(candidatos,key=lambda z:z[0])
        crop=recortar(base,q)
        q=q+np.float32([x1,y1])
        return crop,q

    q=np.float32([[x1,y1],[x2-1,y1],[x2-1,y2-1],[x1,y2-1]])
    return base,q


def extrair_intervalos(video, intervalo=1.2, diagnostico=None):
    """Uma amostra-base por intervalo, com busca barata de frames próximos.

    O vídeo é percorrido UMA vez. Para cada amostra são considerados frames em
    0, +/-0,25 e +/-0,45 s. Isso evita centenas de seeks lentos. Os candidatos
    são ranqueados por nitidez de texto, reflexo e movimento; o OCR começa pelo
    melhor e só tenta os demais se houver falha.
    """
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('Não foi possível abrir o vídeo: '+video.name)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duracao=(total/fps) if total else 0
    print('  Aprendendo a região fixa das cartas...',flush=True)
    roi=aprender_roi_video(cap,total)
    if roi:
        print(f'  Região aprendida: x={roi[0]}..{roi[2]}, y={roi[1]}..{roi[3]}',flush=True)
    else:
        print('  Região automática indisponível; usando recorte amplo de segurança.',flush=True)
    if duracao<=0:
        cap.release();return []

    intervalo=max(.5,float(intervalo))
    bases=list(np.arange(0,duracao,intervalo))
    offsets=(0.0,-.25,.25,-.45,.45)
    alvos=collections.defaultdict(list)
    for gi,tbase in enumerate(bases):
        usados=set()
        for off in offsets:
            t=float(tbase+off)
            if t<0 or t>=duracao:continue
            fi=max(0,min(total-1,int(round(t*fps)))) if total else max(0,int(round(t*fps)))
            if fi in usados:continue
            usados.add(fi)
            alvos[fi].append((gi,abs(off)))

    candidatos=[[] for _ in bases]
    delta=max(1,int(round(fps*.10)))
    historico=collections.deque(maxlen=delta+1)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ultimo_print=-1
    quadro=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break

            H,W=frame.shape[:2]
            if roi is None:
                x1=0;y1=round(H*.18);x2=round(W*.90);y2=round(H*.98)
            else:
                x1,y1,x2,y2=roi
                x1=max(0,min(W-1,x1));x2=max(x1+1,min(W,x2))
                y1=max(0,min(H-1,y1));y2=max(y1+1,min(H,y2))
            bruto=frame[y1:y2,x1:x2]
            if bruto.size==0:
                bruto=frame;x1=y1=0;x2=W;y2=H

            pequeno=cv2.cvtColor(cv2.resize(bruto,(160,224)),cv2.COLOR_BGR2GRAY)
            anterior=historico[0] if len(historico)==historico.maxlen else None

            if quadro in alvos:
                # No modo de posição fixa NÃO tentamos recortar um quadrilátero:
                # isso evita escolher justamente a carta que está passando por cima.
                crop=bruto.copy()
                quad=np.float32([[x1,y1],[x2-1,y1],[x2-1,y2-1],[x1,y2-1]])
                score_ocr,_=qualidade_ocr(crop)
                movimento=(float(cv2.absdiff(pequeno,anterior).mean()/255.0)
                           if anterior is not None else 0.0)
                score=score_ocr-9.0*movimento
                t_real=quadro/fps
                for gi,distbase in alvos[quadro]:
                    candidatos[gi].append((score,len(candidatos[gi]),t_real,crop,quad,distbase))

            historico.append(pequeno)
            quadro+=1
            if total:
                pct=int(min(100,quadro/total*100))
                faixa=pct//10
                if faixa!=ultimo_print:
                    ultimo_print=faixa
                    print(f'  Preparando amostras: {pct}%',flush=True)
    finally:
        cap.release()

    grupos=[]
    for gi,tbase in enumerate(bases):
        cs=candidatos[gi]
        if not cs:continue
        cs.sort(key=lambda x:(x[0],-x[5]),reverse=True)
        grupos.append({'video':str(video),'inicio':float(tbase),'fim':float(tbase),
                       'n':1,'sig':assinatura(cs[0][3]),'referencia':assinatura(cs[0][3]),
                       'centro':cs[0][4].mean(0),'frames':cs,'roi':roi})
        if diagnostico:
            diagnostico.mkdir(parents=True,exist_ok=True)
            for k,item in enumerate(cs[:4]):
                cv2.imwrite(str(diagnostico/f'{len(grupos)-1:04d}_{tbase:.2f}_alt{k}_{item[2]:.2f}.jpg'),item[3])
    return grupos


class Identificador:
    def __init__(self):
        from rapidocr import RapidOCR
        self.ocr = RapidOCR()
        self.catalogos={}
        self.detalhes={}

    def catalogo(self,lingua):
        if lingua not in self.catalogos:
            dados=baixar_json(f'{lingua}/cards')
            if not isinstance(dados,list):
                print(f'  Catálogo {lingua} indisponível; leituras sem validação ficam desconhecidas.',flush=True)
                dados=[]
            indice=collections.defaultdict(list)
            for c in dados:
                if c.get('localId') and c.get('name'):
                    indice[numero_chave(c['localId'])[0]].append(c)
            self.catalogos[lingua]=indice
        return self.catalogos[lingua]

    def ler(self,im):
        # Caminho rápido: uma única chamada de OCR na carta inteira.
        # O rodapé só é relido separadamente se o número não aparecer aqui.
        fator=max(1.,min(2.,800/im.shape[1]))
        img=cv2.resize(im,None,fx=fator,fy=fator)
        H,W=img.shape[:2]
        resultado=self.ocr(img)
        boxes=getattr(resultado, 'boxes', None)
        txts=getattr(resultado, 'txts', None)
        scores=getattr(resultado, 'scores', None)
        if boxes is None or txts is None or scores is None:
            return []
        linhas=[]
        for box,txt,conf in zip(boxes,txts,scores):
            y=(min(p[1] for p in box)+max(p[1] for p in box))/2
            linhas.append((txt,float(conf),y/H))
        return linhas

    def ler_numero_dedicado(self,im):
        """OCR extra só no rodapé, usado apenas quando a leitura normal não acha número.

        A região é ampliada e recebe duas versões de contraste. Como esta rotina só roda
        em falhas, melhora números pequenos sem deixar as cartas fáceis mais lentas.
        """
        H,W=im.shape[:2]
        y1=max(0,round(H*.78)); y2=min(H,max(y1+1,round(H*.995)))
        # O número costuma ficar na metade esquerda; uma segunda região mais larga
        # cobre layouts em que ele aparece um pouco mais ao centro.
        regioes=[im[y1:y2,:max(1,round(W*.68))], im[y1:y2,:]]
        numeros=[]
        for reg in regioes:
            if reg.size==0:continue
            alvo=1500
            fator=max(2.0,min(4.5,alvo/max(1,reg.shape[1])))
            ampliada=cv2.resize(reg,None,fx=fator,fy=fator,interpolation=cv2.INTER_CUBIC)
            cinza=cv2.cvtColor(ampliada,cv2.COLOR_BGR2GRAY)
            clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(cinza)
            variantes=[ampliada,cv2.cvtColor(clahe,cv2.COLOR_GRAY2BGR)]
            for var in variantes:
                resultado=self.ocr(var)
                txts=getattr(resultado,'txts',None);scores=getattr(resultado,'scores',None)
                if txts is None or scores is None:continue
                for txt,conf in zip(txts,scores):
                    if float(conf)<.50:continue
                    t=unicodedata.normalize('NFKC',str(txt)).upper().replace(' ','').replace('／','/')
                    # Correções conservadoras só na região numérica.
                    t=t.replace('O','0')
                    t=re.sub(r'(?<=\d)[|IL\\:-](?=\d)','/',t)
                    for m in re.finditer(r'(?<![A-Z0-9])((?:[A-Z]{1,3})?\d{1,4}[A-Z]?/(?:[A-Z]{1,3})?\d{1,4})(?!\d)',t):
                        numeros.append((m.group(1),float(conf)))
                    for m in re.finditer(r'(?<![A-Z0-9])((?:XY|SM|SWSH|SVP|BW|DP|HGSS)\d{1,4})(?!\d)',t):
                        numeros.append((m.group(1),float(conf)))
        # Mantém a melhor confiança quando o mesmo número apareceu em mais de uma variante.
        melhores={}
        for numero,conf in numeros:
            if numero not in melhores or conf>melhores[numero]:melhores[numero]=conf
        return list(melhores.items())

    def idioma(self,linhas):
        txt=' '+unicodedata.normalize('NFKD',' '.join(x[0] for x in linhas)).lower()+' '
        if len(re.findall(r'[\u3040-\u30ff]',txt))>=4:
            return 'Japonês','ja'
        d={
            'BR':('pt', ['voce','pokemon','energia','dano','danos','ataque','seu','sua','fraqueza','resistencia','recuo','baralho','moeda','habilidade','basico','estagio','oponente','turno','descarte','ligado','ligada','banco','premio','evolui','evolucao','carta']),
            'Inglês':('en',['your','opponent','damage','attack','energy','weakness','resistance','retreat','basic','ability','deck','coin','when','becomes','turn','ends','discard','attached','benched','prize','knocked','evolves']),
            'Alemão':('de',['dein','deines','schaden','energie','angriff','schwache','ruckzug','fahigkeit','gegner','mische','karte'])}
        limpo=unicodedata.normalize('NFKD',txt).encode('ascii','ignore').decode()
        palavras=set(re.findall(r'[a-z]+',limpo))
        scores=sorted([(sum(w in palavras for w in ws),label,code) for label,(code,ws) in d.items()],reverse=True)
        # 'pokemon' sozinho não identifica português.
        if scores[0][0]>=2 and scores[0][0]>=scores[1][0]+1:
            return scores[0][1],scores[0][2]
        return 'Outro',None

    def candidatos_numero(self,linhas):
        nums=[]
        for txt,conf,y in linhas:
            if y<.72 or conf<.76:continue
            t=unicodedata.normalize('NFKC',txt).upper().replace(' ','').replace('／','/')
            for m in re.finditer(r'(?<![A-Z0-9])((?:[A-Z]{1,3})?\d{1,4}[A-Z]?/(?:[A-Z]{1,3})?\d{1,4})(?!\d)',t):
                nums.append((m.group(1),conf))
            for m in re.finditer(r'(?<![A-Z0-9])((?:XY|SM|SWSH|SVP|BW|DP|HGSS)\d{1,4})(?!\d)',t):
                nums.append((m.group(1),conf))
        return list(dict.fromkeys(nums))

    def identificar_linhas(self,linhas,numeros_extra=None):
        """Identifica usando linhas OCR já lidas.

        Separar esta etapa permite juntar leituras de frames próximos da MESMA
        carta: um frame pode ler melhor o nome e outro a numeração, sem fazer
        OCR adicional nem misturar cartas diferentes.
        """
        idioma,lingua=self.idioma(linhas)
        if lingua is None:return None
        numeros=self.candidatos_numero(linhas)
        if numeros_extra:
            vistos_num={n for n,_ in numeros}
            numeros += [(n,c) for n,c in numeros_extra if n not in vistos_num]
        titulos=[]
        for txt,conf,y in linhas:
            if .01<y<.21 and conf>=.72:
                t=re.sub(r'\b(?:BASIC|BÁSICO|BASICO|ESTÁGIO|STAGE|HP|PS|KP)\b','',txt,flags=re.I)
                t=re.sub(r'\d{2,3}\s*$','',t).strip()
                t=normalizar(t)
                t=re.sub(r'^(mega|basic|basico)','',t)
                t=re.sub(r'(?:ps|hp|kp)\d+$','',t)
                titulos.append(t)
        if not numeros or not titulos:return None
        indice=self.catalogo(lingua)
        matches=[]
        for numero,nconf in numeros:
            chave=numero_chave(numero)
            pool=[(lingua,c) for c in indice.get(chave[0],[])]
            if lingua=='pt':
                ids={c['id'] for _,c in pool}
                pool += [('en',c) for c in self.catalogo('en').get(chave[0],[]) if c['id'] not in ids]
            for idioma_catalogo,carta in pool:
                alvo=normalizar(carta['name'])
                score=max((SequenceMatcher(None,alvo,t).ratio() for t in titulos),default=0)
                texto_completo=normalizar(' '.join(x[0] for x in linhas))
                if alvo.endswith('ex') and 'pokemonex' in texto_completo and alvo[:-2] in titulos:
                    score=max(score,.94)
                if any(len(t)>=4 and t in alvo for t in titulos):score=max(score,.60)
                if score<.55:continue
                key=(idioma_catalogo,carta['id'])
                if key not in self.detalhes:self.detalhes[key]=baixar_json(f'{idioma_catalogo}/cards/{carta["id"]}')
                det=self.detalhes[key]
                if not isinstance(det,dict):continue
                if score<.82:
                    ataques=[normalizar(a.get('name','')) for a in det.get('attacks',[])]
                    apoio=any(len(a)>=10 and a in texto_completo for a in ataques)
                    if not apoio or max(map(len,titulos),default=0)<4:continue
                    score=.83
                if len(chave)==2:
                    oficial=det.get('set',{}).get('cardCount',{}).get('official')
                    # TG/GG/SV subsets: don't reinterpret a denominator that the catalog cannot confirm.
                    if oficial is None or numero_chave(str(oficial))[0]!=chave[1]:continue
                elif not re.match(r'^(XY|SM|SWSH|SVP|BW|DP|HGSS)\d+$',numero):continue
                matches.append((score,{'nome':carta['name'],'numero':numero,'idioma':idioma},carta['id']))
        matches.sort(key=lambda x:x[0],reverse=True)
        if not matches:return None
        if len(matches)>1 and matches[0][2]!=matches[1][2] and matches[0][0]-matches[1][0]<.07:
            # Várias impressões com o mesmo nome/número continuam ambíguas.
            if matches[0][1]!=matches[1][1]:return None
        return matches[0][1]

    def identificar_frame(self,im):
        linhas=self.ler(im)
        return self.identificar_linhas(linhas),linhas

    def identificar(self,grupo,limite_tentativas=3):
        """Tenta o melhor frame e vizinhos somente quando necessário.

        Se nenhum frame sozinho fechar nome+numeração+idioma, reaproveita as
        linhas OCR de frames visualmente muito próximos. Assim um frame pode
        fornecer o nome e outro o número sem misturar cartas de uma troca.
        """
        melhor=grupo['frames'][0][3]
        melhor_score=grupo['frames'][0][0]
        refsig=assinatura(melhor)
        linhas_mesma_carta=[]
        cap=None
        try:
            # Até 3 tentativas próximas no fluxo normal; se não resolver, o trecho
            # suspeito recebe a segunda passagem densa. Cartas fáceis custam um OCR.
            for tentativa,item in enumerate(grupo['frames'][:max(1,int(limite_tentativas))]):
                im=item[3]
                sig_im=assinatura(im)
                mesma_base=distancia(sig_im,refsig)<.050
                info,linhas=self.identificar_frame(im)
                if mesma_base:
                    linhas_mesma_carta.extend(linhas)

                if info:
                    return info,im

                # Pode acontecer de o nome sair melhor em um frame e a
                # numeração em outro. Usa apenas frames muito parecidos.
                if mesma_base and len(linhas_mesma_carta)>1:
                    combinado=self.identificar_linhas(linhas_mesma_carta)
                    if combinado:
                        return combinado,im

                # Número muito pequeno (casos como Reuniclus/Zoroark): só então
                # faz OCR ampliado do rodapé, sem penalizar o caminho normal.
                if not self.candidatos_numero(linhas):
                    numeros_rodape=self.ler_numero_dedicado(im)
                    if numeros_rodape:
                        info_num=self.identificar_linhas(linhas,numeros_rodape)
                        if info_num:
                            return info_num,im
                        if mesma_base:
                            combinado=self.identificar_linhas(linhas_mesma_carta,numeros_rodape)
                            if combinado:
                                return combinado,im

                alternativas=[]
                # Se o OCR nem encontrou numeração, tenta corrigir perspectiva
                # dentro do frame original antes de desistir desta tentativa.
                if len(item)>4 and not self.candidatos_numero(linhas):
                    if cap is None:cap=cv2.VideoCapture(grupo['video'])
                    cap.set(cv2.CAP_PROP_POS_MSEC,item[2]*1000)
                    ok,original=cap.read()
                    if ok:
                        for sc,q,area in detectar_linhas(original,item[4])[:3]:
                            centro=q.mean(0)
                            q=centro+(q-centro)*1.012
                            if q[:,0].min()<0 or q[:,1].min()<0 or q[:,0].max()>=original.shape[1] or q[:,1].max()>=original.shape[0]:continue
                            alt=recortar(original,q)
                            alternativas.append(alt)
                        for alt in alternativas:
                            info_alt,linhas_alt=self.identificar_frame(alt)
                            if info_alt:
                                return info_alt,alt
                            if distancia(assinatura(alt),refsig)<.050:
                                linhas_mesma_carta.extend(linhas_alt)
                                combinado=self.identificar_linhas(linhas_mesma_carta)
                                if combinado:
                                    return combinado,alt

                # Última chance para vídeo/carta que tenha chegado invertida.
                if tentativa<2 or not any(y<.19 and c>.8 for _,c,y in linhas):
                    girada=cv2.rotate(im,cv2.ROTATE_180)
                    info_g,linhas_g=self.identificar_frame(girada)
                    if info_g:return info_g,girada

                sc=item[0]
                if sc>melhor_score:
                    melhor_score=sc;melhor=im

            return dict(DESCONHECIDA),melhor
        finally:
            if cap is not None:cap.release()

    def reanalisar_suspeita(self,reg):
        """Segunda passagem densa apenas sobre um trecho que ficou sem identificação.

        Reabre uma janela curta do vídeo em passos de ~0,12 s, conserva somente frames
        visualmente compatíveis com a pendência e reaproveita todo o identificador normal
        com até 12 tentativas. Não roda em cartas já reconhecidas.
        """
        grupos=reg.get('grupos') or []
        if not grupos:return dict(DESCONHECIDA),reg.get('im')
        video=grupos[0].get('video')
        roi=grupos[0].get('roi')
        if not video:return dict(DESCONHECIDA),reg.get('im')
        tempos=[float(g.get('inicio',0.0)) for g in grupos]
        t0=max(0.0,min(tempos)-.50);t1=max(tempos)+.50
        refs=reg.get('sigs') or [reg.get('sig')]
        refs=[x for x in refs if x is not None]
        cap=cv2.VideoCapture(video)
        if not cap.isOpened():return dict(DESCONHECIDA),reg.get('im')
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        duracao=(total/fps) if total else t1
        t1=min(t1,max(0.0,duracao-.001))
        candidatos=[]
        try:
            for k,t in enumerate(np.arange(t0,t1+.001,.12)):
                cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
                ok,frame=cap.read()
                if not ok:continue
                H,W=frame.shape[:2]
                if roi is None:
                    x1=0;y1=round(H*.18);x2=round(W*.90);y2=round(H*.98)
                else:
                    x1,y1,x2,y2=roi
                    x1=max(0,min(W-1,x1));x2=max(x1+1,min(W,x2))
                    y1=max(0,min(H-1,y1));y2=max(y1+1,min(H,y2))
                crop=frame[y1:y2,x1:x2]
                if crop.size==0:continue
                sig=assinatura(crop)
                if refs and min(distancia(sig,r) for r in refs)>.080:
                    continue
                score,_=qualidade_ocr(crop)
                quad=np.float32([[x1,y1],[x2-1,y1],[x2-1,y2-1],[x1,y2-1]])
                candidatos.append((score,k,float(t),crop,quad,0.0))
        finally:
            cap.release()
        if not candidatos:return dict(DESCONHECIDA),reg.get('im')
        candidatos.sort(key=lambda x:x[0],reverse=True)
        grupo={'video':video,'inicio':t0,'fim':t1,'n':len(candidatos),
               'sig':assinatura(candidatos[0][3]),'referencia':assinatura(candidatos[0][3]),
               'centro':candidatos[0][4].mean(0),'frames':candidatos,'roi':roi}
        return self.identificar(grupo,limite_tentativas=12)


def descritor(im):
    g=cv2.cvtColor(cv2.resize(im,(420,588)),cv2.COLOR_BGR2GRAY)
    pontos,desc=cv2.SIFT_create(nfeatures=700).detectAndCompute(g,None)
    return np.float32([p.pt for p in pontos]),desc


def mesma_imagem(a,b):
    pa,da=a;pb,db=b
    if da is None or db is None or len(da)<32 or len(db)<32:return False
    pares=cv2.BFMatcher().knnMatch(da,db,k=2)
    bons=[p[0] for p in pares if len(p)==2 and p[0].distance<.70*p[1].distance]
    if len(bons)<32:return False
    aa=np.float32([pa[m.queryIdx] for m in bons]);bb=np.float32([pb[m.trainIdx] for m in bons])
    _,mask=cv2.findHomography(aa,bb,cv2.RANSAC,3.5)
    if mask is None or mask.sum()<28 or mask.mean()<.65:return False
    ok=mask.ravel().astype(bool)
    return cv2.contourArea(cv2.convexHull(aa[ok]))/(420*588)>.25 and cv2.contourArea(cv2.convexHull(bb[ok]))/(420*588)>.25


def chave_carta(info):
    """Chave de repetição tolerante a zeros à esquerda lidos pelo OCR."""
    if info.get('nome')=='DESCONHECIDA':
        return None
    return (normalizar(info.get('nome','')),
            numero_chave(info.get('numero','')),
            info.get('idioma','Outro'))


def visual_igual(sig,registro):
    """Comparação visual conservadora para continuidade temporal.

    Antes o SIFT podia casar partes fixas da pilha/fundo e declarar Zoroark,
    Reuniclus etc. como uma carta antiga. Agora a decisão usa a assinatura do
    quadro inteiro e um limite estrito. Repetição global continua sendo feita
    por nome+numeração+idioma, nunca por aparência.
    """
    return distancia(sig,registro['sig']) < .040


def executar(args):
    pasta=Path(args.videos).resolve()
    pasta.mkdir(parents=True,exist_ok=True)
    videos=sorted(p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES)
    if not videos:
        print('Coloque seus vídeos na pasta: '+str(pasta))
        return
    dependencias()
    print('Preparando reconhecimento local...',flush=True)
    identificador=Identificador()
    saida=Path(args.saida).resolve()
    if (saida/'cartas.json').exists():
        saida=saida.with_name(saida.name+'_'+time.strftime('%Y%m%d_%H%M%S'))
        n=2
        while saida.exists():
            saida=saida.with_name(saida.name+'_'+str(n));n+=1
    imagens=saida/'imagens';imagens.mkdir(parents=True,exist_ok=True)
    cartas=[];vistos=[];falhas=0
    json_atomico(saida/'cartas.json',{'cartas':cartas})

    def gravar(info,im):
        indice=len(cartas)
        marca='_SUSPEITA' if info.get('suspeita') else ''
        nome=f'{indice+1:04d}{marca}_{nome_arquivo(info["nome"])}_{nome_arquivo(info["numero"])}_{nome_arquivo(info["idioma"])}.jpg'
        ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,96])
        if not ok:raise RuntimeError('Falha ao codificar fotografia')
        (imagens/nome).write_bytes(buf.tobytes())
        cartas.append({'arquivo':nome,**info})
        json_atomico(saida/'cartas.json',{'cartas':cartas})
        return indice,nome

    for video in videos:
        print('\nVídeo: '+video.name,flush=True)
        try:
            grupos=extrair_intervalos(video,args.intervalo)
        except Exception as e:
            falhas+=1;print('  Erro: '+str(e),flush=True);continue

        print(f'  {len(grupos)} amostras de {args.intervalo:g}s. Fazendo OCR (vizinhos só quando necessário)...',flush=True)
        corrente=None

        def salvar_pendente(reg):
            if reg is None or reg.get('tipo')!='pendente':return
            qtd=len(reg.get('grupos') or [])
            print(f'  -> ponto suspeito ({qtd} amostra(s)); fazendo segunda passagem precisa...',flush=True)
            info2,im2=identificador.reanalisar_suspeita(reg)
            key2=chave_carta(info2)
            if key2 is not None:
                repetida=next((r for r in vistos if r['key']==key2),None)
                if repetida is not None:
                    print(f'     recuperada na revisão: repetida: {info2["nome"]} | {info2["numero"]} | {info2["idioma"]}',flush=True)
                    return
                score2,_=qualidade_ocr(im2)
                indice,_=gravar(info2,im2)
                vistos.append({'key':key2,'sig':assinatura(im2),'indice':indice,
                               'score':score2,'amostra':reg['amostra']})
                print(f'     recuperada na revisão: {info2["nome"]} | {info2["numero"]} | {info2["idioma"]}',flush=True)
                return
            # Nada é descartado: fica explicitamente marcado para correção manual.
            suspeita=dict(SUSPEITA)
            indice,nome=gravar(suspeita,reg['im'])
            vistos.append({'key':None,'sig':reg['sig'],'indice':indice,
                           'score':reg['score'],'amostra':reg['amostra']})
            print(f'     não confirmada; salva como SUSPEITA: {nome}',flush=True)

        for i,g in enumerate(grupos,1):
            info,im=identificador.identificar(g)
            sig=assinatura(im)
            score,_=qualidade_ocr(im)
            key=chave_carta(info)

            if key is not None:
                # Uma pendência imediatamente anterior só é descartada se o
                # frame identificado for realmente a MESMA carta. Caso contrário
                # a pendência é preservada como desconhecida, nunca atribuída a
                # alguma carta antiga do vídeo.
                if corrente is not None and corrente.get('tipo')=='pendente':
                    if not visual_igual(sig,corrente):
                        salvar_pendente(corrente)

                repetida=next((r for r in vistos if r['key']==key),None)
                if repetida is not None:
                    print(f'  {i}/{len(grupos)}: repetida: {info["nome"]} | {info["numero"]} | {info["idioma"]}',flush=True)
                    corrente={'tipo':'conhecida','key':key,'sig':sig,
                              'indice':repetida['indice'],'info':info,'amostra':i}
                    continue

                # Recuperação visual de DESCONHECIDA é permitida apenas perto
                # no tempo. Isto evita que uma carta nova substitua uma falha de
                # dezenas de segundos atrás só porque o layout se parece.
                desconhecida=next((r for r in reversed(vistos)
                                   if r['key'] is None and i-r.get('amostra',-999)<=6
                                   and visual_igual(sig,r)),None)
                if desconhecida is not None:
                    idx=desconhecida['indice']
                    antigo=cartas[idx]['arquivo']
                    nome=f'{idx+1:04d}_{nome_arquivo(info["nome"])}_{nome_arquivo(info["numero"])}_{nome_arquivo(info["idioma"])}.jpg'
                    ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,96])
                    if not ok:raise RuntimeError('Falha ao codificar fotografia')
                    (imagens/nome).write_bytes(buf.tobytes())
                    if antigo!=nome:(imagens/antigo).unlink(missing_ok=True)
                    cartas[idx]={'arquivo':nome,**info}
                    desconhecida.update({'key':key,'sig':sig,'score':score,'amostra':i})
                    json_atomico(saida/'cartas.json',{'cartas':cartas})
                    print(f'  {i}/{len(grupos)}: recuperada em frame próximo: {info["nome"]} | {info["numero"]} | {info["idioma"]}',flush=True)
                    corrente={'tipo':'conhecida','key':key,'sig':sig,
                              'indice':idx,'info':info,'amostra':i}
                    continue

                indice,_=gravar(info,im)
                vistos.append({'key':key,'sig':sig,'indice':indice,
                               'score':score,'amostra':i})
                corrente={'tipo':'conhecida','key':key,'sig':sig,
                          'indice':indice,'info':info,'amostra':i}
                print(f'  {i}/{len(grupos)}: {info["nome"]} | {info["numero"]} | {info["idioma"]}',flush=True)
                continue

            # OCR falhou: só podemos dizer "mesma carta" se ela for continuação
            # VISUAL da amostra imediatamente anterior. Não procuramos mais em
            # toda a lista de cartas antigas (causa do Centiskorch/Zoroark).
            if corrente is not None and visual_igual(sig,corrente):
                if corrente.get('tipo')=='conhecida':
                    old=cartas[corrente['indice']]
                    corrente.update({'sig':sig,'amostra':i})
                    print(f'  {i}/{len(grupos)}: OCR falhou neste segundo; mesma carta anterior: {old["nome"]} | {old["numero"]}',flush=True)
                    continue
                corrente.setdefault('grupos',[]).append(g)
                corrente.setdefault('sigs',[]).append(sig)
                if score>corrente['score']:
                    corrente.update({'sig':sig,'im':im,'score':score,'amostra':i})
                else:
                    corrente['amostra']=i
                print(f'  {i}/{len(grupos)}: mesma carta, OCR ainda sem confirmação; marcarei o trecho para revisão precisa',flush=True)
                continue

            # Mudou de aparência. Se havia uma desconhecida anterior, ela não é
            # apagada nem confundida com uma carta antiga: é salva e começa uma
            # nova pendência.
            if corrente is not None and corrente.get('tipo')=='pendente':
                salvar_pendente(corrente)
            corrente={'tipo':'pendente','sig':sig,'im':im,'score':score,'amostra':i,
                      'grupos':[g],'sigs':[sig]}
            print(f'  {i}/{len(grupos)}: OCR falhou; nova carta pendente, tentando os próximos frames',flush=True)

        if corrente is not None and corrente.get('tipo')=='pendente':
            salvar_pendente(corrente)

    conhecidas=sum(x['nome']!='DESCONHECIDA' for x in cartas)
    suspeitas=sum(bool(x.get('suspeita')) for x in cartas)
    print(f'\nConcluído: {len(cartas)} imagens; {conhecidas} identificadas; {len(cartas)-conhecidas} sem confirmação ({suspeitas} suspeitas).')
    if falhas:print(f'{falhas} vídeo(s) não puderam ser processados.')
    print('Saída: '+str(saida))
    print(f'Cada {args.intervalo:g}s usa o melhor frame próximo; trechos sem confirmação recebem uma segunda passagem precisa.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos',default=str(BASE/'videos'))
    parser.add_argument('--saida',default=str(BASE/'resultado'))
    parser.add_argument('--intervalo',type=float,default=1.2,help='Intervalo entre amostras-base em segundos (padrão: 1.2).')
    args=parser.parse_args()
    if not .5<=args.intervalo<=5:parser.error('--intervalo deve estar entre 0.5 e 5 segundos')
    try:
        executar(args)
    except KeyboardInterrupt:
        print('\nInterrompido. Os resultados já gravados foram preservados.')
    except Exception as e:
        print('\nErro: '+str(e),file=sys.stderr)
        print('Verifique a internet na primeira execução. Este arquivo suporta Python 3.13.',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
