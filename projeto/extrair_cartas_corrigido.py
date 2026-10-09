#!/usr/bin/env python3
"""Coloque vídeos em videos/ e execute este arquivo com Python 3.8 ou superior.
Na primeira execução instala OpenCV e RapidOCR (internet necessária).
O OCR é local: nenhum vídeo ou imagem é enviado à internet. TCGdex fornece
apenas o catálogo público. Cache fica na pasta de cache do usuário.
O scanner conta cópias por passagem física. A primeira carta já presente no
scanner é o evento 0; cada troca detectada cria exatamente um novo evento.
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
    vert.sort(reverse=True);vs=[]
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
            ys += [y for y in tops if H*.15<y<H*.65]
            for y in sorted(set(round(y/6)*6 for y in ys)):
                for factor in [.975,1.,1.025]:
                    y2=y+height*factor
                    pp=np.float32([[lft[1]*y+lft[2],y],[rgt[1]*y+rgt[2],y],[rgt[1]*y2+rgt[2],y2],[lft[1]*y2+lft[2],y2]])
                    if pp[:,0].min()<5 or pp[:,0].max()>W-5 or y<5 or y2>H-5:continue
                    supports=[]
                    for k in range(4):
                        xy=pp[k,None,:]*(1-coords[:,None])+pp[(k+1)%4,None,:]*coords[:,None]
                        xy=np.rint(xy).astype(int)
                        supports.append(float(np.exp(-dist[xy[:,1],xy[:,0]]/3).mean()))
                    if min(supports)<.36 or np.mean(supports)<.65:continue
                    score=np.mean(supports)+.3*min(supports)
                    candidatos.append((score,pp/escala,cv2.contourArea(pp)))
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
    amostras=[]
    for i in np.linspace(0,max(0,total-1),32).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
        ok,f=cap.read()
        if not ok:continue
        for score,pp,area in detectar(f):
            if area < .12*(900/max(f.shape[:2]))**2*f.shape[0]*f.shape[1]:continue
            if np.linalg.norm(pp[1]-pp[0])>np.linalg.norm(pp[3]-pp[0]):continue
            amostras.append(pp)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    if len(amostras)<4:return None
    clusters=[]
    for pp in amostras:
        grupo=next((g for g in clusters if np.linalg.norm(pp.mean(0)-g[0].mean(0))<.12*np.linalg.norm(g[0][2]-g[0][0]) and .75<cv2.contourArea(pp)/cv2.contourArea(g[0])<1.33),None)
        if grupo is None:clusters.append([pp])
        else:grupo.append(pp)
    grupo=max(clusters,key=len)
    if len(grupo)<4:return None
    pp=np.median(grupo,axis=0).astype(np.float32)
    return pp.mean(0)+(pp-pp.mean(0))*1.035


def aprender_roi_video(cap, total):
    """Aprende a região em que as cartas mudam ao longo do vídeo."""
    if not total:return None
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
    h=min(x.shape[0] for x in amostras);w=min(x.shape[1] for x in amostras)
    pilha=np.stack([x[:h,:w] for x in amostras])
    variacao=pilha.std(axis=0).mean(axis=2)
    mask=(variacao>18).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    contornos=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contornos:return None
    grandes=[c for c in contornos if cv2.contourArea(c)>.025*h*w]
    if not grandes:grandes=[max(contornos,key=cv2.contourArea)]
    pts=np.vstack(grandes);x,y,rw,rh=cv2.boundingRect(pts)
    margem_x=max(8,round(rw*.05));margem_y=max(8,round(rh*.04))
    x=max(0,x-margem_x);y=max(0,y-margem_y)
    x2=min(w,x+rw+2*margem_x);y2=min(h,y+rh+2*margem_y)
    escala_x=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/w;escala_y=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/h
    X1=round(x*escala_x);Y1=round(y*escala_y);X2=round(x2*escala_x);Y2=round(y2*escala_y)
    if X2-X1<120 or Y2-Y1<170:return None
    return (X1,Y1,X2,Y2)


def recorte_amostra(frame, roi):
    """Obtém a carta sem descartar frames quando uma borda está fora da câmera."""
    H,W=frame.shape[:2]
    if roi is None:
        x1=0;y1=round(H*.18);x2=round(W*.90);y2=round(H*.98)
    else:
        x1,y1,x2,y2=roi
        x1=max(0,min(W-1,x1));x2=max(x1+1,min(W,x2))
        y1=max(0,min(H-1,y1));y2=max(y1+1,min(H,y2))
    base=frame[y1:y2,x1:x2]
    if base.size==0:base=frame;x1=y1=0;x2=W;y2=H
    candidatos=detectar(base)
    if candidatos:
        _,q,_=max(candidatos,key=lambda z:z[0])
        crop=recortar(base,q);q=q+np.float32([x1,y1])
        return crop,q
    q=np.float32([[x1,y1],[x2-1,y1],[x2-1,y2-1],[x1,y2-1]])
    return base,q

def detectar_eventos_passagem(sinais, fps, duracao):
    """Detecta as TROCAS físicas; a carta que já está no scanner não é uma troca.

    O sinal combina três evidências baratas: diferença média, área alterada e
    movimento numa faixa horizontal aprendida automaticamente (o "portão" por
    onde as cartas passam). Há dois níveis de detecção: o principal e uma
    recuperação de picos mais fracos. Eventos próximos são fundidos para que uma
    única jogada, mesmo com mão + carta + sombra, conte apenas uma vez.
    """
    if len(sinais)<max(12,int(fps*.45)):
        return []

    # Cada item: frame, tempo, diferença média, área alterada, atividade por linha.
    perfis=np.stack([x[4] for x in sinais]).astype(np.float32)
    h=perfis.shape[1]

    # Aprende uma faixa que é atravessada repetidamente. Evita as pontas do ROI,
    # que tendem a conter bordas do suporte e movimento parcial da mão.
    recorrencia=np.mean(perfis,axis=0)
    band_h=max(10,int(round(h*.16)))
    kernel=np.ones(band_h,np.float32)/band_h
    suav=np.convolve(recorrencia,kernel,mode='same')
    margem=max(2,int(round(h*.10)))
    if h>2*margem:
        centro=int(np.argmax(suav[margem:h-margem])+margem)
    else:
        centro=int(np.argmax(suav))
    y0=max(0,centro-band_h//2);y1=min(h,y0+band_h)
    y0=max(0,y1-band_h)

    gate=np.asarray([float(x[4][y0:y1].mean()) for x in sinais],np.float32)
    mov=np.asarray([x[2] for x in sinais],np.float32)
    area=np.asarray([x[3] for x in sinais],np.float32)

    # O gate recebe peso alto porque uma carta verdadeira cruza essa faixa; uma
    # pequena mudança de exposição espalhada pelo quadro não costuma fazê-lo.
    bruto=1.55*mov + .62*area + .82*gate
    janela=max(1,int(round(fps*.035)))
    if janela>1:
        k=np.ones(janela,np.float32)/janela
        score=np.convolve(bruto,k,mode='same')
    else:
        score=bruto

    med=float(np.median(score))
    mad=float(np.median(np.abs(score-med)))+1e-6
    p80=float(np.percentile(score,80))
    # Menos conservador que a versão anterior: o segundo estágio abaixo é que
    # impede um brilho/tremor isolado de virar passagem.
    alto=max(.070,med+4.2*mad,p80*.56)
    baixo=max(.024,med+1.55*mad,alto*.31)
    min_altos=max(1,int(round(fps*.025)))
    quietos_fim=max(2,int(round(fps*.075)))

    def montar_evento(a,b,origem='forte'):
        a=max(0,int(a));b=min(len(sinais)-1,int(b))
        if b<a:return None
        trecho=score[a:b+1]
        if trecho.size==0:return None
        p=a+int(np.argmax(trecho))
        ini_frame,ini_t=sinais[a][0],sinais[a][1]
        fim_frame,fim_t=sinais[b][0],sinais[b][1]
        return {'inicio':float(ini_t),'fim':float(fim_t),
                'inicio_frame':int(ini_frame),'fim_frame':int(fim_frame),
                'pico':float(sinais[p][1]),'pico_frame':int(sinais[p][0]),
                'pico_score':float(score[p]),
                'area_max':float(area[a:b+1].max()),
                'gate_max':float(gate[a:b+1].max()),
                'origem':origem,'gate_y0':int(y0),'gate_y1':int(y1)}

    eventos=[]
    ativo=False;ini=0;altos=0;quietos=0
    for i,sc in enumerate(score):
        forte=(sc>=alto and (gate[i]>=.050 or area[i]>=.060) and mov[i]>=.006)
        if not ativo:
            if forte:
                altos+=1
                if altos==1:ini=i
                if altos>=min_altos:
                    ativo=True;quietos=0
            else:
                altos=0
            continue
        if sc<baixo:
            quietos+=1
        else:
            quietos=0
        if quietos>=quietos_fim:
            fim=max(ini,i-quietos+1)
            ev=montar_evento(ini,fim,'forte')
            if ev is not None:
                dur=ev['fim']-ev['inicio']
                if .025<=dur<=2.8 and (ev['area_max']>=.055 or ev['gate_max']>=.085):
                    eventos.append(ev)
            ativo=False;altos=0;quietos=0
    if ativo:
        ev=montar_evento(ini,len(sinais)-1,'forte')
        if ev is not None and .025<=ev['fim']-ev['inicio']<=2.8:
            eventos.append(ev)

    def fundir(lista,gap=.30):
        if not lista:return []
        lista=sorted(lista,key=lambda e:e['inicio'])
        out=[dict(lista[0])]
        for e in lista[1:]:
            a=out[-1]
            if e['inicio']-a['fim']<=gap:
                a['fim']=max(a['fim'],e['fim']);a['fim_frame']=max(a['fim_frame'],e['fim_frame'])
                a['area_max']=max(a.get('area_max',0.),e.get('area_max',0.))
                a['gate_max']=max(a.get('gate_max',0.),e.get('gate_max',0.))
                if e.get('pico_score',0.)>a.get('pico_score',0.):
                    for k in ('pico','pico_frame','pico_score','origem'):
                        a[k]=e[k]
            else:
                out.append(dict(e))
        return out

    eventos=fundir(eventos)

    # Recupera passagens fracas que o limiar principal não alcançou. Como no vídeo
    # deste scanner há vários segundos entre cartas, exigimos separação temporal
    # razoável de um evento já aceito para não duplicar a mesma jogada.
    fraco=max(.048,med+2.15*mad,alto*.53)
    min_sep=.58
    picos=[]
    for i in range(1,len(score)-1):
        if score[i]<fraco or score[i]<score[i-1] or score[i]<score[i+1]:continue
        if gate[i]<.042 and area[i]<.048:continue
        if mov[i]<.0045:continue
        t=float(sinais[i][1])
        if any(abs(t-e['pico'])<min_sep for e in eventos):continue
        if any(abs(t-sinais[j][1])<min_sep for j in picos):
            # Dentro da mesma vizinhança conserva somente o pico maior.
            j=picos[-1]
            if score[i]>score[j]:picos[-1]=i
            continue
        picos.append(i)

    lim_fraco=max(.018,baixo*.72,med+1.05*mad)
    for p in picos:
        a=p;b=p
        limite_frames=max(2,int(round(fps*.42)))
        while a>0 and p-a<limite_frames and score[a-1]>=lim_fraco:a-=1
        while b+1<len(score) and b-p<limite_frames and score[b+1]>=lim_fraco:b+=1
        ev=montar_evento(a,b,'recuperada')
        if ev is None:continue
        if ev['area_max']<.048 and ev['gate_max']<.072:continue
        if any(abs(ev['pico']-e['pico'])<min_sep for e in eventos):continue
        eventos.append(ev)

    eventos=fundir(eventos)
    # Evita um evento artificial nos primeiros instantes: a primeira carta já
    # está posicionada e só uma troca posterior deve incrementar a quantidade.
    eventos=[e for e in eventos if e['pico']>.16 and e['inicio']<duracao-.02]
    return eventos



def selecionar_trocas_confiaveis(eventos):
    """Separa movimentos de troca das pequenas oscilações usando o próprio vídeo.

    Movimentar a mão, balançar uma carta ou mudar o reflexo pode criar vários
    picos pequenos. Eles NÃO são cartas. Procura duas populações de intensidades
    e um vão bem definido entre elas; não usa uma quantidade esperada de cartas.
    Em filmagens sem separação clara, prefere as passagens sustentadas em vez de
    recuperar todo pico fraco como era feito antes.
    """
    if not eventos:
        return [], [], None
    valores = sorted(float(e.get('pico_score', 0.0)) for e in eventos)
    limite = None
    if len(valores) >= 8:
        melhor = None
        # Exige pelo menos 3 picos em cada população para não separar outliers.
        for i in range(3, len(valores) - 2):
            baixo, alto = valores[i-1], valores[i]
            salto = alto - baixo
            if (salto >= max(.16, valores[-1] * .14) and
                    alto >= baixo * 1.55):
                if melhor is None or salto > melhor[0]:
                    melhor = (salto, (baixo + alto) / 2)
        if melhor is not None:
            limite = melhor[1]
    if limite is None:
        fortes = [float(e['pico_score']) for e in eventos if e.get('origem') == 'forte']
        # Sem bimodalidade clara, não transforma o estágio de "recuperação"
        # em dezenas de cópias artificiais. Picos comparáveis aos fortes ficam.
        nivel = float(np.median(fortes)) if fortes else float(np.median(valores))
        limite = max(.055, nivel * .70)
    aceitos = [dict(e) for e in eventos if e.get('pico_score', 0) >= limite]
    descartados = [dict(e) for e in eventos if e.get('pico_score', 0) < limite]
    return aceitos, descartados, float(limite)


def validar_intervalos_estaveis(video, eventos, duracao):
    """Funde picos de uma ÚNICA carta que levou tempo para terminar de deslizar.

    O vídeo pode exibir uma carta nova parada pela metade sobre a antiga, e depois
    ela desliza até o fundo. Detectar dois picos nesse caso NÃO significa duas
    cartas. Para separá-los, precisa existir entre os picos pelo menos uma imagem
    de carta realmente assentada no scanner.

    Evidências independentes de carta assentada:
      1. Contorno retangular completo reconhecido pelo detector de cartas;
      2. Fundo inferior do scanner livre (a carta não avança para fora da tela).
    A segunda evidência cobre cartas parcialmente recortadas ou transparentes,
    onde o detector geométrico ocasionalmente não encontra quatro cantos.
    """
    if len(eventos) < 2:
        return eventos, []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return eventos, []

    def ler(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0., float(t)) * 1000)
        ok, frame = cap.read()
        return frame if ok else None

    def fundo_inferior(frame):
        h, w = frame.shape[:2]
        x0, x1 = round(w * .10), round(w * .90)
        y0, y1 = round(h * .855), round(h * .965)
        return cv2.resize(frame[y0:y1, x0:x1], (96, 32)).astype(np.float32)

    try:
        fundos = []
        for t in np.linspace(0, max(0., duracao-.25), 29):
            frame = ler(t)
            if frame is not None:
                fundos.append(fundo_inferior(frame))
        if len(fundos) < 8:
            return eventos, []
        # Mediana rejeita passagens transitórias e aprende o fundo do scanner.
        padrao = np.median(np.stack(fundos), axis=0)
        consolidacoes = []
        resultado = [dict(eventos[0])]
        for proximo in eventos[1:]:
            anterior = resultado[-1]
            comeco = anterior['fim'] + .06
            final = proximo['inicio'] - .06
            assentada = False
            amostras_validas = 0
            if final > comeco:
                for frac in (.20, .40, .60, .80):
                    t = comeco + (final-comeco) * frac
                    frame = ler(t)
                    if frame is None:
                        continue
                    amostras_validas += 1
                    # Se o suporte está visível abaixo da carta, ela assentou.
                    # Uma carta suspensa/deslocada invade a área inferior.
                    dif = np.abs(fundo_inferior(frame) - padrao).mean(axis=2)
                    invadido = float(np.mean(dif > 30))
                    if invadido < .10:
                        assentada = True
                        break
                    # Checagem geométrica independente quando iluminação/sombra
                    # tornam a comparação do fundo pouco confiável.
                    quadros = detectar(frame)
                    if quadros and quadros[0][0] > .75:
                        assentada = True
                        break

            if not assentada and amostras_validas:
                # Não é uma carta intermediária: os dois movimentos pertencem
                # a uma mesma passagem, que ainda não havia terminado.
                anterior['fim'] = max(anterior['fim'], proximo['fim'])
                anterior['fim_frame'] = max(anterior['fim_frame'], proximo['fim_frame'])
                anterior['area_max'] = max(anterior.get('area_max', 0.), proximo.get('area_max', 0.))
                anterior['gate_max'] = max(anterior.get('gate_max', 0.), proximo.get('gate_max', 0.))
                if proximo.get('pico_score', 0.) > anterior.get('pico_score', 0.):
                    for chave in ('pico', 'pico_frame', 'pico_score'):
                        anterior[chave] = proximo[chave]
                anterior['origem'] = 'fundida'
                consolidacoes.append({'inicio': round(float(comeco), 3),
                                       'fim': round(float(final), 3),
                                       'motivo': 'nenhuma carta assentada entre os movimentos'})
            else:
                resultado.append(dict(proximo))
        return resultado, consolidacoes
    finally:
        cap.release()


def construir_estados(eventos,duracao):
    """Converte N trocas em N+1 estados físicos de carta."""
    estados=[]
    n=len(eventos)
    for eid in range(n+1):
        bruto_ini=0.0 if eid==0 else float(eventos[eid-1]['fim'])
        bruto_fim=float(duracao) if eid==n else float(eventos[eid]['inicio'])
        largura=max(0.,bruto_fim-bruto_ini)
        # Afasta as amostras da mão/carta em movimento, mas não elimina estados curtos.
        margem=min(.16,max(.035,largura*.10)) if largura>.10 else 0.0
        ini=bruto_ini+(margem if eid>0 else min(.05,largura*.05))
        fim=bruto_fim-(margem if eid<n else min(.05,largura*.05))
        if fim-ini<.10:
            meio=(bruto_ini+bruto_fim)/2
            ini=max(bruto_ini,meio-.055);fim=min(bruto_fim,meio+.055)
        estados.append({'evento_id':eid,'inicio':float(max(0.,ini)),
                        'fim':float(max(ini,min(duracao,fim))),
                        'inicio_bruto':float(bruto_ini),'fim_bruto':float(bruto_fim)})
    return estados


def _roi_fixa(frame,roi):
    H,W=frame.shape[:2]
    if roi is None:
        x1=0;y1=round(H*.18);x2=round(W*.90);y2=round(H*.98)
    else:
        x1,y1,x2,y2=roi
        x1=max(0,min(W-1,x1));x2=max(x1+1,min(W,x2))
        y1=max(0,min(H-1,y1));y2=max(y1+1,min(H,y2))
    crop=frame[y1:y2,x1:x2]
    if crop.size==0:
        crop=frame;x1=y1=0;x2=W;y2=H
    q=np.float32([[x1,y1],[x2-1,y1],[x2-1,y2-1],[x1,y2-1]])
    return crop,q


def coletar_frames_estados(video,roi,estados,fps,total,diagnostico=None):
    """Coleta candidatos de OCR somente nos intervalos estáveis entre passagens."""
    alvos=collections.defaultdict(list)
    for estado in estados:
        ini=float(estado['inicio']);fim=float(estado['fim'])
        largura=max(0.,fim-ini)
        if largura<=.001:
            tempos=[ini]
        elif largura<.35:
            tempos=[ini+largura*.25,ini+largura*.50,ini+largura*.75]
        else:
            fracoes=(.08,.22,.38,.55,.72,.88)
            tempos=[ini+largura*f for f in fracoes]
        usados=set()
        for t in tempos:
            fi=max(0,min(total-1,int(round(t*fps)))) if total else max(0,int(round(t*fps)))
            if fi in usados:continue
            usados.add(fi);alvos[fi].append((estado['evento_id'],float(t)))

    por_estado=collections.defaultdict(list)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():return {}
    quadro=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if quadro in alvos:
                crop,quad=_roi_fixa(frame,roi)
                score,_=qualidade_ocr(crop)
                for eid,t in alvos[quadro]:
                    por_estado[eid].append((score,len(por_estado[eid]),t,crop.copy(),quad,0.0))
            quadro+=1
    finally:
        cap.release()

    if diagnostico:
        diagnostico.mkdir(parents=True,exist_ok=True)
        for eid,cs in por_estado.items():
            for k,item in enumerate(sorted(cs,key=lambda x:x[0],reverse=True)[:4]):
                cv2.imwrite(str(diagnostico/f'evento_{eid:04d}_alt{k}_{item[2]:.2f}.jpg'),item[3])
    return dict(por_estado)


def extrair_passagens(video, diagnostico=None):
    """Detecta primeiro as trocas e só então cria UMA unidade de OCR por carta física."""
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
        cap.release();return [],[]

    sinais=[]
    delta=max(1,int(round(fps*.075)))
    historico=collections.deque(maxlen=delta+1)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ultimo_print=-1;quadro=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            bruto,_=_roi_fixa(frame,roi)
            pequeno=cv2.cvtColor(cv2.resize(bruto,(160,224)),cv2.COLOR_BGR2GRAY)
            anterior=historico[0] if len(historico)==historico.maxlen else None
            if anterior is not None:
                delta_luz=float(np.median(pequeno)-np.median(anterior))
                dif=np.abs(pequeno.astype(np.float32)-anterior.astype(np.float32)-delta_luz)
                movimento=float(dif.mean()/255.0)
                mask=(dif>18)
                area_mov=float(mask.mean())
                linhas=mask.mean(axis=1).astype(np.float32)
                sinais.append((quadro,quadro/fps,movimento,area_mov,linhas))
            historico.append(pequeno);quadro+=1
            if total:
                pct=int(min(100,quadro/total*100));faixa=pct//10
                if faixa!=ultimo_print:
                    ultimo_print=faixa;print(f'  Detectando passagens: {pct}%',flush=True)
    finally:
        cap.release()

    candidatos=detectar_eventos_passagem(sinais,fps,duracao)
    eventos,descartados,limiar=selecionar_trocas_confiaveis(candidatos)
    eventos,consolidacoes=validar_intervalos_estaveis(video,eventos,duracao)
    estados=construir_estados(eventos,duracao)
    print(f'  Movimentos candidatos: {len(candidatos)}; ruídos descartados: {len(descartados)}',flush=True)
    if limiar is not None:
        print(f'  Limiar de movimento real aprendido: {limiar:.3f}',flush=True)
    print(f'  Etapas da mesma passagem unificadas: {len(consolidacoes)}',flush=True)
    print(f'  Trocas físicas confirmadas: {len(eventos)}',flush=True)
    print(f'  Cartas físicas estimadas: {len(estados)} (inclui a carta já presente no início)',flush=True)

    por_estado=coletar_frames_estados(video,roi,estados,fps,total,diagnostico)
    grupos=[]
    for estado in estados:
        eid=estado['evento_id']
        cs=list(por_estado.get(eid,[]))
        if not cs:
            # Estado curto/excepcional: tenta exatamente o meio do intervalo.
            t=(estado['inicio']+estado['fim'])/2
            cap=cv2.VideoCapture(str(video))
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
                ok,frame=cap.read()
                if ok:
                    crop,quad=_roi_fixa(frame,roi);score,_=qualidade_ocr(crop)
                    cs=[(score,0,float(t),crop,quad,0.0)]
            finally:
                cap.release()
        if not cs:continue
        cs.sort(key=lambda x:(x[0],-abs(x[2]-(estado['inicio']+estado['fim'])/2)),reverse=True)
        best=cs[0]
        grupos.append({'video':str(video),'inicio':estado['inicio'],'fim':estado['fim'],
                       'inicio_bruto':estado['inicio_bruto'],'fim_bruto':estado['fim_bruto'],
                       'n':1,'sig':assinatura(best[3]),'referencia':assinatura(best[3]),
                       'centro':best[4].mean(0),'frames':cs,'roi':roi,'evento_id':eid,
                       'tempo_representativo':float(best[2])})
    return grupos,eventos


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
        """Relê densamente somente o intervalo ESTÁVEL de uma passagem não identificada."""
        grupos=reg.get('grupos') or []
        if not grupos:return dict(DESCONHECIDA),reg.get('im')
        g0=grupos[0]
        video=g0.get('video');roi=g0.get('roi')
        if not video:return dict(DESCONHECIDA),reg.get('im')
        t0=max(0.0,float(g0.get('inicio',0.0)))
        t1=max(t0,float(g0.get('fim',t0+.2)))
        refs=reg.get('sigs') or [reg.get('sig')]
        refs=[x for x in refs if x is not None]
        cap=cv2.VideoCapture(video)
        if not cap.isOpened():return dict(DESCONHECIDA),reg.get('im')
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        duracao=(total/fps) if total else t1
        t1=min(t1,max(0.0,duracao-.001))
        largura=max(.001,t1-t0)
        n=min(18,max(6,int(math.ceil(largura/.12))))
        tempos=np.linspace(t0,t1,n)
        candidatos=[]
        try:
            for k,t in enumerate(tempos):
                cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
                ok,frame=cap.read()
                if not ok:continue
                crop,quad=_roi_fixa(frame,roi)
                if crop.size==0:continue
                sig=assinatura(crop)
                if refs and min(distancia(sig,r) for r in refs)>.105:
                    continue
                score,_=qualidade_ocr(crop)
                candidatos.append((score,k,float(t),crop,quad,0.0))
        finally:
            cap.release()
        if not candidatos:return dict(DESCONHECIDA),reg.get('im')
        candidatos.sort(key=lambda x:x[0],reverse=True)
        grupo={'video':video,'inicio':t0,'fim':t1,'n':len(candidatos),
               'sig':assinatura(candidatos[0][3]),'referencia':assinatura(candidatos[0][3]),
               'centro':candidatos[0][4].mean(0),'frames':candidatos,'roi':roi,
               'evento_id':g0.get('evento_id')}
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
    cartas=[];falhas=0;agregados={};melhor_score_imagem={}
    diagnostico_passagens=[]
    json_atomico(saida/'cartas.json',{'cartas':cartas})

    def gravar(info,im):
        indice=len(cartas);info=dict(info);info.setdefault('quantidade',1)
        marca='_SUSPEITA' if info.get('suspeita') else ''
        nome=f'{indice+1:04d}{marca}_{nome_arquivo(info["nome"])}_{nome_arquivo(info["numero"])}_{nome_arquivo(info["idioma"])}.jpg'
        ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,96])
        if not ok:raise RuntimeError('Falha ao codificar fotografia')
        (imagens/nome).write_bytes(buf.tobytes())
        cartas.append({'arquivo':nome,**info})
        json_atomico(saida/'cartas.json',{'cartas':cartas})
        return indice,nome

    def registrar_ocorrencia(info,im,score):
        """Cada chamada representa exatamente UMA carta física detectada."""
        key=chave_carta(info)
        if key is None:return None
        if key in agregados:
            idx=agregados[key]
            cartas[idx]['quantidade']=int(cartas[idx].get('quantidade',1) or 1)+1
            if score>melhor_score_imagem.get(idx,-1e9):
                nome=cartas[idx]['arquivo']
                ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,96])
                if ok:
                    (imagens/nome).write_bytes(buf.tobytes());melhor_score_imagem[idx]=score
            json_atomico(saida/'cartas.json',{'cartas':cartas})
            return idx
        idx,_=gravar({**info,'quantidade':1},im)
        agregados[key]=idx;melhor_score_imagem[idx]=score
        return idx

    for video in videos:
        print('\nVídeo: '+video.name,flush=True)
        try:
            grupos,eventos=extrair_passagens(video)
        except Exception as e:
            falhas+=1;print('  Erro: '+str(e),flush=True);continue

        diagnostico_passagens.append({
            'video':video.name,
            'trocas_detectadas':len(eventos),
            'cartas_fisicas_estimadas':len(grupos),
            'eventos':[{'numero':i+1,'inicio':round(e['inicio'],3),'fim':round(e['fim'],3),
                        'pico':round(e['pico'],3),'forca':round(e.get('pico_score',0.),4),
                        'origem':e.get('origem','forte')} for i,e in enumerate(eventos)]
        })
        json_atomico(saida/'passagens.json',{'videos':diagnostico_passagens})

        print(f'  {len(grupos)} carta(s) física(s). Fazendo OCR uma vez por passagem...',flush=True)
        for i,g in enumerate(grupos,1):
            info,im=identificador.identificar(g)
            score,_=qualidade_ocr(im)
            recuperada=False

            if chave_carta(info) is None:
                reg={'tipo':'pendente','sig':assinatura(im),'im':im,'score':score,
                     'amostra':i,'grupos':[g],'sigs':[assinatura(im)],'evento_id':g.get('evento_id')}
                print(f'  {i}/{len(grupos)}: OCR inicial falhou; revisando somente esta passagem...',flush=True)
                info2,im2=identificador.reanalisar_suspeita(reg)
                if chave_carta(info2) is not None:
                    info,im=info2,im2;score,_=qualidade_ocr(im);recuperada=True

            if chave_carta(info) is not None:
                idx=registrar_ocorrencia(info,im,score)
                qtd=cartas[idx].get('quantidade',1)
                prefixo='recuperada: ' if recuperada else ''
                print(f'  {i}/{len(grupos)}: {prefixo}{info["nome"]} | {info["numero"]} | {info["idioma"]} | qtd={qtd}',flush=True)
            else:
                suspeita={**SUSPEITA,'quantidade':1,'evento_id':g.get('evento_id')}
                _,nome=gravar(suspeita,im)
                print(f'  {i}/{len(grupos)}: não confirmada; salva como SUSPEITA: {nome}',flush=True)

    identificadas_copias=sum(int(x.get('quantidade',1) or 1) for x in cartas if x.get('nome')!='DESCONHECIDA')
    suspeitas_copias=sum(int(x.get('quantidade',1) or 1) for x in cartas if x.get('nome')=='DESCONHECIDA')
    total_copias=identificadas_copias+suspeitas_copias
    print(f'\nConcluído: {len(cartas)} registros; {total_copias} cópia(s) física(s); {identificadas_copias} identificada(s); {suspeitas_copias} suspeita(s).')
    if falhas:print(f'{falhas} vídeo(s) não puderam ser processados.')
    print('Saída: '+str(saida))
    print('A contagem vem exclusivamente das passagens: primeira carta + uma carta por troca detectada.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos',default=str(BASE/'videos'))
    parser.add_argument('--saida',default=str(BASE/'resultado'))
    args=parser.parse_args()
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
