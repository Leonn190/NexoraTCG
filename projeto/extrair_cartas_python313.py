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


def extrair_intervalos(video, fps_analise=5, diagnostico=None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('Não foi possível abrir o vídeo: '+video.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    salto = max(1,round(fps/fps_analise))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('  Localizando a área recorrente das cartas...',flush=True)
    ancora=aprender_posicao(cap,total)
    grupos, ativos = [], []
    quadro = 0
    ultimo_print = -1
    def encerrar(track):
        if track['n'] >= 4 and track['fim']-track['inicio'] >= .6:
            grupos.append(track)
    try:
        while cap.grab():
            i = quadro
            quadro += 1
            if i % salto:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            t = i/fps
            if total and int(i/total*10) != ultimo_print:
                ultimo_print = int(i/total*10)
                print(f'  Analisando: {min(100,round(i/total*100))}%',flush=True)
            deteccoes = []
            qs=detectar(frame)
            if ancora is not None:
                area_ancora=cv2.contourArea(ancora)
                qs=[q for q in qs if cv2.contourArea(q[1])>area_ancora*.72]
                proximos=[q for q in qs if np.linalg.norm(q[1].mean(0)-ancora.mean(0))<.18*math.sqrt(area_ancora)]
                if not proximos:
                    qs.append((.65,ancora,area_ancora))
            for geometria, quad, area in qs:
                crop = recortar(frame,quad)
                score,nitidez = qualidade(crop)
                if nitidez < 18:
                    continue
                deteccoes.append((quad,crop,assinatura(crop),score+geometria))
            usados=set()
            for quad,crop,sig,score in deteccoes:
                centro = quad.mean(0)
                melhores=[]
                for k,tr in enumerate(ativos):
                    if k in usados or t-tr['fim'] > .65:
                        continue
                    d=distancia(sig,tr['sig'])
                    pos=np.linalg.norm(centro-tr['centro']) / max(frame.shape[:2])
                    if d < .13 and distancia(sig,tr['referencia'])<.18 and distancia(sig[25:80],tr['referencia'][25:80])<.20 and pos < .13:
                        melhores.append((d,k))
                if melhores:
                    d,k=min(melhores);tr=ativos[k];usados.add(k)
                    tr['n']+=1;tr['fim']=t;tr['sig']=sig;tr['centro']=centro
                    # Quadro em movimento perde pontos mesmo quando o retângulo permanece visível.
                    score-=d*5
                else:
                    tr={'video':str(video),'inicio':t,'fim':t,'n':1,'sig':sig,'referencia':sig,'centro':centro,'frames':[]}
                    ativos.append(tr);usados.add(len(ativos)-1)
                # Candidatos distribuídos no tempo, sem guardar centenas de imagens em memória.
                slot=round(t/.45)
                existente=next((x for x in tr['frames'] if x[1]==slot),None)
                if existente is None or score>existente[0]:
                    if existente is not None:tr['frames'].remove(existente)
                    tr['frames'].append((score,slot,t,crop,quad))
                    tr['frames'].sort(key=lambda x:x[0],reverse=True)
                    del tr['frames'][8:]
            restantes=[]
            for tr in ativos:
                if t-tr['fim'] > .65:encerrar(tr)
                else:restantes.append(tr)
            ativos=restantes
        for tr in ativos:encerrar(tr)
    finally:
        cap.release()
    grupos.sort(key=lambda x:x['inicio'])
    if diagnostico:
        diagnostico.mkdir(parents=True,exist_ok=True)
        for k,g in enumerate(grupos):
            cv2.imwrite(str(diagnostico/f'{k:03d}_{g["inicio"]:.2f}_{g["fim"]:.2f}.jpg'),g['frames'][0][3])
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
        # Ampliação moderada ajuda o detector de caracteres pequenos, não melhora a foto salva.
        fator=max(1.,min(2.,800/im.shape[1]))
        img=cv2.resize(im,None,fx=fator,fy=fator)
        H,W=img.shape[:2]
        linhas=[]
        for reg,dy in [(img,0),(img[int(H*.77):],int(H*.77))]:
            resultado=self.ocr(reg)
            boxes=getattr(resultado, 'boxes', None)
            txts=getattr(resultado, 'txts', None)
            scores=getattr(resultado, 'scores', None)
            if boxes is None or txts is None or scores is None:
                continue
            for box,txt,conf in zip(boxes,txts,scores):
                y=(min(p[1] for p in box)+max(p[1] for p in box))/2+dy
                linhas.append((txt,float(conf),y/H))
        return linhas

    def idioma(self,linhas):
        txt=' '+unicodedata.normalize('NFKD',' '.join(x[0] for x in linhas)).lower()+' '
        if len(re.findall(r'[\u3040-\u30ff]',txt))>=4:
            return 'Japonês','ja'
        d={
            'BR':('pt', ['voce','pokemon','energia','dano','danos','ataque','seu','sua','fraqueza','resistencia','recuo','baralho','moeda','habilidade','basico','estagio']),
            'Inglês':('en',['your','opponent','damage','attack','energy','weakness','resistance','retreat','basic','ability','deck','coin','when','becomes','turn','ends','discard','attached','benched','prize','knocked','evolves']),
            'Alemão':('de',['dein','deines','schaden','energie','angriff','schwache','ruckzug','fahigkeit','gegner','mische','karte'])}
        limpo=unicodedata.normalize('NFKD',txt).encode('ascii','ignore').decode()
        palavras=set(re.findall(r'[a-z]+',limpo))
        scores=sorted([(sum(w in palavras for w in ws),label,code) for label,(code,ws) in d.items()],reverse=True)
        # 'pokemon' sozinho não identifica português.
        if scores[0][0]>=3 and scores[0][0]>=scores[1][0]+2:
            return scores[0][1],scores[0][2]
        return 'Outro',None

    def candidatos_numero(self,linhas):
        nums=[]
        for txt,conf,y in linhas:
            if y<.78 or conf<.83:continue
            t=unicodedata.normalize('NFKC',txt).upper().replace(' ','').replace('／','/')
            for m in re.finditer(r'(?<![A-Z0-9])((?:[A-Z]{1,3})?\d{1,4}[A-Z]?/(?:[A-Z]{1,3})?\d{1,4})(?!\d)',t):
                nums.append((m.group(1),conf))
            for m in re.finditer(r'(?<![A-Z0-9])((?:XY|SM|SWSH|SVP|BW|DP|HGSS)\d{1,4})(?!\d)',t):
                nums.append((m.group(1),conf))
        return list(dict.fromkeys(nums))

    def identificar_frame(self,im):
        linhas=self.ler(im)
        idioma,lingua=self.idioma(linhas)
        if lingua is None:return None,linhas
        numeros=self.candidatos_numero(linhas)
        titulos=[]
        for txt,conf,y in linhas:
            if .01<y<.19 and conf>=.78:
                t=re.sub(r'\b(?:BASIC|BÁSICO|BASICO|ESTÁGIO|STAGE|HP|PS|KP)\b','',txt,flags=re.I)
                t=re.sub(r'\d{2,3}\s*$','',t).strip()
                t=normalizar(t)
                t=re.sub(r'^(mega|basic|basico)','',t)
                t=re.sub(r'(?:ps|hp|kp)\d+$','',t)
                titulos.append(t)
        if not numeros or not titulos:return None,linhas
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
        if not matches:return None,linhas
        if len(matches)>1 and matches[0][2]!=matches[1][2] and matches[0][0]-matches[1][0]<.07:
            # Várias impressões com o mesmo nome/número continuam ambíguas.
            if matches[0][1]!=matches[1][1]:return None,linhas
        return matches[0][1],linhas

    def identificar(self,grupo):
        votos=collections.defaultdict(list)
        melhor=grupo['frames'][0][3]
        cap=None
        try:
            for tentativa,item in enumerate(grupo['frames'][:6]):
                im=item[3]
                info,linhas=self.identificar_frame(im)
                alternativas=[]
                if not info and len(item)>4 and not self.candidatos_numero(linhas):
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
                        if alternativas and item is grupo['frames'][0]:
                            melhor=alternativas[0]
                        for alt in alternativas:
                            info,ls=self.identificar_frame(alt)
                            if info:
                                im=alt
                                break
                if not info and (tentativa<2 or not any(y<.19 and c>.8 for _,c,y in linhas)):
                    girada=cv2.rotate(im,cv2.ROTATE_180)
                    info,_=self.identificar_frame(girada)
                    if info:im=girada
                if info:
                    chave=(info['nome'],info['numero'],info['idioma'])
                    votos[chave].append((item[0],im))
                    if len(votos[chave])>=2:
                        escolhido=max(votos[chave],key=lambda v:v[0])[1]
                        return info,escolhido
            return dict(DESCONHECIDA),melhor
        finally:
            if cap is not None:cap.release()


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
    # Uma nova execução nunca sobrescreve fotografias de uma execução anterior.
    if (saida/'cartas.json').exists():
        saida=saida.with_name(saida.name+'_'+time.strftime('%Y%m%d_%H%M%S'))
        n=2
        while saida.exists():
            saida=saida.with_name(saida.name+'_'+str(n));n+=1
    imagens=saida/'imagens';imagens.mkdir(parents=True,exist_ok=True)
    cartas=[];vistos=[];falhas=0
    json_atomico(saida/'cartas.json',{'cartas':cartas})
    for video in videos:
        print('\nVídeo: '+video.name,flush=True)
        try:
            grupos=extrair_intervalos(video,args.fps)
        except Exception as e:
            falhas+=1;print('  Erro: '+str(e),flush=True);continue
        print(f'  {len(grupos)} intervalos estáveis. Conferindo leituras...',flush=True)
        for i,g in enumerate(grupos,1):
            info,im=identificador.identificar(g)
            sig=assinatura(im)
            desc=descritor(im)
            key=(info['nome'],info['numero'],info['idioma'])
            repetido=False
            substituir=None
            for j,(oldkey,oldsig,olddesc) in enumerate(vistos):
                d=distancia(sig,oldsig)
                desconhecida=info['nome']=='DESCONHECIDA' or oldkey[0]=='DESCONHECIDA'
                igual=(key==oldkey and info['nome']!='DESCONHECIDA' and (d<.15 or mesma_imagem(desc,olddesc)))
                igual=igual or (desconhecida and (d<.035 or mesma_imagem(desc,olddesc)))
                if igual:
                    if oldkey[0]=='DESCONHECIDA' and info['nome']!='DESCONHECIDA':substituir=j
                    else:repetido=True
                    break
            if repetido:
                print(f'  {i}/{len(grupos)}: repetida ignorada',flush=True);continue
            indice=substituir if substituir is not None else len(cartas)
            nome=f'{indice+1:04d}_{nome_arquivo(info["nome"])}_{nome_arquivo(info["numero"])}_{nome_arquivo(info["idioma"])}.jpg'
            ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,96])
            if not ok:raise RuntimeError('Falha ao codificar fotografia')
            (imagens/nome).write_bytes(buf.tobytes())
            antigo=None
            if substituir is not None:
                antigo=cartas[indice]['arquivo']
                cartas[indice]={'arquivo':nome,**info}
                vistos[indice]=(key,sig,desc)
            else:
                cartas.append({'arquivo':nome,**info});vistos.append((key,sig,desc))
            json_atomico(saida/'cartas.json',{'cartas':cartas})
            if antigo and antigo!=nome:(imagens/antigo).unlink(missing_ok=True)
            print(f'  {i}/{len(grupos)}: {info["nome"]} | {info["numero"]} | {info["idioma"]}',flush=True)
    conhecidas=sum(x['nome']!='DESCONHECIDA' for x in cartas)
    print(f'\nConcluído: {len(cartas)} imagens; {conhecidas} identificadas; {len(cartas)-conhecidas} sem confirmação.')
    if falhas:print(f'{falhas} vídeo(s) não puderam ser processados.')
    print('Saída: '+str(saida))
    print('Reflexos que ocultam texto em todos os frames não podem ser recuperados.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos',default=str(BASE/'videos'))
    parser.add_argument('--saida',default=str(BASE/'resultado'))
    parser.add_argument('--fps',type=float,default=5.,help='Frames analisados por segundo (padrão: 5).')
    args=parser.parse_args()
    if not 1<=args.fps<=15:parser.error('--fps deve estar entre 1 e 15')
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
