import json
import re
import shutil
import statistics
import sys
import time
import unicodedata
from collections import defaultdict, deque
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PASTA_IMAGENS = BASE_DIR / "imagens"
PASTA_BACKUPS = BASE_DIR / "backups"
ARQUIVO_CARTAS = BASE_DIR / "cartas.json"
ARQUIVO_LEON = BASE_DIR / "leon.json"
PARCIAL_CARTAS = BASE_DIR / "cartas.parcial.json"
PARCIAL_LEON = BASE_DIR / "leon.parcial.json"

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

# Intervalo MÍNIMO entre QUALQUER request para a MYP.
# Se quiser ser ainda mais conservador, aumente para 2.5 ou 3.0.
INTERVALO_MINIMO_REQUESTS = 1.8

# A MYP pagina as ofertas de cartas com muitos anúncios.
# True = tenta seguir as páginas extras para melhorar as estatísticas.
COLETAR_PAGINAS_EXTRAS = True
MAX_PAGINAS_EXTRAS_POR_CARTA = 3

# Retry para bloqueios/erros temporários.
MAX_TENTATIVAS_REQUEST = 7
ESPERA_INICIAL_429 = 8
ESPERA_MAXIMA_429 = 120
STATUS_REPETIR = {429, 500, 502, 503, 504}


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

ESTADOS_VALIDOS = ("M", "NM", "SP", "MP", "HP", "DM", "D")
ESTADO_PARA_MYP = {"D": "DM"}

IDIOMAS = {
    "portugues": "Português",
    "português": "Português",
    "pt-br": "Português",
    "pt_br": "Português",
    "brazilian portuguese": "Português",
    "ingles": "Inglês",
    "inglês": "Inglês",
    "english": "Inglês",
    "en-us": "Inglês",
    "en_us": "Inglês",
    "japones": "Japonês",
    "japonês": "Japonês",
    "japanese": "Japonês",
    "chines simplificado": "Chinês Simplificado",
    "chinês simplificado": "Chinês Simplificado",
    "simplified chinese": "Chinês Simplificado",
    "chines tradicional": "Chinês Tradicional",
    "chinês tradicional": "Chinês Tradicional",
    "traditional chinese": "Chinês Tradicional",
    "espanhol": "Espanhol",
    "spanish": "Espanhol",
    "frances": "Francês",
    "francês": "Francês",
    "french": "Francês",
    "alemao": "Alemão",
    "alemão": "Alemão",
    "german": "Alemão",
    "italiano": "Italiano",
    "italian": "Italiano",
    "coreano": "Coreano",
    "korean": "Coreano",
    "chines": "Chinês",
    "chinês": "Chinês",
    "chinese": "Chinês",
    "suico": "Suíço",
    "suíço": "Suíço",
    "swiss": "Suíço",
}


# Controle global do rate limit.
_ULTIMO_REQUEST = 0.0


# ============================================================
# UTILIDADES
# ============================================================

def sem_acento(valor: str) -> str:
    valor = str(valor or "")
    return "".join(
        c for c in unicodedata.normalize("NFKD", valor)
        if not unicodedata.combining(c)
    )


def texto_normalizado(valor: str) -> str:
    return re.sub(r"\s+", " ", str(valor or "")).strip()


def chave_texto(valor: str) -> str:
    return sem_acento(texto_normalizado(valor)).lower()


def moeda_para_float(valor: str) -> float:
    valor = str(valor).replace("R$", "").replace("US$", "")
    valor = valor.replace("\xa0", "").replace(" ", "").strip()

    if "," in valor and "." in valor:
        valor = valor.replace(".", "").replace(",", ".")
    elif "," in valor:
        valor = valor.replace(",", ".")
    elif "." in valor:
        partes = valor.split(".")
        if not (len(partes) == 2 and len(partes[1]) == 2):
            valor = valor.replace(".", "")

    return float(valor)


def extrair_valores_reais(texto: str) -> list[float]:
    encontrados = re.findall(
        r"R\$\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})|"
        r"R\$\s*\d+(?:[.,]\d{2})",
        str(texto),
        flags=re.I,
    )
    return [moeda_para_float(x) for x in encontrados]


def resumo_precos(precos: list[float]) -> dict | None:
    valores = [float(x) for x in precos if x is not None]
    if not valores:
        return None
    return {
        "menor": round(min(valores), 2),
        "media": round(statistics.mean(valores), 2),
        "mediana": round(statistics.median(valores), 2),
        "maior": round(max(valores), 2),
    }


def salvar_json(caminho: Path, dados) -> None:
    caminho.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def carregar_lista_json(caminho: Path) -> list[dict]:
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    if not isinstance(dados, list):
        raise ValueError(f"{caminho.name} precisa ter uma lista na raiz.")
    if not all(isinstance(x, dict) for x in dados):
        raise ValueError(f"Todos os itens de {caminho.name} precisam ser objetos.")
    return dados


def carregar_json_se_existir(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    try:
        return carregar_lista_json(caminho)
    except Exception:
        return []


def criar_backup(caminho: Path) -> Path | None:
    if not caminho.exists():
        return None
    PASTA_BACKUPS.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = PASTA_BACKUPS / f"{caminho.stem}.backup_{carimbo}{caminho.suffix}"
    shutil.copy2(caminho, destino)
    return destino


def nome_arquivo_seguro(nome: str) -> str:
    nome = re.sub(r'[\\/:*?"<>|]+', "_", str(nome))
    nome = re.sub(r"\s+", " ", nome).strip()
    return nome[:180] or "carta"


def normalizar_estado(estado: str | None) -> str | None:
    if not estado:
        return None
    estado = sem_acento(str(estado)).upper().strip()
    estado = ESTADO_PARA_MYP.get(estado, estado)
    return estado if estado in {"M", "NM", "SP", "MP", "HP", "DM"} else None


def normalizar_idioma(idioma: str | None) -> str | None:
    if not idioma:
        return None
    k = chave_texto(idioma)
    if k in IDIOMAS:
        return IDIOMAS[k]
    for chave, nome in IDIOMAS.items():
        if chave in k:
            return nome
    return texto_normalizado(idioma)


def proximo_id(existentes: list[dict]) -> int:
    maior = 0
    for carta in existentes:
        m = re.search(r"(\d+)$", str(carta.get("id", "")))
        if m:
            maior = max(maior, int(m.group(1)))
    return maior + 1


def gerar_id(numero: int) -> str:
    return f"card_{numero:06d}"


def id_produto_myp(url: str) -> str | None:
    m = re.search(r"/produto/(\d+)", str(url or ""))
    return m.group(1) if m else None


def caminho_imagem_por_id(card_id: str, extensao: str = ".jpg") -> Path:
    return PASTA_IMAGENS / f"{nome_arquivo_seguro(card_id)}{extensao}"


def caminho_relativo(caminho: Path) -> str:
    try:
        return str(caminho.resolve().relative_to(BASE_DIR.resolve()))
    except Exception:
        return str(caminho.resolve())


# ============================================================
# REQUESTS / 429
# ============================================================

def segundos_retry_after(resposta: requests.Response) -> float | None:
    valor = resposta.headers.get("Retry-After")
    if not valor:
        return None

    valor = valor.strip()
    if valor.isdigit():
        return max(0.0, float(valor))

    try:
        data = parsedate_to_datetime(valor)
        agora = datetime.now(data.tzinfo) if data.tzinfo else datetime.now()
        return max(0.0, (data - agora).total_seconds())
    except Exception:
        return None


def esperar_rate_limit() -> None:
    global _ULTIMO_REQUEST
    agora = time.monotonic()
    falta = INTERVALO_MINIMO_REQUESTS - (agora - _ULTIMO_REQUEST)
    if falta > 0:
        time.sleep(falta)


def get_com_retry(
    sessao: requests.Session,
    url: str,
    *,
    timeout: int = 30,
) -> requests.Response:
    global _ULTIMO_REQUEST
    ultimo_erro = None

    for tentativa in range(1, MAX_TENTATIVAS_REQUEST + 1):
        esperar_rate_limit()

        try:
            resposta = sessao.get(url, headers=HEADERS, timeout=timeout)
            _ULTIMO_REQUEST = time.monotonic()

            if resposta.status_code not in STATUS_REPETIR:
                resposta.raise_for_status()
                return resposta

            if tentativa >= MAX_TENTATIVAS_REQUEST:
                resposta.raise_for_status()

            if resposta.status_code == 429:
                espera = segundos_retry_after(resposta)
                if espera is None:
                    espera = min(
                        ESPERA_INICIAL_429 * (2 ** (tentativa - 1)),
                        ESPERA_MAXIMA_429,
                    )
                print(f"  429 -> {espera:.0f}s", flush=True)
            else:
                espera = min(5 * tentativa, 30)
                print(f"  HTTP {resposta.status_code} -> {espera:.0f}s", flush=True)

            time.sleep(espera)

        except (requests.Timeout, requests.ConnectionError) as e:
            ultimo_erro = e
            _ULTIMO_REQUEST = time.monotonic()
            if tentativa >= MAX_TENTATIVAS_REQUEST:
                raise
            espera = min(5 * tentativa, 30)
            print(f"  conexão -> {espera:.0f}s", flush=True)
            time.sleep(espera)

    if ultimo_erro:
        raise ultimo_erro
    raise RuntimeError("Não foi possível concluir a requisição.")


# ============================================================
# PARSER MYP
# ============================================================

def achar_titulo(soup: BeautifulSoup, texto: str):
    alvo = chave_texto(texto)
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        if alvo in chave_texto(tag.get_text(" ", strip=True)):
            return tag
    return None


def elementos_entre_titulos(inicio, textos_fim):
    fins = [chave_texto(x) for x in textos_fim]
    for el in inicio.find_all_next():
        if el is inicio:
            continue
        if getattr(el, "name", None) in ("h1", "h2", "h3", "h4"):
            txt = chave_texto(el.get_text(" ", strip=True))
            if any(fim in txt for fim in fins):
                break
        yield el


def extrair_estado_anuncio(texto: str) -> str | None:
    # Pega a primeira condição explícita; isso evita usar um "NM" que esteja
    # só dentro do comentário de um anúncio cuja condição real é MP/SP.
    for m in re.finditer(r"(?<![A-Za-z])(NM|SP|MP|HP|DM|D|M)(?![A-Za-z])", str(texto), re.I):
        estado = normalizar_estado(m.group(1))
        if estado:
            return estado
    return None




def extrair_estado_elemento(el, texto: str) -> str | None:
    celulas = [texto_normalizado(x.get_text(" ", strip=True)) for x in el.find_all(["td", "th"])]
    if len(celulas) >= 3:
        # Ignora vendedor (primeira célula) e procura a condição no começo da
        # célula. Comentários como "MP NM na frente" continuam sendo MP.
        for candidato in celulas[1:]:
            if "R$" in candidato or re.search(r"\b\d+\s*un\.", candidato, re.I):
                continue
            m = re.match(r"^\s*(NM|SP|MP|HP|DM|D|M)(?=\s|$|[-+/])", candidato, re.I)
            if m:
                return normalizar_estado(m.group(1))
    return extrair_estado_anuncio(texto)

def detectar_idioma_texto(texto: str) -> str | None:
    k = chave_texto(texto)
    for chave, nome in IDIOMAS.items():
        if chave in k:
            return nome
    return None


def extrair_idioma_elemento(el) -> str | None:
    # Primeiro tenta texto visível.
    idioma = detectar_idioma_texto(el.get_text(" ", strip=True))
    if idioma:
        return idioma

    # Depois atributos que normalmente carregam tooltip/flag/data-language.
    for tag in [el] + list(el.find_all(True)):
        for attr, valor in getattr(tag, "attrs", {}).items():
            attr_k = chave_texto(attr)
            if isinstance(valor, list):
                valor = " ".join(map(str, valor))
            valor = str(valor)

            if any(x in attr_k for x in ("idioma", "language", "lang", "title", "alt")):
                idioma = detectar_idioma_texto(valor)
                if idioma:
                    return idioma

                # Códigos curtos só são aceitos em atributo que indique idioma.
                codigo = valor.strip().lower().replace("_", "-")
                codigos = {
                    "pt": "Português", "pt-br": "Português",
                    "en": "Inglês", "en-us": "Inglês",
                    "ja": "Japonês", "jp": "Japonês",
                    "es": "Espanhol", "fr": "Francês",
                    "de": "Alemão", "it": "Italiano",
                    "ko": "Coreano", "zh": "Chinês",
                }
                if codigo in codigos:
                    return codigos[codigo]

    return None


def extrair_variante_anuncio(el, texto: str) -> str | None:
    celulas = [texto_normalizado(x.get_text(" ", strip=True)) for x in el.find_all(["td", "th"])]

    # Na tabela da MYP, a coluna depois do vendedor costuma ser a variante.
    if len(celulas) >= 4:
        for candidato in celulas[1:3]:
            ck = chave_texto(candidato)
            if not candidato:
                continue
            if "r$" in candidato.lower() or re.search(r"\b\d+\s*un\.", candidato, re.I):
                continue
            if extrair_estado_anuncio(candidato):
                continue
            if len(candidato) <= 80:
                return candidato

    # Fallback por palavras conhecidas.
    padroes = [
        "Masterball Foil", "Master Ball Foil", "Pokeball Foil", "Pokéball Foil",
        "Reverse Foil", "Reverse Holo", "Cracked Ice Holo", "Cosmos Holo",
        "Tinsel Holo", "Holo + Stamp", "Foil - Shiny Vault", "Foil - Radiante",
        "Foil - Shining", "Foil - Arte especial", "Foil - Full Art Promo",
        "Foil - Full Art", "Full-Art", "Full Art", "Promo", "Foil", "Holo",
        "Jumbo", "Oversized", "Gigante",
    ]
    for p in padroes:
        if chave_texto(p) in chave_texto(texto):
            return p
    return None


def normalizar_foil(variante: str | None) -> str:
    if not variante:
        return "Normal"
    k = chave_texto(variante)
    if k in {"nao", "não", "normal", "regular"}:
        return "Normal"
    if "master" in k and "ball" in k:
        return "Masterball Foil"
    if "pokeball" in k or "poke ball" in k:
        return "Pokeball Foil"
    if "reverse" in k:
        return "Reverse Foil"
    if "cracked ice" in k:
        return "Cracked Ice Holo"
    if "cosmos" in k:
        return "Cosmos Holo"
    if "tinsel" in k:
        return "Tinsel Holo"
    if "stamp" in k:
        return texto_normalizado(variante)
    if "radiante" in k:
        return "Foil - Radiante"
    if "shiny vault" in k:
        return "Foil - Shiny Vault"
    if "shining" in k:
        return "Foil - Shining"
    if "foil" in k or "holo" in k:
        return texto_normalizado(variante)
    return "Normal"


def detectar_jumbo(variante: str | None, texto: str = "") -> bool:
    k = chave_texto(f"{variante or ''} {texto}")
    return any(x in k for x in ("jumbo", "oversized", "gigante", "oversize"))


def elemento_parece_anuncio(el) -> bool:
    texto = texto_normalizado(el.get_text(" ", strip=True))
    return (
        "R$" in texto
        and re.search(r"\b\d+\s*un\.", texto, re.I) is not None
    )


def extrair_anuncios(soup: BeautifulSoup) -> list[dict]:
    inicio = achar_titulo(soup, "Lojistas e Certificados")
    if not inicio:
        # Algumas cartas podem não ter lojistas certificados e mostrar só demais vendedores.
        inicio = achar_titulo(soup, "Demais vendedores")
    if not inicio:
        return []

    elementos = list(elementos_entre_titulos(
        inicio,
        ["Outras Edições", "Produtos relacionados", "Avaliações", "Outros Produtos"],
    ))

    # Preferimos TRs porque evitam wrappers duplicando anúncios.
    trs = [el for el in elementos if getattr(el, "name", None) == "tr" and elemento_parece_anuncio(el)]

    candidatos = trs
    if not candidatos:
        candidatos = []
        for el in elementos:
            if getattr(el, "name", None) not in ("li", "div"):
                continue
            if not elemento_parece_anuncio(el):
                continue
            # Só pega elementos folha: se um filho já parece anúncio, o pai é wrapper.
            if any(
                elemento_parece_anuncio(filho)
                for filho in el.find_all(["tr", "li", "div"], recursive=False)
            ):
                continue
            candidatos.append(el)

    anuncios = []
    vistos = set()

    for el in candidatos:
        texto = texto_normalizado(el.get_text(" ", strip=True))
        precos = extrair_valores_reais(texto)
        if not precos:
            continue

        preco = precos[-1]  # promoção = último preço mostrado
        estado = extrair_estado_elemento(el, texto)
        variante = extrair_variante_anuncio(el, texto)
        idioma = extrair_idioma_elemento(el)
        jumbo = detectar_jumbo(variante, texto)

        # Primeiro texto/célula costuma ser o vendedor; serve para deduplicação.
        celulas = [texto_normalizado(x.get_text(" ", strip=True)) for x in el.find_all(["td", "th"])]
        vendedor = celulas[0] if celulas else texto[:80]

        anuncio = {
            "preco": round(preco, 2),
            "estado": estado,
            "idioma": idioma,
            "foil": normalizar_foil(variante),
            "jumbo": jumbo,
            "variante": texto_normalizado(variante) if variante else None,
            "_vendedor": vendedor,
        }

        chave = (
            chave_texto(vendedor),
            anuncio["preco"],
            anuncio["estado"],
            anuncio["idioma"],
            anuncio["foil"],
            anuncio["jumbo"],
            chave_texto(texto),
        )
        if chave in vistos:
            continue
        vistos.add(chave)
        anuncios.append(anuncio)

    return anuncios


def extrair_numero_ofertas(soup: BeautifulSoup) -> int | None:
    texto = soup.get_text(" ", strip=True)
    m = re.search(r"Ofertas\s*\(\s*(\d+)\s*\)", texto, re.I)
    return int(m.group(1)) if m else None


def extrair_imagem(soup: BeautifulSoup, url_pagina: str) -> str | None:
    for attrs in (
        {"property": "og:image"},
        {"name": "twitter:image"},
        {"property": "twitter:image"},
    ):
        meta = soup.find("meta", attrs=attrs)
        if meta and meta.get("content"):
            return urljoin(url_pagina, meta["content"])

    # Fallback conservador.
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue
        desc = chave_texto(" ".join([
            img.get("alt", ""),
            " ".join(img.get("class", [])) if img.get("class") else "",
        ]))
        if any(x in desc for x in ("logo", "avatar", "icon")):
            continue
        return urljoin(url_pagina, src)
    return None


def texto_antes_dos_vendedores(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    fim = achar_titulo(soup, "Lojistas e Certificados") or achar_titulo(soup, "Demais vendedores")
    if not h1 or not fim:
        return ""

    partes = []
    for el in h1.find_all_next():
        if el is fim:
            break
        if getattr(el, "name", None) in {"script", "style"}:
            continue
        if getattr(el, "name", None) in {"h1", "h2", "h3", "h4", "p", "div", "span", "a"}:
            txt = texto_normalizado(el.get_text(" ", strip=True))
            if txt:
                partes.append(txt)
    return " ".join(partes)


def extrair_preco_tcgplayer(soup: BeautifulSoup) -> float | None:
    # 1) Tenta achar explicitamente um elemento/atributo ligado a TCGPlayer.
    candidatos = []
    for tag in soup.find_all(True):
        attrs = " ".join(
            f"{k}={' '.join(v) if isinstance(v, list) else v}"
            for k, v in tag.attrs.items()
        )
        marcador = chave_texto(f"{tag.name} {attrs} {tag.get_text(' ', strip=True)[:120]}")
        if "tcgplayer" not in marcador and "tcg player" not in marcador:
            continue

        atual = tag
        for _ in range(4):
            if atual is None:
                break
            valores = extrair_valores_reais(atual.get_text(" ", strip=True))
            if valores:
                candidatos.extend(valores)
                break
            atual = getattr(atual, "parent", None)

    if candidatos:
        # Evita zero/duplicatas bizarras, preservando a primeira referência próxima.
        for valor in candidatos:
            if valor > 0:
                return round(valor, 2)

    # 2) Fallback para o layout atual da MYP: antes das ofertas aparecem os
    # cards-resumo; normalmente o 2º valor em R$ é a referência TCGPlayer.
    # Só usamos este fallback se a página realmente contém marca TCGPlayer.
    html_k = chave_texto(str(soup))
    if "tcgplayer" in html_k or "tcg player" in html_k:
        valores = extrair_valores_reais(texto_antes_dos_vendedores(soup))
        # Remove repetições consecutivas produzidas por wrappers.
        limpos = []
        for v in valores:
            if not limpos or v != limpos[-1]:
                limpos.append(v)
        if len(limpos) >= 2:
            return round(limpos[1], 2)

    return None


def extrair_dados_carta(soup: BeautifulSoup, fonte: dict) -> dict:
    texto = soup.get_text("\n", strip=True)

    h1 = soup.find("h1")
    titulo = texto_normalizado(h1.get_text(" ", strip=True)) if h1 else ""

    numeracao = texto_normalizado(str(fonte.get("numeracao", "")))

    # O JSON fonte já possui o nome correto da carta. A MYP às vezes monta um
    # H1 duplicado, por exemplo:
    # "Mega Latias ex (163/132) Mega Latias ex - 163/132".
    # Por isso o nome do inventário é a fonte canônica; o H1 é só fallback.
    nome_fonte = texto_normalizado(fonte.get("nome", ""))
    if nome_fonte:
        nome = nome_fonte
    else:
        nome = titulo
        if numeracao:
            # Fica com tudo antes da primeira ocorrência da numeração, removendo
            # parênteses/hífens que sobrariam.
            pos = nome.find(numeracao)
            if pos >= 0:
                nome = nome[:pos].rstrip(" (-–—").strip()
        else:
            m_num = re.search(r"\(([^()]+)\)", titulo)
            if m_num:
                numeracao = m_num.group(1).strip()
                nome = titulo[:m_num.start()].strip()

    colecao = None
    sigla = None
    m = re.search(r"(?:^|\n)Edição\s*\n?\s*([^\n]+?)\s*\(([^()\n]+)\)", texto, re.I)
    if not m:
        m = re.search(r"Edição\s+([^\n]+?)\s*\(([^()\n]+)\)", texto, re.I)
    if m:
        colecao = texto_normalizado(m.group(1))
        sigla = texto_normalizado(m.group(2))

    if not sigla:
        m_codigo = re.search(r"Código\s+pokemon_([^_\s]+)_", texto, re.I)
        if m_codigo:
            sigla = m_codigo.group(1).strip()

    raridade = None
    m = re.search(r"(?:^|\n)Raridade\s*\n?\s*([^\n]+)", texto, re.I)
    if m:
        raridade = texto_normalizado(m.group(1))

    ano = None
    m = re.search(r"Data de lançamento\s*\n?\s*(\d{2}/\d{2}/(\d{4}))", texto, re.I)
    if m:
        ano = int(m.group(2))
    else:
        # Fallback para descrições importadas que trazem algo como "Aug 3, 2022".
        m_en = re.search(
            r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2},\s+(\d{4})\b",
            texto,
            re.I,
        )
        if m_en:
            ano = int(m_en.group(2))

    return {
        "nome": nome or fonte.get("nome"),
        "numeracao": numeracao or fonte.get("numeracao"),
        "colecao": colecao,
        "sigla_colecao": sigla,
        "ano": ano,
        "raridade": raridade,
    }


def determinar_era(ano: int | None, sigla: str | None, colecao: str | None) -> str | None:
    if not ano and not sigla and not colecao:
        return None

    s = chave_texto(sigla or "")
    c = chave_texto(colecao or "")

    # Era Megaevolução moderna. Mantemos códigos atuais conhecidos e um
    # fallback por nome para não classificar sets SV de 2025 incorretamente.
    if s.upper() in {"MEG", "PFL", "ASC", "30C"} or "mega evol" in c or "megaevol" in c:
        return "Megaevolução"

    if ano is None:
        return None
    if ano >= 2026:
        return "Megaevolução"
    if ano >= 2023:
        return "Escarlate e Violeta"
    if ano >= 2020:
        return "Espada e Escudo"
    if ano >= 2017:
        return "Sol e Lua"
    if ano >= 2013:
        return "XY"
    if ano >= 2011:
        return "Preto e Branco"
    if ano >= 2010:
        return "HeartGold & SoulSilver"
    if ano >= 2009:
        return "Platina"
    if ano >= 2007:
        return "Diamante e Pérola"
    if ano >= 2003:
        return "EX"
    return "Clássica"


def raridade_permite_variantes_foil(nome: str, raridade: str | None, tipo_carta: str | None = None) -> bool:
    # Regra deliberadamente conservadora: cartas cuja própria raridade/mecânica
    # já define o acabamento não ganham multiplicador de foil.
    k = chave_texto(f"{nome} {raridade or ''} {tipo_carta or ''}")

    frases_bloqueadas = [
        "radiante", "shining", "luminescente",
        "ilustracao rara", "illustration rare", "special illustration",
        "arte rara", "ultra rara", "hiper rara", "hyper rare", "secret rare",
        "full art",
    ]
    if any(x in k for x in frases_bloqueadas):
        return False

    # Mecânicas como EX/GX/V/Mega precisam aparecer como palavra/sufixo real.
    # Assim um Pokémon chamado "Exeggutor" não é confundido com uma carta EX.
    if re.search(
        r"(?:^|[\s-])(gx|ex|v|vmax|vstar|v-union|mega|turbo|break)(?:$|[\s-])",
        k,
    ):
        return False

    return True


def urls_paginas_ofertas(soup: BeautifulSoup, url_base: str) -> set[str]:
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "estoque-cert-page=" in href or "estoque-outros-page=" in href:
            urls.add(urljoin(url_base, href))
    return urls


def coletar_paginas_myp(sessao: requests.Session, url: str) -> tuple[list[BeautifulSoup], str]:
    resposta = get_com_retry(sessao, url)
    url_final = resposta.url
    primeira = BeautifulSoup(resposta.text, "html.parser")
    paginas = [primeira]

    if not COLETAR_PAGINAS_EXTRAS:
        return paginas, url_final

    fila = deque(sorted(urls_paginas_ofertas(primeira, url_final)))
    visitadas = {url_final}
    extras = 0

    while fila and extras < MAX_PAGINAS_EXTRAS_POR_CARTA:
        proxima = fila.popleft()
        if proxima in visitadas:
            continue
        visitadas.add(proxima)

        try:
            r = get_com_retry(sessao, proxima)
        except Exception:
            continue

        extras += 1
        sp = BeautifulSoup(r.text, "html.parser")
        paginas.append(sp)

        for nova in sorted(urls_paginas_ofertas(sp, r.url)):
            if nova not in visitadas:
                fila.append(nova)

    return paginas, url_final


def deduplicar_anuncios(anuncios: list[dict]) -> list[dict]:
    saida = []
    vistos = set()

    # Ao abrir paginação dos "demais vendedores", a MYP pode repetir na nova
    # página a seção de lojistas certificados que já vimos. A assinatura com
    # vendedor evita contar essas repetições duas vezes.
    for a in anuncios:
        chave = (
            chave_texto(a.get("_vendedor")),
            a.get("preco"), a.get("estado"), a.get("idioma"),
            a.get("foil"), a.get("jumbo"), a.get("variante"),
        )
        if chave in vistos:
            continue
        vistos.add(chave)
        saida.append(a)

    return saida


# ============================================================
# PREÇOS E MULTIPLICADORES
# ============================================================

def grupo_idioma(idioma: str | None) -> str | None:
    """Agrupa idiomas em BR, Estrangeiro e Oriental."""
    idioma = normalizar_idioma(idioma)
    if not idioma:
        return None

    k = chave_texto(idioma)
    if any(x in k for x in ("portugues", "brazil", "pt-br", "pt_br")):
        return "BR"

    if any(x in k for x in (
        "japones", "japanese",
        "chines", "chinese",
        "coreano", "korean",
    )):
        return "Oriental"

    return "Estrangeiro"


def resumo_anuncios(anuncios: list[dict]) -> dict | None:
    stats = resumo_precos([float(a["preco"]) for a in anuncios if a.get("preco") is not None])
    if not stats:
        return None
    stats["anuncios"] = len(anuncios)
    return stats


def anuncios_base(anuncios: list[dict], permite_foil: bool) -> list[dict]:
    """Preço principal = carta padrão, não jumbo, preferindo BR."""
    base = [a for a in anuncios if not a.get("jumbo")]

    if permite_foil:
        normais = [a for a in base if a.get("foil", "Normal") == "Normal"]
        if normais:
            base = normais

    br = [a for a in base if grupo_idioma(a.get("idioma")) == "BR"]
    desconhecido = [a for a in base if grupo_idioma(a.get("idioma")) is None]
    if br:
        base = br + desconhecido

    return base


def precos_por_estado(anuncios: list[dict]) -> dict:
    grupos = defaultdict(list)
    for a in anuncios:
        estado = normalizar_estado(a.get("estado"))
        if estado:
            grupos[estado].append(a)

    ordem = ["M", "NM", "SP", "MP", "HP", "DM"]
    saida = {}
    for estado in ordem:
        grupo = grupos.get(estado, [])
        if not grupo:
            continue
        stats = resumo_anuncios(grupo)
        if stats:
            saida[estado] = stats
    return saida


def _grupo_com_estados(anuncios: list[dict]) -> dict | None:
    stats = resumo_anuncios(anuncios)
    if not stats:
        return None
    por_estado = precos_por_estado(anuncios)
    if por_estado:
        stats["por_estado"] = por_estado
    return stats


def precos_por_idioma(anuncios: list[dict]) -> dict:
    """
    Sem multiplicador: grava os preços REAIS em três grupos.
    Grupo ausente na MYP simplesmente não entra no JSON.
    """
    grupos = defaultdict(list)
    for a in anuncios:
        g = grupo_idioma(a.get("idioma"))
        if g:
            grupos[g].append(a)

    saida = {}
    for g in ("BR", "Estrangeiro", "Oriental"):
        if not grupos.get(g):
            continue
        stats = _grupo_com_estados(grupos[g])
        if stats:
            saida[g] = stats
    return saida


def precos_por_foil(anuncios: list[dict], permite_foil: bool) -> dict:
    """Preços reais por tratamento foil, sem multiplicadores."""
    if not permite_foil:
        return {}

    grupos = defaultdict(list)
    for a in anuncios:
        if a.get("jumbo"):
            continue
        foil = a.get("foil", "Normal") or "Normal"
        grupos[foil].append(a)

    # Se só existe Normal, não há uma variante de foil útil para guardar.
    if not any(k != "Normal" for k in grupos):
        return {}

    saida = {}
    # Normal primeiro; demais em ordem alfabética para JSON estável.
    chaves = (["Normal"] if "Normal" in grupos else []) + sorted(k for k in grupos if k != "Normal")
    for foil in chaves:
        stats = _grupo_com_estados(grupos[foil])
        if stats:
            saida[foil] = stats
    return saida


def precos_por_tamanho(anuncios: list[dict]) -> dict:
    """Só cria a seção quando há pelo menos uma oferta jumbo."""
    jumbo = [a for a in anuncios if a.get("jumbo")]
    if not jumbo:
        return {}

    normal = [a for a in anuncios if not a.get("jumbo")]
    saida = {}
    if normal:
        stats = _grupo_com_estados(normal)
        if stats:
            saida["Normal"] = stats
    stats_jumbo = _grupo_com_estados(jumbo)
    if stats_jumbo:
        saida["Jumbo"] = stats_jumbo
    return saida


def filtrar_anuncios_do_dono(
    anuncios: list[dict],
    registro: dict,
    permite_foil: bool,
) -> list[dict]:
    """
    Usa apenas ofertas equivalentes ao exemplar do dono:
    estado + grupo de idioma + foil (quando aplicável) + tamanho.
    """
    estado = normalizar_estado(registro.get("estado"))
    grupo = grupo_idioma(registro.get("lingua") or registro.get("idioma"))
    foil_dono = normalizar_foil(registro.get("foil"))
    jumbo_dono = bool(registro.get("jumbo")) or detectar_jumbo(registro.get("foil"), str(registro))

    if not estado or not grupo:
        return []

    saida = []
    for a in anuncios:
        if normalizar_estado(a.get("estado")) != estado:
            continue
        if grupo_idioma(a.get("idioma")) != grupo:
            continue
        if bool(a.get("jumbo")) != jumbo_dono:
            continue
        if permite_foil and (a.get("foil", "Normal") or "Normal") != foil_dono:
            continue
        saida.append(a)
    return saida


def preco_para_dono_anuncios(
    anuncios: list[dict],
    registro: dict,
    permite_foil: bool,
) -> dict | None:
    return resumo_anuncios(filtrar_anuncios_do_dono(anuncios, registro, permite_foil))


def preco_para_dono_catalogo(catalogo: dict, registro: dict) -> dict | None:
    """Fallback para quando a coleta falha e só temos o catálogo salvo."""
    estado = normalizar_estado(registro.get("estado"))
    grupo = grupo_idioma(registro.get("lingua") or registro.get("idioma"))
    if not estado or not grupo:
        return None

    bloco = catalogo.get("precos_por_idioma", {}).get(grupo)
    if not bloco:
        return None
    stats = bloco.get("por_estado", {}).get(estado)
    if not stats:
        return None
    return {
        "menor": stats.get("menor"),
        "media": stats.get("media"),
        "mediana": stats.get("mediana"),
        "maior": stats.get("maior"),
        "anuncios": stats.get("anuncios"),
    }


# ============================================================
# COLETA DE UMA CARTA
# ============================================================

def baixar_imagem(
    sessao: requests.Session,
    url_imagem: str | None,
    card_id: str,
) -> str | None:
    if not url_imagem:
        return None

    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)

    extensao = Path(urlparse(url_imagem).path).suffix.lower()
    if extensao not in EXTENSOES_IMAGEM:
        extensao = ".jpg"

    caminho = caminho_imagem_por_id(card_id, extensao)

    # Reexecução não gasta outro request com imagem já salva.
    if caminho.is_file() and caminho.stat().st_size > 0:
        return caminho_relativo(caminho)

    resposta = get_com_retry(sessao, url_imagem)
    caminho.write_bytes(resposta.content)
    return caminho_relativo(caminho)


def coletar_carta_myp(
    fonte: dict,
    card_id: str,
    sessao: requests.Session,
) -> tuple[dict, list[dict], bool]:
    url = texto_normalizado(fonte.get("link", ""))
    if "mypcards.com" not in url.lower():
        raise ValueError("link MYP inválido")

    paginas, url_final = coletar_paginas_myp(sessao, url)
    primeira = paginas[0]

    dados = extrair_dados_carta(primeira, fonte)
    permite_foil = raridade_permite_variantes_foil(
        dados.get("nome", ""),
        dados.get("raridade"),
    )

    todos_anuncios = []
    for pagina in paginas:
        todos_anuncios.extend(extrair_anuncios(pagina))
    todos_anuncios = deduplicar_anuncios(todos_anuncios)

    base = anuncios_base(todos_anuncios, permite_foil)
    geral = resumo_anuncios(base)
    por_estado = precos_por_estado(base)

    idiomas = precos_por_idioma(todos_anuncios)
    foils = precos_por_foil(todos_anuncios, permite_foil)
    tamanhos = precos_por_tamanho(todos_anuncios)

    numero_ofertas = extrair_numero_ofertas(primeira)
    if numero_ofertas is None:
        numero_ofertas = len(todos_anuncios)

    url_imagem = extrair_imagem(primeira, url_final)
    try:
        imagem = baixar_imagem(sessao, url_imagem, card_id)
    except Exception:
        imagem = None

    carta = {
        "id": card_id,
        "nome": dados.get("nome"),
        "numeracao": dados.get("numeracao"),
        "colecao": dados.get("colecao"),
        "sigla_colecao": dados.get("sigla_colecao"),
        "era": determinar_era(dados.get("ano"), dados.get("sigla_colecao"), dados.get("colecao")),
        "ano": dados.get("ano"),
        "raridade": dados.get("raridade"),
        "menor_preco": geral.get("menor") if geral else None,
        "media": geral.get("media") if geral else None,
        "mediana": geral.get("mediana") if geral else None,
        "maior_preco": geral.get("maior") if geral else None,
        "precos_por_estado": por_estado,
        "preco_tcgplayer": extrair_preco_tcgplayer(primeira),
        "numero_anuncios_myp": numero_ofertas,
        "caminho_foto": imagem,
        "link_myp": url_final,
    }

    if idiomas:
        carta["precos_por_idioma"] = idiomas
    if foils:
        carta["precos_por_foil"] = foils
    if tamanhos:
        carta["precos_por_tamanho"] = tamanhos

    return carta, todos_anuncios, permite_foil


def criar_registro_leon(
    catalogo: dict,
    fonte: dict,
    anuncios: list[dict] | None = None,
    permite_foil: bool = False,
) -> dict:
    if anuncios is not None:
        precos = preco_para_dono_anuncios(anuncios, fonte, permite_foil)
    else:
        precos = preco_para_dono_catalogo(catalogo, fonte)

    return {
        "nome": fonte.get("nome") or catalogo.get("nome"),
        "id": catalogo.get("id"),
        "estado": fonte.get("estado"),
        "idioma": fonte.get("lingua") or fonte.get("idioma"),
        "menor_preco": precos.get("menor") if precos else None,
        "preco_medio": precos.get("media") if precos else None,
        "mediana_preco": precos.get("mediana") if precos else None,
        "maior_preco": precos.get("maior") if precos else None,
        "quantidade": fonte.get("quantidade", 1),
    }


def _fmt_preco(valor) -> str:
    if valor is None:
        return "--"
    try:
        return f"{float(valor):.2f}"
    except Exception:
        return "--"


def texto_preco_dono(registro_leon: dict) -> str:
    estado = texto_normalizado(registro_leon.get("estado")) or "?"
    grupo = grupo_idioma(registro_leon.get("idioma")) or "?"
    return (
        f"{estado} {grupo} | "
        f"{_fmt_preco(registro_leon.get('menor_preco'))} | "
        f"{_fmt_preco(registro_leon.get('preco_medio'))} | "
        f"{_fmt_preco(registro_leon.get('mediana_preco'))} | "
        f"{_fmt_preco(registro_leon.get('maior_preco'))}"
    )


# ============================================================
# IDs ESTÁVEIS
# ============================================================

def indexar_catalogo_existente(cartas: list[dict]):
    por_link = {}
    por_nome_numero = defaultdict(list)

    for c in cartas:
        link = texto_normalizado(c.get("link_myp", ""))
        if link:
            por_link[link] = c
            pid = id_produto_myp(link)
            if pid:
                por_link[f"mypid:{pid}"] = c

        chave = (chave_texto(c.get("nome")), chave_texto(c.get("numeracao")))
        por_nome_numero[chave].append(c)

    return por_link, por_nome_numero


def achar_existente(
    fonte: dict,
    por_link: dict,
    por_nome_numero: dict,
) -> dict | None:
    link = texto_normalizado(fonte.get("link", ""))
    if link in por_link:
        return por_link[link]

    pid = id_produto_myp(link)
    if pid and f"mypid:{pid}" in por_link:
        return por_link[f"mypid:{pid}"]

    chave = (chave_texto(fonte.get("nome")), chave_texto(fonte.get("numeracao")))
    candidatos = por_nome_numero.get(chave, [])
    if len(candidatos) == 1:
        return candidatos[0]
    return None


# ============================================================
# COLETAR
# ============================================================

def coletar_todas(caminho_fonte: Path) -> None:
    fonte = carregar_lista_json(caminho_fonte)
    catalogo_antigo = carregar_json_se_existir(ARQUIVO_CARTAS)
    por_link, por_nome_numero = indexar_catalogo_existente(catalogo_antigo)
    proximo = proximo_id(catalogo_antigo)

    # Agrupa pelo link para não consultar duas vezes a mesma carta.
    grupos = defaultdict(list)
    sem_link = []
    for reg in fonte:
        link = texto_normalizado(reg.get("link", ""))
        if link:
            pid = id_produto_myp(link)
            chave = f"myp:{pid}" if pid else link
            grupos[chave].append(reg)
        else:
            sem_link.append(reg)

    total = len(grupos)
    sessao = requests.Session()

    catalogo_final = []
    leon_final = []
    erros = []

    usados_ids = set()
    print(f"\nCOLETAR | {total} cartas únicas | {len(fonte)} registros do dono")
    print("preços: menor | média | mediana | maior\n")

    for i, (_, registros) in enumerate(grupos.items(), start=1):
        primeiro = registros[0]
        nome = primeiro.get("nome", "Sem nome")
        num = primeiro.get("numeracao", "?")

        existente = achar_existente(primeiro, por_link, por_nome_numero)
        if existente and existente.get("id"):
            card_id = existente["id"]
        else:
            card_id = gerar_id(proximo)
            proximo += 1

        try:
            carta, anuncios, permite_foil = coletar_carta_myp(primeiro, card_id, sessao)
            status = ""
        except Exception as e:
            if existente:
                carta = dict(existente)
                carta["erro_ultima_coleta"] = str(e)
                anuncios = None
                permite_foil = False
                status = "ANTIGO "
            else:
                erros.append({
                    "nome": nome,
                    "numeracao": num,
                    "link": primeiro.get("link"),
                    "erro": str(e),
                })
                print(f"[{i}/{total}] {nome} {num} | PULADA")
                continue

        if carta.get("id") not in usados_ids:
            catalogo_final.append(carta)
            usados_ids.add(carta.get("id"))

        registros_leon = []
        for reg in registros:
            r_leon = criar_registro_leon(carta, reg, anuncios, permite_foil)
            leon_final.append(r_leon)
            registros_leon.append(r_leon)

        if registros_leon:
            print(f"[{i}/{total}] {status}{nome} {num} | {texto_preco_dono(registros_leon[0])}")
            for extra in registros_leon[1:]:
                print(f"           {texto_preco_dono(extra)}")
        else:
            print(f"[{i}/{total}] {status}{nome} {num}")

        # Progresso separado dos arquivos finais: uma interrupção não destrói
        # uma coleta anterior completa.
        salvar_json(PARCIAL_CARTAS, catalogo_final)
        salvar_json(PARCIAL_LEON, leon_final)

    for reg in sem_link:
        erros.append({
            "nome": reg.get("nome"),
            "numeracao": reg.get("numeracao"),
            "link": None,
            "erro": "sem link MYP",
        })

    # Só substitui os JSONs oficiais ao concluir o percurso.
    salvar_json(ARQUIVO_CARTAS, catalogo_final)
    salvar_json(ARQUIVO_LEON, leon_final)

    for parcial in (PARCIAL_CARTAS, PARCIAL_LEON):
        if parcial.exists():
            parcial.unlink()

    if erros:
        salvar_json(BASE_DIR / "erros_coleta.json", erros)

    print(f"\nOK | cartas.json: {len(catalogo_final)} | leon.json: {len(leon_final)} | puladas: {len(erros)}")


# ============================================================
# CORTAR
# ============================================================

def resolver_caminho_foto(valor: str | None) -> Path | None:
    if not valor:
        return None
    p = Path(str(valor))
    return p if p.is_absolute() else BASE_DIR / p


def tem_imagem(carta: dict) -> bool:
    p = resolver_caminho_foto(carta.get("caminho_foto"))
    if p and p.is_file():
        return True

    card_id = carta.get("id")
    if not card_id or not PASTA_IMAGENS.exists():
        return False

    return any(
        caminho_imagem_por_id(card_id, ext).is_file()
        for ext in EXTENSOES_IMAGEM
    )


def cortar_sem_imagem(caminho_fonte: Path | None = None) -> None:
    if not ARQUIVO_CARTAS.exists() or not ARQUIVO_LEON.exists():
        print("\nFaltam cartas.json/leon.json. Rode COLETAR primeiro.")
        return

    cartas = carregar_lista_json(ARQUIVO_CARTAS)
    leon = carregar_lista_json(ARQUIVO_LEON)

    mantidas = [c for c in cartas if tem_imagem(c)]
    removidas = [c for c in cartas if not tem_imagem(c)]

    if not removidas:
        print("\nCORTAR | nada para remover")
        return

    ids_removidos = {c.get("id") for c in removidas}
    links_removidos = {
        texto_normalizado(c.get("link_myp"))
        for c in removidas
        if c.get("link_myp")
    }
    pids_removidos = {
        id_produto_myp(link) for link in links_removidos if id_produto_myp(link)
    }

    leon_mantido = [x for x in leon if x.get("id") not in ids_removidos]

    criar_backup(ARQUIVO_CARTAS)
    criar_backup(ARQUIVO_LEON)
    salvar_json(ARQUIVO_CARTAS, mantidas)
    salvar_json(ARQUIVO_LEON, leon_mantido)

    # Também corta o JSON-fonte legado. Assim a próxima coleta não traz a
    # carta apagada de volta. O link é usado só para identificar a fonte;
    # o ID do catálogo continua independente da MYP.
    fonte_removidas = 0
    if caminho_fonte and caminho_fonte.exists():
        try:
            fonte = carregar_lista_json(caminho_fonte)
            nova_fonte = []
            for reg in fonte:
                link = texto_normalizado(reg.get("link", ""))
                pid = id_produto_myp(link)
                remover = link in links_removidos or (pid and pid in pids_removidos)
                if remover:
                    fonte_removidas += 1
                else:
                    nova_fonte.append(reg)

            if fonte_removidas:
                criar_backup(caminho_fonte)
                salvar_json(caminho_fonte, nova_fonte)
        except Exception:
            pass

    salvar_json(BASE_DIR / "removidas_sem_imagem.json", removidas)

    print(
        f"\nCORTAR | cartas: -{len(removidas)} | leon: -{len(leon) - len(leon_mantido)}"
        + (f" | fonte: -{fonte_removidas}" if caminho_fonte else "")
    )


# ============================================================
# ENTRADA / MENU
# ============================================================

def localizar_json_fonte(argumento: str | None = None) -> Path:
    if argumento:
        p = Path(argumento.strip().strip('"')).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"JSON não encontrado: {p}")
        return p

    ignorar = {
        ARQUIVO_CARTAS.name,
        ARQUIVO_LEON.name,
        PARCIAL_CARTAS.name,
        PARCIAL_LEON.name,
        "erros_coleta.json",
        "removidas_sem_imagem.json",
    }
    arquivos = [
        p for p in BASE_DIR.glob("*.json")
        if p.name not in ignorar and not p.name.startswith("removidas_")
    ]

    if len(arquivos) == 1:
        return arquivos[0]

    if len(arquivos) > 1:
        print("\nJSON fonte:")
        for i, p in enumerate(sorted(arquivos), 1):
            print(f"{i} - {p.name}")
        while True:
            x = input("> ").strip()
            if x.isdigit() and 1 <= int(x) <= len(arquivos):
                return sorted(arquivos)[int(x) - 1]

    entrada = input("JSON fonte: ").strip().strip('"')
    p = Path(entrada).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"JSON não encontrado: {p}")
    return p


def menu() -> str:
    print("\n1 - COLETAR")
    print("2 - CORTAR")
    print("0 - SAIR")
    while True:
        x = input("> ").strip().lower()
        if x in {"1", "coletar", "c"}:
            return "coletar"
        if x in {"2", "cortar", "corta"}:
            return "cortar"
        if x in {"0", "sair", "exit"}:
            return "sair"


def main():
    acao = None
    argumento = None

    if len(sys.argv) >= 2:
        x = sys.argv[1].strip().lower()
        if x in {"coletar", "cortar", "corta"}:
            acao = "cortar" if x in {"cortar", "corta"} else "coletar"
            if len(sys.argv) >= 3:
                argumento = sys.argv[2]

    if acao is None:
        acao = menu()

    if acao == "sair":
        return

    try:
        caminho_fonte = localizar_json_fonte(argumento)
        if acao == "coletar":
            coletar_todas(caminho_fonte)
        else:
            cortar_sem_imagem(caminho_fonte)
    except KeyboardInterrupt:
        print("\nInterrompido. Os arquivos .parcial preservam o progresso desta execução.")
    except Exception as e:
        print(f"\nERRO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
