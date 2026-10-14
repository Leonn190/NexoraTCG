# Coletor MYP v5.0 — Registrar / Cotizar
#
# Uso: python Coletor_MYP_v5_0_Registrar_Cotizar.py
#   1 = Registrar JSON pessoal (ex.: Leon.json) em LeonFormatado.json + cartas.json.
#   2 = Cotizar apenas JSONs Formatado (ex.: LeonFormatado.json).
#
# Alternativa sem menu:
#   python Coletor_MYP_v5_0_Registrar_Cotizar.py --modo registrar --arquivo Leon.json
#   python Coletor_MYP_v5_0_Registrar_Cotizar.py --modo cotizar --arquivo LeonFormatado.json --dias 1
#
# Na importação incremental, as quantidades no JSON original são totais por
# combinação estado/idioma/foil/tags; repetir a importação é idempotente.
# O Registrar não apaga cartas ausentes da fonte. Remoções são manuais.
# Em ambas as operações, os arquivos são atualizados a cada carta concluída.
# A coleta de ilustrador fica apenas no cadastro geral (cartas.json).

import json
import os
import re
import shutil
import statistics
import sys
import time
import tempfile
import unicodedata
from collections import defaultdict, deque
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
import argparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PASTA_IMAGENS = BASE_DIR / "imagens"
PASTA_BACKUPS = BASE_DIR / "backups"
ARQUIVO_CARTAS = BASE_DIR / "cartas.json"
PARCIAL_CARTAS = BASE_DIR / "cartas.parcial.json"

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

# Intervalo MÍNIMO entre QUALQUER request para a MYP.
# Se quiser ser ainda mais conservador, aumente para 2.5 ou 3.0.
INTERVALO_MINIMO_REQUESTS = 1.75

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
_BLOQUEIO_429_ATE = 0.0


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
    """Escrita atômica: não deixa JSON truncado em interrupções."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=caminho.parent,
            prefix=f".{caminho.stem}.", suffix=".tmp", delete=False,
        ) as f:
            temporario = Path(f.name)
            json.dump(dados, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
        os.replace(temporario, caminho)
    finally:
        if temporario is not None and temporario.exists():
            temporario.unlink()


def carregar_lista_json(caminho: Path) -> list[dict]:
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    if not isinstance(dados, list):
        raise ValueError(f"{caminho.name} precisa ter uma lista na raiz.")
    if not all(isinstance(x, dict) for x in dados):
        raise ValueError(f"Todos os itens de {caminho.name} precisam ser objetos.")
    return dados


def carregar_json_se_existir(caminho: Path) -> list[dict]:
    # Erros nunca são escondidos: substituir um catálogo corrompido por []
    # apagaria cartas já registradas.
    if not caminho.exists():
        return []
    return carregar_lista_json(caminho)


def expandir_registros_fonte(fonte: list[dict]) -> list[dict]:
    """Converte o JSON agrupado em exemplares lógicos para a coleta existente.

    As propriedades da variante substituem as da carta (inclusive tags=[]).
    Esta expansão é somente em memória; os arquivos finais permanecem agrupados.
    Também aceita o JSON antigo, com um registro por combinação de atributos.
    """
    registros = []
    for pos, carta in enumerate(fonte, start=1):
        variantes = carta.get("variantes")
        if variantes is None:
            registros.append(dict(carta))
            continue
        if not isinstance(variantes, list) or not variantes:
            raise ValueError(f"Carta {pos} ({carta.get('nome')}): variantes deve ser lista não vazia.")

        comum = {k: v for k, v in carta.items() if k != "variantes"}
        total = 0
        for indice, variante in enumerate(variantes, start=1):
            if not isinstance(variante, dict):
                raise ValueError(f"Carta {pos}, variante {indice}: formato inválido.")
            atual = dict(comum)
            # O formato agrupado pode misturar nomes legados de tags.
            # Uma definição na variante sempre substitui todas as do pai.
            nomes_tags = ("tags", "tags_extras", "tags_extra")
            if any(k in variante for k in nomes_tags):
                for campo_tag in nomes_tags:
                    atual.pop(campo_tag, None)
            atual.update(variante)
            qtd = atual.get("quantidade", 1)
            if isinstance(qtd, bool) or not isinstance(qtd, int) or qtd < 1:
                raise ValueError(f"Carta {pos}, variante {indice}: quantidade inválida: {qtd!r}")
            total += qtd
            # Se não houver tags na variante, herda do registro principal.
            # Se houver [], representa explicitamente a ausência de tags.
            registros.append(atual)

        if total != carta.get("quantidade"):
            raise ValueError(
                f"Carta {pos} ({carta.get('nome')}): quantidade geral "
                f"{carta.get('quantidade')} != soma das variantes {total}."
            )
    return registros


def criar_backup(caminho: Path) -> Path | None:
    if not caminho.exists():
        return None
    PASTA_BACKUPS.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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


def tags_extras_registro(registro: dict) -> list[str]:
    """Normaliza tags extras vindas do JSON-fonte; na ausência, retorna lista vazia."""
    valor = registro.get("tags_extras")
    if valor is None:
        valor = registro.get("tags_extra")
    if valor is None:
        valor = registro.get("tags")

    if valor is None:
        return []
    if isinstance(valor, (list, tuple, set)):
        saida = []
        vistos = set()
        for tag in valor:
            t = texto_normalizado(tag)
            if t and chave_texto(t) not in vistos:
                vistos.add(chave_texto(t))
                saida.append(t)
        return saida

    t = texto_normalizado(valor)
    return [t] if t else []


def combinar_tags_extras(registros: list[dict]) -> list[str]:
    saida = []
    vistos = set()
    for registro in registros:
        for tag in tags_extras_registro(registro):
            k = chave_texto(tag)
            if k not in vistos:
                vistos.add(k)
                saida.append(tag)
    return saida


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
    global _ULTIMO_REQUEST, _BLOQUEIO_429_ATE
    ultimo_erro = None

    if time.monotonic() < _BLOQUEIO_429_ATE:
        raise RuntimeError("HTTP 429: pausa automática da MYP ativa; registro ficará pendente")

    for tentativa in range(1, MAX_TENTATIVAS_REQUEST + 1):
        esperar_rate_limit()

        try:
            resposta = sessao.get(url, headers=HEADERS, timeout=timeout)
            _ULTIMO_REQUEST = time.monotonic()

            if resposta.status_code not in STATUS_REPETIR:
                resposta.raise_for_status()
                return resposta

            if resposta.status_code == 429:
                espera = segundos_retry_after(resposta)
                _BLOQUEIO_429_ATE = time.monotonic() + max(60.0, espera or 180.0)
                # Não insiste nem perde o lote: marca a carta como pendente
                # e as seguintes não pressionam o servidor durante o bloqueio.
                raise RuntimeError("HTTP 429: limite da MYP; carta pendente para outra execução")

            if tentativa >= MAX_TENTATIVAS_REQUEST:
                resposta.raise_for_status()

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
    if k in {"nao", "não", "nao existe foil", "sem foil", "normal", "regular"}:
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


def extrair_quantidade_unidades(texto: str) -> int:
    m = re.search(r"\b(\d[\d.]*)\s*un\.", str(texto), re.I)
    if not m:
        return 1
    try:
        return max(1, int(m.group(1).replace(".", "")))
    except Exception:
        return 1


def anuncio_certificado(el) -> bool | None:
    """True = Lojistas e Certificados; False = Demais vendedores."""
    for titulo in el.find_all_previous(["h1", "h2", "h3", "h4"]):
        k = chave_texto(titulo.get_text(" ", strip=True))
        if "demais vendedores" in k:
            return False
        if "lojistas e certificados" in k:
            return True
    return None


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
        quantidade_unidades = extrair_quantidade_unidades(texto)
        certificado = anuncio_certificado(el)

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
            "quantidade_unidades": quantidade_unidades,
            "certificado": certificado,
            "_vendedor": vendedor,
        }

        chave = (
            chave_texto(vendedor),
            anuncio["preco"],
            anuncio["estado"],
            anuncio["idioma"],
            anuncio["foil"],
            anuncio["jumbo"],
            anuncio["quantidade_unidades"],
            anuncio["certificado"],
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


def extrair_ilustrador(soup: BeautifulSoup) -> str | None:
    """Lê o campo real 'Ilustrador' na página MYP, sem adivinhar.

    Funciona quando o rótulo e o nome estão no mesmo elemento ou em linhas
    consecutivas. Não confunde o nome com HP, Raridade ou Tipo Carta.
    """
    linhas = [texto_normalizado(linha) for linha in soup.get_text("\n", strip=True).splitlines()]
    linhas = [linha for linha in linhas if linha]
    rotulo = re.compile(r"^(?:Ilustrador(?:a)?|Illustrator|Illustrated by|Ilustrado por)\b\s*[:\-–]?\s*(.*)$", re.I)
    proibidos = re.compile(
        r"^(?:HP\b|Tipo\b|Raridade\b|Data de lançamento\b|Descrição\b|"
        r"Ofertas\b|Outras Edições\b|Código\b|Edição\b)", re.I
    )
    for indice, linha in enumerate(linhas):
        achou = rotulo.match(linha)
        if not achou:
            continue
        candidato = texto_normalizado(achou.group(1))
        if not candidato and indice + 1 < len(linhas):
            candidato = linhas[indice + 1]
        # Se o layout juntou metadados numa única linha, termina antes deles.
        candidato = re.split(
            r"\s+(?=HP\s*\d+\b|Tipo Energia\b|Tipo Carta\b|"
            r"Raridade\b|Data de lançamento\b|Descrição\b)",
            candidato, maxsplit=1, flags=re.I
        )[0].strip(" :–-\t")
        if candidato and len(candidato) <= 120 and not proibidos.match(candidato):
            return candidato
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
        "ilustrador": extrair_ilustrador(soup),
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


def segundo_menor_preco(anuncios: list[dict]) -> float | None:
    valores = sorted(float(a["preco"]) for a in anuncios if a.get("preco") is not None)
    if len(valores) < 2:
        return None
    return round(valores[1], 2)


def multiplicadores_certificado_por_estado(anuncios: list[dict]) -> dict[str, float | None]:
    """
    Compara menor certificado / menor não certificado por estado.
    O idioma é deliberadamente ignorado neste cálculo.
    """
    saida = {}
    for estado in ("M", "NM", "SP", "MP", "HP", "DM"):
        do_estado = [a for a in anuncios if normalizar_estado(a.get("estado")) == estado]
        certificados = [float(a["preco"]) for a in do_estado if a.get("certificado") is True and a.get("preco") is not None]
        nao_certificados = [float(a["preco"]) for a in do_estado if a.get("certificado") is False and a.get("preco") is not None]

        if certificados and nao_certificados and min(nao_certificados) > 0:
            saida[estado] = round(min(certificados) / min(nao_certificados), 4)
        else:
            saida[estado] = None
    return saida


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


def precos_por_estado(
    anuncios: list[dict],
    *,
    incluir_segundo_menor: bool = False,
    multiplicadores_certificado: dict[str, float | None] | None = None,
) -> dict:
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
            if incluir_segundo_menor:
                stats["segundo_menor"] = segundo_menor_preco(grupo)
            if multiplicadores_certificado is not None:
                stats["multiplicador_certificado"] = multiplicadores_certificado.get(estado)
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
    Grava os preços reais em BR, Estrangeiro e Oriental.
    Cada idioma recebe segundo menor e estados; o multiplicador de certificado
    de cada estado é calculado globalmente, sem filtrar idioma.
    """
    grupos = defaultdict(list)
    for a in anuncios:
        g = grupo_idioma(a.get("idioma"))
        if g:
            grupos[g].append(a)

    multiplicadores = multiplicadores_certificado_por_estado(anuncios)
    saida = {}
    for g in ("BR", "Estrangeiro", "Oriental"):
        grupo = grupos.get(g, [])
        if not grupo:
            continue

        stats = resumo_anuncios(grupo)
        if not stats:
            continue
        stats["segundo_menor"] = segundo_menor_preco(grupo)

        por_estado = precos_por_estado(
            grupo,
            incluir_segundo_menor=True,
            multiplicadores_certificado=multiplicadores,
        )
        if por_estado:
            stats["por_estado"] = por_estado

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
    idioma_exato = normalizar_idioma(registro.get("lingua") or registro.get("idioma"))
    for a in anuncios:
        if normalizar_estado(a.get("estado")) != estado:
            continue
        if normalizar_idioma(a.get("idioma")) != idioma_exato:
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


def chave_preco_exato(estado: str, idioma: str, foil: str, jumbo: bool,
                      permite_foil: bool) -> str:
    foil_key = normalizar_foil(foil) if permite_foil else "Normal"
    return f"{idioma}|{estado}|{foil_key}|{'Jumbo' if jumbo else 'Normal'}"


def precos_por_combinacao(anuncios: list[dict], permite_foil: bool) -> dict:
    """Preços sem misturar idioma, estado, acabamento nem tamanho."""
    grupos = defaultdict(list)
    for a in anuncios:
        estado = normalizar_estado(a.get("estado"))
        idioma = normalizar_idioma(a.get("idioma"))
        if not estado or not idioma:
            continue
        chave = chave_preco_exato(estado, idioma, a.get("foil", "Normal"),
                                 bool(a.get("jumbo")), permite_foil)
        grupos[chave].append(a)
    return {chave: resumo_anuncios(grupo) for chave, grupo in sorted(grupos.items())}


def preco_para_dono_catalogo(catalogo: dict, registro: dict) -> dict | None:
    """Cache exato; evita usar preço incorreto de outro foil/idioma."""
    estado = normalizar_estado(registro.get("estado"))
    idioma = normalizar_idioma(registro.get("lingua") or registro.get("idioma"))
    if not estado or not idioma:
        return None
    if "precos_por_combinacao" in catalogo:
        jumbo = bool(registro.get("jumbo")) or detectar_jumbo(registro.get("foil"))
        chave = chave_preco_exato(estado, idioma, registro.get("foil", "Normal"),
                                 jumbo, bool(catalogo.get("permite_variantes_foil", False)))
        return catalogo["precos_por_combinacao"].get(chave)

    # Catálogos antigos podem não conter combinações exatas. Não atribuir
    # preços agregados quando há variantes foil ou tamanho incompatíveis.
    if catalogo.get("precos_por_foil") or catalogo.get("precos_por_tamanho"):
        return None
    grupo = grupo_idioma(idioma)
    bloco = catalogo.get("precos_por_idioma", {}).get(grupo, {})
    stats = bloco.get("por_estado", {}).get(estado)
    # O agrupamento legado mistura Inglês, Alemão etc. Apenas BR é inequívoco.
    if grupo != "BR":
        return None
    if not stats:
        return None
    return {k: stats.get(k) for k in ("menor", "media", "mediana", "maior", "anuncios")}


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

    idiomas = precos_por_idioma(todos_anuncios)
    foils = precos_por_foil(todos_anuncios, permite_foil)
    tamanhos = precos_por_tamanho(todos_anuncios)

    numero_ofertas = extrair_numero_ofertas(primeira)
    # Diferencie produto sem ofertas de HTML bloqueado/layout alterado:
    # nunca sobrescreva todas as cotações por uma coleta silenciosamente vazia.
    if not todos_anuncios and numero_ofertas != 0:
        raise RuntimeError("Nenhum anúncio extraído; possível bloqueio ou mudança no layout da MYP")
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
        "ilustrador": dados.get("ilustrador"),  # apenas da página MYP, nunca do inventário pessoal
        "menor_preco": geral.get("menor") if geral else None,
        "media": geral.get("media") if geral else None,
        "mediana": geral.get("mediana") if geral else None,
        "maior_preco": geral.get("maior") if geral else None,
        "preco_tcgplayer": extrair_preco_tcgplayer(primeira),
        "numero_anuncios_myp": numero_ofertas,
        "numero_unidades_myp": sum(int(a.get("quantidade_unidades", 1) or 1) for a in todos_anuncios),
        "data_hora_coleta": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tags_extras": tags_extras_registro(fonte),
        "caminho_foto": imagem,
        "link_myp": url_final,
        "precos_por_combinacao": precos_por_combinacao(todos_anuncios, permite_foil),
        "permite_variantes_foil": permite_foil,
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

    # Ilustrador não faz parte das propriedades pessoais da carta;
    # é metadado global, salvo unicamente em cartas.json.
    return {
        "nome": fonte.get("nome") or catalogo.get("nome"),
        "id": catalogo.get("id"),
        "estado": fonte.get("estado"),
        "idioma": fonte.get("lingua") or fonte.get("idioma"),
        "foil": fonte.get("foil"),
        "menor_preco": precos.get("menor") if precos else None,
        "preco_medio": precos.get("media") if precos else None,
        "mediana_preco": precos.get("mediana") if precos else None,
        "maior_preco": precos.get("maior") if precos else None,
        "quantidade": fonte.get("quantidade", 1),
        "tags_extras": tags_extras_registro(fonte),
        "data_hora_coleta": catalogo.get("data_hora_coleta"),
    }


def agrupar_registros_leon(registros: list[dict], carta: dict) -> dict:
    """Um único registro formatado por carta, com variações e preços próprios.

    Só atributos realmente compartilhados sobem ao nível principal. O conjunto
    de campos por variante continua completo após herdar os valores do pai.
    Não soma preços nem mistura estados, idiomas, foil ou tags de exemplares.
    """
    if not registros:
        raise ValueError("Não há registros para agrupar.")

    total = sum(int(r.get("quantidade", 1)) for r in registros)
    primeiro = registros[0]
    resultado = {
        "nome": carta.get("nome") or primeiro.get("nome"),
        "numeracao": carta.get("numeracao"),
        "id": carta.get("id"),
        "link": carta.get("link_myp"),
        "quantidade": total,
        "data_hora_coleta": carta.get("data_hora_coleta"),
        "data_hora_cotizacao": carta.get("data_hora_cotizacao") or carta.get("data_hora_coleta"),
    }

    # A chave nome pode diferir em traduções; somente nesse caso é variante.
    # Tags devem ficar somente nos exemplares que as possuem.
    atributos = (
        "estado", "idioma", "foil", "tags_extras",
        "menor_preco", "preco_medio", "mediana_preco", "maior_preco",
    )
    comuns = {}
    compartilhados = set()
    for campo in atributos:
        valores = [r.get(campo) for r in registros]
        if all(v == valores[0] for v in valores):
            compartilhados.add(campo)
            valor = valores[0]
            if campo != "tags_extras" or valor:
                comuns[campo] = valor
    resultado.update(comuns)

    if len(registros) == 1:
        return resultado

    variantes = []
    for r in registros:
        variante = {"quantidade": r.get("quantidade", 1)}
        if r.get("nome") and r["nome"] != resultado["nome"]:
            variante["nome"] = r["nome"]
        for campo in atributos:
            if campo in compartilhados:
                continue
            valor = r.get(campo)
            # Com tags heterogêneas, [] é importante: significa sem tag
            # enquanto outra versão tem tags.
            if campo == "tags_extras":
                variante[campo] = valor or []
            else:
                variante[campo] = valor
        variantes.append(variante)
    resultado["variantes"] = variantes
    return resultado


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
# MODOS REGISTRAR / COTIZAR
# ============================================================

CAMPOS_PRECO_DONO = ("menor_preco", "preco_medio", "mediana_preco", "maior_preco")


def data_agora() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def interpretar_data(valor):
    if not valor:
        return None
    try:
        dt = datetime.fromisoformat(str(valor).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt.astimezone()
    except (ValueError, TypeError):
        return None


def link_carta(registro: dict) -> str:
    return texto_normalizado(registro.get("link") or registro.get("link_myp") or "")


def url_myp_valida(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"mypcards.com", "www.mypcards.com"} and bool(id_produto_myp(url))


def chave_carta(registro: dict) -> tuple:
    link = link_carta(registro)
    pid = id_produto_myp(link)
    if pid:
        return ("myp", pid)
    return ("nome", chave_texto(registro.get("nome")), chave_texto(registro.get("numeracao")))


def chave_variante(registro: dict) -> tuple:
    lingua = normalizar_idioma(registro.get("lingua") or registro.get("idioma"))
    # Aceita também estados desconhecidos (?) sem confundi-los com NM.
    estado = normalizar_estado(registro.get("estado")) or chave_texto(registro.get("estado"))
    foil = normalizar_foil(registro.get("foil"))
    jumbo = bool(registro.get("jumbo")) or detectar_jumbo(registro.get("foil"))
    tags = tuple(sorted({chave_texto(t) for t in tags_extras_registro(registro)}))
    return (estado, chave_texto(lingua), chave_texto(foil), jumbo, tags,
            chave_texto(registro.get("nome")))


def combinar_registros_iguais(registros: list[dict]) -> list[dict]:
    saida = []
    indices = {}
    for registro in registros:
        chave = chave_variante(registro)
        if chave in indices:
            atual = saida[indices[chave]]
            atual["quantidade"] += int(registro.get("quantidade", 1))
        else:
            novo = dict(registro)
            novo["quantidade"] = int(novo.get("quantidade", 1))
            indices[chave] = len(saida)
            saida.append(novo)
    return saida


def localizar_formatado(fonte: Path) -> Path:
    """Prioriza um Formatado já existente, com ou sem espaço no nome."""
    candidatos = [
        fonte.with_name(f"{fonte.stem}Formatado.json"),
        fonte.with_name(f"{fonte.stem} Formatado.json"),
    ]
    existentes = [p for p in candidatos if p.exists()]
    if len(existentes) > 1:
        raise ValueError(f"Há dois formatados para {fonte.name}. Mantenha apenas um: "
                         + ", ".join(p.name for p in existentes))
    return existentes[0] if existentes else candidatos[0]


def fonte_para_leon(r: dict, catalogo: dict, *, anuncios=None, permite_foil=False) -> dict:
    """Transforma fonte pessoal em registro individual cotado."""
    base = criar_registro_leon(catalogo, r, anuncios, permite_foil)
    base["data_hora_cotizacao"] = (
        catalogo.get("data_hora_cotizacao") or catalogo.get("data_hora_coleta")
    )
    return base


def valor_montante(r: dict, quantidade: int) -> tuple[float, float, int, int]:
    minimo = r.get("menor_preco")
    medio = r.get("preco_medio")
    v_min = float(minimo) * quantidade if minimo is not None else 0.0
    v_med = float(medio) * quantidade if medio is not None else 0.0
    return v_min, v_med, quantidade if minimo is None else 0, quantidade if medio is None else 0


class Montante:
    def __init__(self):
        self.minimo = 0.0
        self.medio = 0.0
        self.sem_min = 0
        self.sem_med = 0
        self.unidades = 0
        self.soma_percentual = 0.0
        self.unidades_comparaveis = 0

    def adicionar(self, r: dict, quantidade: int):
        if quantidade <= 0:
            return
        minimo, medio, sem_min, sem_med = valor_montante(r, quantidade)
        self.minimo += minimo
        self.medio += medio
        self.sem_min += sem_min
        self.sem_med += sem_med
        self.unidades += quantidade

    def comparar(self, velho: dict, novo: dict, quantidade: int):
        if quantidade <= 0:
            return
        a = velho.get("preco_medio")
        b = novo.get("preco_medio")
        if a is not None and b is not None and float(a) > 0:
            self.soma_percentual += (float(b) / float(a) - 1.0) * 100.0 * quantidade
            self.unidades_comparaveis += quantidade

    def resumo(self, *, cotizacao=False) -> str:
        texto = (f"MONTANTE DA LEVA: mínimo R$ {self.minimo:,.2f} | "
                 f"médio R$ {self.medio:,.2f} | {self.unidades} unidades")
        if self.sem_min or self.sem_med:
            texto += (f" | sem preço: {self.sem_min} (mínimo), "
                      f"{self.sem_med} (média); totais parciais")
        if cotizacao:
            if self.unidades_comparaveis:
                percentual = self.soma_percentual / self.unidades_comparaveis
                texto += (f"\nVARIAÇÃO MÉDIA VS. ÚLTIMA COTIZAÇÃO: "
                          f"{percentual:+.2f}% ({self.unidades_comparaveis} unidades comparáveis)")
            else:
                texto += "\nVARIAÇÃO MÉDIA VS. ÚLTIMA COTIZAÇÃO: N/D (sem preços anteriores comparáveis)"
        return texto


def preservar_dados_gerais(novo: dict, antigo: dict | None) -> dict:
    """Preserva metadados já conhecidos se MYP não fornecer na atualização."""
    if not antigo:
        return novo
    for k in ("colecao", "sigla_colecao", "era", "ano", "raridade", "ilustrador",
              "preco_tcgplayer", "caminho_foto"):
        if novo.get(k) is None:
            novo[k] = antigo.get(k)
    # As tags do catálogo geral pertencem ao histórico da carta; o registro
    # de novas pessoas não deve apagá-las indiscriminadamente.
    novo["tags_extras"] = combinar_tags_extras([antigo, novo])
    return novo


def achar_dono_existente(dono: list[dict], reg: dict, card_id: str | None):
    """Localiza uma única carta pessoal sem fundir produtos MYP diferentes."""
    pid = id_produto_myp(link_carta(reg))
    if card_id:
        correspondentes = [(i, d) for i, d in enumerate(dono) if d.get("id") == card_id]
        if len(correspondentes) == 1:
            return correspondentes[0]
    if pid:
        correspondentes = [(i, d) for i, d in enumerate(dono)
                           if id_produto_myp(link_carta(d)) == pid]
        if len(correspondentes) == 1:
            return correspondentes[0]
    alvo = (chave_texto(reg.get("nome")), chave_texto(reg.get("numeracao")))
    correspondentes = [(i, d) for i, d in enumerate(dono)
                       if (chave_texto(d.get("nome")), chave_texto(d.get("numeracao"))) == alvo
                       and (not pid or not id_produto_myp(link_carta(d))
                            or id_produto_myp(link_carta(d)) == pid)]
    return correspondentes[0] if len(correspondentes) == 1 else (None, None)


def achar_catalogo_seguro(registro: dict, catalogo: list[dict]) -> dict | None:
    card_id = registro.get("id")
    if card_id:
        for c in catalogo:
            if c.get("id") == card_id:
                return c
    por_link, por_nome_numero = indexar_catalogo_existente(catalogo)
    c = achar_existente(registro, por_link, por_nome_numero)
    # Não une cartas distintas que tenham mesmo nome/número, mas links diversos.
    if c and id_produto_myp(link_carta(c)) and id_produto_myp(link_carta(registro)):
        if id_produto_myp(link_carta(c)) != id_produto_myp(link_carta(registro)):
            return None
    return c


def atualizar_catalogo(catalogo: list[dict], carta: dict) -> None:
    cid = carta.get("id")
    for i, velho in enumerate(catalogo):
        if cid and velho.get("id") == cid:
            catalogo[i] = carta
            return
    catalogo.append(carta)


def preparar_salvamento(caminhos: list[Path]) -> None:
    for caminho in caminhos:
        if caminho.exists():
            criar_backup(caminho)


def gravar_par(catalogo: list[dict], dono: list[dict], caminho_dono: Path):
    # Arquivos finais são checkpoints. Se uma execução for interrompida após
    # apenas a primeira gravação, a próxima reconcilia o segundo arquivo sem
    # duplicar exemplares. Cada arquivo é substituído atomicamente.
    salvar_json(ARQUIVO_CARTAS, catalogo)
    salvar_json(caminho_dono, dono)


def registrar(caminho_fonte: Path):
    caminho_dono = localizar_formatado(caminho_fonte)
    bruto = carregar_lista_json(caminho_fonte)
    registros = expandir_registros_fonte(bruto)
    catalogo = carregar_json_se_existir(ARQUIVO_CARTAS)
    dono = carregar_json_se_existir(caminho_dono)
    proximo = proximo_id(catalogo)
    grupos = defaultdict(list)
    for r in registros:
        grupos[chave_carta(r)].append(r)

    print(f"\nREGISTRAR | {caminho_fonte.name} -> {caminho_dono.name}")
    print(f"{len(grupos)} cartas distintas, {len(registros)} combinações de variantes")
    print("preços: menor | média | mediana | maior (R$)")
    print("Registro incremental: cartas/variantes iguais não serão duplicadas.\n")
    preparar_salvamento([ARQUIVO_CARTAS, caminho_dono])
    sessao = requests.Session()
    montante = Montante()
    adicionadas = 0
    corrigidas_total = 0
    repetidas = 0
    problemas = []
    sucesso = 0

    for n, (_, regs) in enumerate(grupos.items(), 1):
        base = regs[0]
        existente = achar_catalogo_seguro(base, catalogo)
        id_carta = existente.get("id") if existente else None
        idx_dono, registro_antigo = achar_dono_existente(dono, base, id_carta)
        if not id_carta and registro_antigo:
            id_carta = registro_antigo.get("id")
        if not id_carta:
            id_carta = gerar_id(proximo)
            proximo += 1

        anteriores = (expandir_registros_fonte([registro_antigo])
                      if registro_antigo else [])
        anteriores = combinar_registros_iguais(anteriores)
        novos_brutos = combinar_registros_iguais(regs)
        merged = [dict(r) for r in anteriores]
        # Se a fonte atual substituir um registro F/? antigo por uma
        # conservação definida, isso é uma correção, NÃO uma nova carta.
        # Quando a fonte contém o total inteiro e uma variante foi trocada,
        # também podemos reconhecer a correção sem inflar a quantidade.
        chaves_fonte = {chave_variante(r) for r in novos_brutos}
        soma_antiga = sum(r["quantidade"] for r in anteriores)
        soma_fonte = sum(r["quantidade"] for r in novos_brutos)
        corrigidas = set()
        for atual in novos_brutos:
            chave_atual = chave_variante(atual)
            if any(chave_variante(x) == chave_atual for x in merged):
                continue
            candidatos = []
            for pos, anterior in enumerate(merged):
                if chave_variante(anterior) in chaves_fonte:
                    continue
                if chave_variante(anterior) in corrigidas:
                    continue
                mesma_base = (
                    chave_variante(anterior)[1:4] == chave_atual[1:4]
                    and chave_variante(anterior)[4:] == chave_atual[4:]
                    and int(anterior.get("quantidade", 1)) == int(atual.get("quantidade", 1))
                )
                estado_antigo = chave_texto(anterior.get("estado"))
                if mesma_base and (estado_antigo in {"?", "f"}
                                   or (soma_antiga == soma_fonte and soma_fonte > 0)):
                    candidatos.append(pos)
            if len(candidatos) == 1:
                pos = candidatos[0]
                merged[pos] = dict(atual)
                corrigidas.add(chave_atual)

        indices = {chave_variante(r): i for i, r in enumerate(merged)}
        incrementos = {}
        for r in novos_brutos:
            chave = chave_variante(r)
            pos = indices.get(chave)
            if pos is None:
                indices[chave] = len(merged)
                merged.append(dict(r))
                incremento = int(r.get("quantidade", 1))
            else:
                anterior = int(merged[pos].get("quantidade", 1))
                novo_total = max(anterior, int(r.get("quantidade", 1)))
                merged[pos]["quantidade"] = novo_total
                incremento = novo_total - anterior
            if incremento:
                incrementos[chave] = incremento

        unidade_nova = sum(incrementos.values())
        pendente = bool(registro_antigo and registro_antigo.get("registro_pendente"))
        if not unidade_nova and not pendente and existente and not corrigidas:
            repetidas += 1
            print(f"[{n}/{len(grupos)}] {base.get('nome')} {base.get('numeracao')} | JÁ REGISTRADA")
            continue

        link = link_carta(base) or (link_carta(registro_antigo) if registro_antigo else "")
        if not link and existente:
            link = link_carta(existente)
        for r in merged:
            r.setdefault("link", link)
        carta = dict(existente) if existente else {
            "id": id_carta, "nome": base.get("nome"),
            "numeracao": base.get("numeracao"), "link_myp": link,
            "data_hora_coleta": None, "data_hora_cotizacao": None,
        }
        carta["id"] = id_carta
        if link and not carta.get("link_myp"):
            carta["link_myp"] = link
        # Para uma nova variante, uma cotação nova evita aplicar preços de
        # estado/foil/idioma incompatíveis existentes nos caches legados.
        novos_precificaveis = [r for r in merged
                              if chave_variante(r) in incrementos or chave_variante(r) in corrigidas
                              and normalizar_estado(r.get("estado"))
                              and normalizar_idioma(r.get("lingua") or r.get("idioma"))]
        falta_cotacao_exata = any(
            "precos_por_combinacao" not in carta or
            preco_para_dono_catalogo(carta, r) is None
            for r in novos_precificaveis
        )
        precisa_consultar = bool(
            (not existente) or pendente or falta_cotacao_exata
        )
        anuncios = None
        permite_foil = bool(carta.get("permite_variantes_foil", False))
        erro = None
        if precisa_consultar:
            if url_myp_valida(link):
                try:
                    fonte_myp = dict(base, link=link)
                    recolhida, anuncios, permite_foil = coletar_carta_myp(fonte_myp, id_carta, sessao)
                    recolhida["data_hora_cotizacao"] = recolhida["data_hora_coleta"]
                    carta = preservar_dados_gerais(recolhida, existente)
                    carta["tags_extras"] = combinar_tags_extras([carta] + merged)
                    atualizar_catalogo(catalogo, carta)
                    sucesso += 1
                except Exception as exc:
                    erro = str(exc)
            else:
                erro = "link MYP ausente ou inválido"

        if erro:
            problemas.append({"modo": "registrar", "arquivo": caminho_fonte.name,
                              "nome": base.get("nome"), "numeracao": base.get("numeracao"),
                              "link": link, "erro": erro})
            print(f"[{n}/{len(grupos)}] {base.get('nome')} {base.get('numeracao')} | PENDENTE: {erro}")

        if anuncios is not None:
            novos_dono = [fonte_para_leon(r, carta, anuncios=anuncios,
                                          permite_foil=permite_foil) for r in merged]
        else:
            # Preserve de verdade a precificação de variantes já existentes;
            # use o catálogo somente para os novos registros ainda sem preço.
            chaves_anteriores = {chave_variante(r): r for r in anteriores}
            novos_dono = []
            for r in merged:
                chave = chave_variante(r)
                if chave in chaves_anteriores and chave not in corrigidas:
                    cache = dict(chaves_anteriores[chave])
                    cache["quantidade"] = r["quantidade"]
                    novos_dono.append(cache)
                else:
                    novos_dono.append(fonte_para_leon(r, carta))

        carta_pessoal = dict(carta)
        if anuncios is None and registro_antigo:
            carta_pessoal["data_hora_coleta"] = registro_antigo.get("data_hora_coleta")
            carta_pessoal["data_hora_cotizacao"] = (
                registro_antigo.get("data_hora_cotizacao") or registro_antigo.get("data_hora_coleta")
            )
        novo_dono = agrupar_registros_leon(novos_dono, carta_pessoal)
        novo_dono["id"] = id_carta if existente or anuncios is not None else None
        novo_dono["link"] = link or None
        novo_dono["registro_pendente"] = bool(erro)
        # sem link ou falha de coleta: o registro pessoal fica protegido;
        # somente cartas efetivamente consultadas entram no catálogo geral.
        if idx_dono is None:
            dono.append(novo_dono)
        else:
            dono[idx_dono] = novo_dono
        gravar_par(catalogo, dono, caminho_dono)
        for r in novos_dono:
            montante.adicionar(r, incrementos.get(chave_variante(r), 0))
        adicionadas += unidade_nova
        corrigidas_total += len(corrigidas)
        print(f"[{n}/{len(grupos)}] {base.get('nome')} {base.get('numeracao')} | "
              f"+{unidade_nova} unidades, {len(corrigidas)} correções | "
              + ("COTADA" if anuncios is not None else "CACHE" if not erro else "SEM COTAÇÃO"))
        for r in novos_dono:
            if chave_variante(r) in incrementos:
                print(f"            {texto_preco_dono(r)} | +{incrementos[chave_variante(r)]} un.")
        print("            " + montante.resumo())

    sessao.close()
    salvar_json(BASE_DIR / "erros_coleta.json", problemas)
    print(f"\nREGISTRO CONCLUÍDO | novas unidades: {adicionadas} | "
          f"variantes corrigidas: {corrigidas_total} | "
          f"repetidas: {repetidas} | pendências: {len(problemas)} | "
          f"consultadas na MYP: {sucesso}")
    print(montante.resumo())
    print(f"Arquivos: {caminho_dono.name} e {ARQUIVO_CARTAS.name}")


def cotizar(caminho_dono: Path, dias: float):
    if dias < 0:
        raise ValueError("O número de dias precisa ser >= 0.")
    dono = carregar_lista_json(caminho_dono)
    catalogo = carregar_json_se_existir(ARQUIVO_CARTAS)
    proximo = proximo_id(catalogo)
    limite = datetime.now().astimezone() - timedelta(days=dias)
    print(f"\nCOTIZAR | {caminho_dono.name} | intervalo mínimo: {dias:g} dia(s)")
    print("0 dias = cotizar todas, inclusive as cotizadas agora")
    print("preços: menor | média | mediana | maior (R$)\n")
    preparar_salvamento([ARQUIVO_CARTAS, caminho_dono])
    sessao = requests.Session()
    montante = Montante()
    cotizadas = puladas = falhas = 0
    problemas = []

    for n in range(len(dono)):
        antiga = dono[n]
        nome = antiga.get("nome")
        num = antiga.get("numeracao")
        data_anterior = interpretar_data(
            antiga.get("data_hora_cotizacao") or antiga.get("data_hora_coleta")
        )
        if dias > 0 and data_anterior is not None and data_anterior > limite:
            puladas += 1
            print(f"[{n+1}/{len(dono)}] {nome} {num} | RECENTE, PULADA | "
                  f"última: {data_anterior.isoformat(timespec='minutes')}")
            continue

        existente = achar_catalogo_seguro(antiga, catalogo)
        cid = (existente.get("id") if existente else None) or antiga.get("id")
        if not cid:
            cid = gerar_id(proximo)
            proximo += 1
        link = link_carta(antiga) or link_carta(existente or {})
        if not url_myp_valida(link):
            falhas += 1
            problemas.append({"modo": "cotizar", "arquivo": caminho_dono.name,
                              "nome": nome, "numeracao": num, "link": link,
                              "erro": "link MYP ausente ou inválido"})
            print(f"[{n+1}/{len(dono)}] {nome} {num} | SEM LINK, não alterada")
            continue

        try:
            fonte = dict(antiga, link=link)
            nova_carta, anuncios, permite_foil = coletar_carta_myp(fonte, cid, sessao)
        except Exception as exc:
            falhas += 1
            problemas.append({"modo": "cotizar", "arquivo": caminho_dono.name,
                              "nome": nome, "numeracao": num, "link": link, "erro": str(exc)})
            print(f"[{n+1}/{len(dono)}] {nome} {num} | ERRO: {exc} | "
                  "preços e data anterior preservados")
            salvar_json(BASE_DIR / "erros_coleta.json", problemas)
            continue

        nova_carta["data_hora_cotizacao"] = nova_carta["data_hora_coleta"]
        nova_carta = preservar_dados_gerais(nova_carta, existente)
        antigos_r = combinar_registros_iguais(expandir_registros_fonte([antiga]))
        nova_carta["tags_extras"] = combinar_tags_extras([nova_carta] + antigos_r)
        novos_r = [fonte_para_leon(r, nova_carta, anuncios=anuncios,
                                    permite_foil=permite_foil) for r in antigos_r]
        novo_dono = agrupar_registros_leon(novos_r, nova_carta)
        novo_dono["id"] = cid
        novo_dono["registro_pendente"] = False
        if antiga.get("link") and not novo_dono.get("link"):
            novo_dono["link"] = antiga.get("link")

        antigos_por_chave = {chave_variante(r): r for r in antigos_r}
        for r in novos_r:
            quantidade = int(r.get("quantidade", 1))
            montante.adicionar(r, quantidade)
            velho = antigos_por_chave.get(chave_variante(r))
            if velho:
                montante.comparar(velho, r, quantidade)

        atualizar_catalogo(catalogo, nova_carta)
        dono[n] = novo_dono
        gravar_par(catalogo, dono, caminho_dono)
        cotizadas += 1
        print(f"[{n+1}/{len(dono)}] {nome} {num} | ATUALIZADA | "
              f"última: {nova_carta['data_hora_cotizacao']}")
        for r in novos_r:
            print(f"            {texto_preco_dono(r)} | {r.get('quantidade',1)} un.")
        print("            " + montante.resumo(cotizacao=True))

    sessao.close()
    salvar_json(BASE_DIR / "erros_coleta.json", problemas)
    print(f"\nCOTIZAÇÃO CONCLUÍDA | cotizadas: {cotizadas} | "
          f"puladas por idade: {puladas} | falhas: {falhas}")
    print(montante.resumo(cotizacao=True))
    print(f"Arquivos atualizados: {caminho_dono.name} e {ARQUIVO_CARTAS.name}")


def localizar_arquivos(modo: str) -> list[Path]:
    encontrados = []
    for p in sorted(BASE_DIR.glob("*.json"), key=lambda p: p.name.casefold()):
        stem = chave_texto(p.stem).replace(" ", "")
        if stem in {"cartas", "erros_coleta", "controle_registros", "removidas_sem_imagem"}:
            continue
        if (".parcial" in stem or ".backup" in stem or stem.startswith("removidas_")
                or stem.endswith("parcial") or stem.endswith("backup")):
            continue
        e_formatado = stem.endswith("formatado")
        if (modo == "cotizar") == e_formatado:
            encontrados.append(p)
    return encontrados


def escolher_arquivo(modo: str, informado: str | None = None) -> Path:
    arquivos = localizar_arquivos(modo)
    if informado:
        caminho = Path(informado)
        if not caminho.is_absolute():
            caminho = BASE_DIR / caminho
        caminho = caminho.resolve()
        if caminho not in [p.resolve() for p in arquivos]:
            raise ValueError(f"Arquivo '{informado}' não elegível para {modo}. "
                             "Coloque o JSON na pasta do Python e confira o nome.")
        return caminho
    if not arquivos:
        tipo = "*Formatado.json" if modo == "cotizar" else "JSON pessoal não formatado"
        raise FileNotFoundError(f"Nenhum {tipo} disponível na pasta do coletor.")
    print("\nJSONs disponíveis:")
    for n, arquivo in enumerate(arquivos, 1):
        print(f"  {n}. {arquivo.name}")
    while True:
        valor = input("Escolha o número do JSON: ").strip()
        if valor.isdigit() and 1 <= int(valor) <= len(arquivos):
            return arquivos[int(valor) - 1]
        print("Número inválido, tente novamente.")


def escolher_dias() -> float:
    while True:
        valor = input("Cotizar cartas sem atualização há quantos dias? "
                      "(0 = todas, 1 = 24h): ").strip().replace(",", ".")
        try:
            dias = float(valor)
            if 0 <= dias < 100000:
                return dias
        except ValueError:
            pass
        print("Informe um número não negativo (ex.: 0, 1, 2, 7).")


def main():
    ap = argparse.ArgumentParser(
        description="MYP v5.0: registrar novidades sem duplicatas ou cotizar coleções formatadas."
    )
    ap.add_argument("--modo", choices=["registrar", "cotizar"],
                    help="Opcional. Sem o argumento, abre o menu interativo.")
    ap.add_argument("--arquivo", help="Nome do JSON na mesma pasta do Python.")
    ap.add_argument("--dias", type=float, help="Apenas na cotização: idade mínima em dias, 0=todas.")
    args = ap.parse_args()

    try:
        modo = args.modo
        if not modo:
            print("\nCOLETOR MYP v5.0")
            print("  1. Registrar novidades de um JSON pessoal")
            print("  2. Cotizar um JSON pessoal Formatado")
            while True:
                selecao = input("Selecione 1 ou 2: ").strip()
                if selecao in {"1", "2"}:
                    modo = "registrar" if selecao == "1" else "cotizar"
                    break
                print("Escolha 1 ou 2.")
        arquivo = escolher_arquivo(modo, args.arquivo)
        if modo == "registrar":
            registrar(arquivo)
        else:
            dias = args.dias if args.dias is not None else escolher_dias()
            cotizar(arquivo, dias)
    except KeyboardInterrupt:
        print("\nInterrompido. Os arquivos finais já incluem cada carta concluída; "
              "execute novamente quando quiser.")
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERRO INESPERADO: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
