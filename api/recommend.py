import base64, io, os
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def get_spotify_client():
    auth_manager = SpotifyClientCredentials()
    return spotipy.Spotify(auth_manager=auth_manager, requests_timeout=20, retries=3, status_forcelist=(429,500,502,503,504))

GENRE_CANONICALS = {"pop", "rock", "cumbia", "salsa", "banda", "otros"}
GENRE_MAP = {
    "pop":"pop","pops":"pop",
    "rock":"rock","rocks":"rock",
    "cumbia":"cumbia","cumbias":"cumbia",
    "salsa":"salsa","salsas":"salsa",
    "banda":"banda","banda sinaloense":"banda",
    "ingles":None,"inglés":None,"english":None,
}

def _clean_text(x): 
    if pd.isna(x): return ""
    return str(x).strip().lower()

def normalize_single_genre(token):
    t = _clean_text(token).replace("-", " ").replace("_", " ").replace(".", " ")
    t = " ".join(t.split())
    if not t: return None
    if t in GENRE_MAP: return GENRE_MAP[t]
    if t.endswith("s") and t[:-1] in GENRE_MAP: return GENRE_MAP[t[:-1]]
    if t in GENRE_CANONICALS: return t
    for k in ["cumbia","banda","rock","pop","salsa"]:
        if k in t: return k
    return None

def extract_idioma(cell_text:str)->str:
    t = _clean_text(cell_text)
    if any(w in t for w in ["ingles","inglés","english"]): return "en"
    return "es"

def parse_generos_cell(cell_text:str):
    t = _clean_text(cell_text)
    if not t: return []
    raw = [s.strip() for s in t.replace(";", ",").split(",") if s.strip()]
    mapped = []
    for tok in raw:
        g = normalize_single_genre(tok)
        if g is None: continue
        mapped.append(g)
    if not mapped: mapped = ["otros"]
    uniques = list(dict.fromkeys(mapped).keys())
    w = 1.0/len(uniques) if uniques else 1.0
    return [(g, w) for g in uniques]

def load_tiendas_from_bytes(content: bytes, sheet_name: Optional[str] = None)->pd.DataFrame:
    df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
    def normup(c): return df[c].astype(str).str.strip().str.upper() if c in df.columns else ""
    for col in ["CADENA","ESTADO","CIUDAD","FORMATO"]:
        if col in df.columns: df[col] = normup(col)
        else: df[col] = ""
    if "TYPE" in df.columns:
        df["TYPE"] = df["TYPE"].astype(str).str.strip().str.lower()
    else:
        df["TYPE"] = ""
    if "GENERO" not in df.columns: df["GENERO"] = ""
    df["IDIOMA"] = df["GENERO"].apply(extract_idioma)
    df["_GEN_LIST"] = df["GENERO"].apply(parse_generos_cell)
    df_ex = df.loc[df["_GEN_LIST"].str.len()>0].copy().explode("_GEN_LIST", ignore_index=True)
    df_ex["GENERO_CAN"] = df_ex["_GEN_LIST"].apply(lambda t: t[0])
    df_ex["PESO_GENERO"] = df_ex["_GEN_LIST"].apply(lambda t: t[1])
    df_ex.drop(columns=["_GEN_LIST"], inplace=True)
    df_ex["GENERO_CAN"] = df_ex["GENERO_CAN"].where(df_ex["GENERO_CAN"].isin(GENRE_CANONICALS), "otros")
    return df_ex

def _normalize_weights(s:pd.Series)->pd.Series:
    tot = s.sum()
    if tot<=0: return pd.Series([1/len(s)]*len(s), index=s.index)
    return s/tot

def build_aggregates(tiendas:pd.DataFrame):
    by_estado = (tiendas.groupby(["ESTADO","GENERO_CAN"], as_index=False)["PESO_GENERO"].sum()
                       .rename(columns={"PESO_GENERO":"W"}))
    by_estado["W"] = by_estado.groupby("ESTADO")["W"].transform(_normalize_weights)
    by_cadena = (tiendas.groupby(["CADENA","GENERO_CAN"], as_index=False)["PESO_GENERO"].sum()
                       .rename(columns={"PESO_GENERO":"W"}))
    by_cadena["W"] = by_cadena.groupby("CADENA")["W"].transform(_normalize_weights)
    by_global = (tiendas.groupby(["GENERO_CAN"], as_index=False)["PESO_GENERO"].sum()
                       .rename(columns={"PESO_GENERO":"W"}))
    by_global["W"] = _normalize_weights(by_global["W"])
    return by_estado, by_cadena, by_global

def get_mix_for_location(cadena, estado, by_estado, by_cadena, by_global, lambdas=(0.6,0.3,0.1)):
    l1,l2,l3 = lambdas
    pe = by_estado.loc[by_estado["ESTADO"]==str(estado).upper(), ["GENERO_CAN","W"]].rename(columns={"W":"W_E"})
    pc = by_cadena.loc[by_cadena["CADENA"]==str(cadena).upper(), ["GENERO_CAN","W"]].rename(columns={"W":"W_C"})
    pg = by_global[["GENERO_CAN","W"]].rename(columns={"W":"W_G"})
    base = pd.DataFrame({"GENERO_CAN": sorted(GENRE_CANONICALS)})
    base = base.merge(pe, on="GENERO_CAN", how="left").merge(pc, on="GENERO_CAN", how="left").merge(pg, on="GENERO_CAN", how="left")
    base = base.fillna(0.0)
    base["W_COMB"] = l1*base["W_E"] + l2*base["W_C"] + l3*base["W_G"]
    s = base["W_COMB"].sum()
    base["W_COMB"] = base["W_COMB"]/s if s>0 else 1.0/len(base)
    return dict(zip(base["GENERO_CAN"], base["W_COMB"]))

def get_contract_for_location(cadena:str, estado:str, tiendas:pd.DataFrame, default:str="contrato regular")->str:
    mask = (tiendas["CADENA"]==str(cadena).upper()) & (tiendas["ESTADO"]==str(estado).upper())
    vals = tiendas.loc[mask & tiendas["TYPE"].ne(""), "TYPE"].head(1)
    if not vals.empty:
        return str(vals.iloc[0]).strip().lower()
    vals = tiendas.loc[tiendas["TYPE"].ne(""), "TYPE"].head(1)
    return str(vals.iloc[0]).strip().lower() if not vals.empty else default

DAYPART_RULES = {("morning",5,11),("midday",11,15),("peak",15,20),("close",20,24),("late",0,5)}
DAYPART_TARGETS = {
    "morning":{"energy":(0.3,0.6),"valence":(0.5,0.9),"tempo":(80,110)},
    "midday":{"energy":(0.4,0.7),"valence":(0.4,0.8),"tempo":(90,115)},
    "peak":{"energy":(0.6,0.9),"valence":(0.4,0.8),"tempo":(105,130)},
    "close":{"energy":(0.3,0.6),"valence":(0.3,0.7),"tempo":(70,95)},
    "late":{"energy":(0.2,0.5),"valence":(0.2,0.5),"tempo":(60,85)},
}

def _tri(x, lo, hi):
    mid = (lo+hi)/2.0
    if x<=lo or x>=hi: return 0.0
    return 1.0 - abs(x-mid)/(hi-lo)*2.0

def score_track_for_daypart(feat:dict, daypart:str)->float:
    t = DAYPART_TARGETS.get(daypart, DAYPART_TARGETS["midday"])
    return 0.4*_tri(feat.get("energy",0.5), *t["energy"]) + 0.3*_tri(feat.get("valence",0.5), *t["valence"]) + 0.3*_tri(feat.get("tempo",100), *t["tempo"])

GENRE_QUERIES = {
    "pop":   ["pop"],
    "rock":  ["rock"],
    "salsa": ["salsa", "salsa romantica", "salsa clasica"],
    "banda": ["banda", "musica banda", "banda sinaloense"],
    "cumbia":["cumbia", "cumbia sonidera", "cumbia villera"],
    "otros": ["latin", "pop latino", "reggaeton", "urbano latino"],
}

def harvest_tracks_from_playlists(sp, query:str, limit_playlists:int=5, per_playlist:int=50, market:Optional[str]="MX"):
    items = []
    try:
        pl_resp = sp.search(q=query, type="playlist", limit=limit_playlists, market=market)
        playlists = (pl_resp or {}).get("playlists", {}).get("items", []) or []
    except Exception:
        playlists = []

    for p in playlists:
        if not isinstance(p, dict): 
            continue
        pid = p.get("id")
        if not pid: 
            continue
        try:
            tracks_resp = sp.playlist_items(pid, market=market, limit=per_playlist)
            track_items = (tracks_resp or {}).get("items", []) or []
        except Exception:
            track_items = []
        for it in track_items:
            if not isinstance(it, dict):
                continue
            tr = (it.get("track") or {}) if it else {}
            if not isinstance(tr, dict): 
                continue
            if tr.get("type") != "track": 
                continue
            items.append(tr)
    return items

def spotify_audio_features_df(sp, track_ids: List[str]) -> pd.DataFrame:
    feats = []
    for chunk in [track_ids[i:i+100] for i in range(0, len(track_ids), 100)]:
        try:
            af = sp.audio_features(chunk)
        except Exception:
            af = []
        feats.extend([f for f in (af or []) if f])
    if not feats: return pd.DataFrame()
    keep = ["id","energy","valence","tempo"]
    return pd.DataFrame([{k: rec.get(k) for k in keep} for rec in feats])

def build_spotify_pool_once(sp, mix_weights:dict, per_genre:int=250, market=None, seed:int=42)->pd.DataFrame:
    import numpy as _np
    rng = _np.random.default_rng(seed)
    rows = []
    for genero, w in mix_weights.items():
        goal = per_genre
        items = []
        for q in GENRE_QUERIES.get(genero, [genero]):
            if len(items) >= goal: break
            try:
                batch = sp.search(q=q, type="track", limit=50, offset=0, market=market)
                batch = (batch or {}).get("tracks",{}).get("items",[]) or []
            except Exception:
                batch = []
            items.extend(batch)
        if len(items) < goal:
            for q in GENRE_QUERIES.get(genero, [genero]):
                if len(items) >= goal: break
                pl_tracks = harvest_tracks_from_playlists(sp, q, limit_playlists=4, per_playlist=50, market=market)
                items.extend(pl_tracks)

        rng.shuffle(items); items = items[:goal]
        if not items: continue
        recs = []
        for it in items:
            if not isinstance(it, dict): continue
            track_id = it.get("id"); uri = it.get("uri") or ""
            name = it.get("name") or ""; album = (it.get("album") or {}).get("name", "") if isinstance(it.get("album"), dict) else ""
            artists_list = it.get("artists") or []
            artists = ", ".join([a.get("name","") for a in artists_list if isinstance(a, dict)]) if isinstance(artists_list, list) else ""
            popularity = it.get("popularity", np.nan)
            explicit = bool(it.get("explicit", False))
            recs.append({"GENERO_CAN": genero, "artist": artists or "Desconocido", "album": album or "Desconocido",
                         "track": name or "Desconocido", "uri": uri, "id": track_id,
                         "popularity": popularity, "explicit": explicit})
        df = pd.DataFrame(recs)
        df["energy"]=np.nan; df["valence"]=np.nan; df["tempo"]=np.nan
        af_ids = [x for x in df["id"].dropna().tolist() if isinstance(x, str)]
        if af_ids:
            af = spotify_audio_features_df(sp, af_ids)
            if not af.empty:
                df = df.merge(af, on="id", how="left", suffixes=("", "_af"))
                for col in ["energy","valence","tempo"]:
                    if f"{col}_af" in df.columns:
                        df[col] = df[f"{col}_af"].combine_first(df[col])
                        df.drop(columns=[f"{col}_af"], inplace=True, errors="ignore")
        df["energy"] = pd.to_numeric(df["energy"], errors="coerce").fillna(0.5)
        df["valence"] = pd.to_numeric(df["valence"], errors="coerce").fillna(0.5)
        df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce").fillna(100)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["GENERO_CAN","artist","album","track","uri","energy","valence","tempo","popularity","explicit"])
    pool = pd.concat(rows, ignore_index=True)
    return pool[["GENERO_CAN","artist","album","track","uri","energy","valence","tempo","popularity","explicit"]]

def build_spotify_pool(sp, mix_weights:dict, per_genre:int=250, market:str="MX", seed:int=42)->pd.DataFrame:
    markets = [market, "MX", "US", None]
    tried = []
    for m in markets:
        if m in tried: continue
        tried.append(m)
        pool = build_spotify_pool_once(sp, mix_weights, per_genre=per_genre, market=m, seed=seed)
        if not pool.empty:
            return pool
    return pool

POPULARITY_THRESHOLDS = {
    "baja":  (0, 33),
    "media": (34, 66),
    "alta":  (67, 100),
}
CONTRACT_TO_TIER = {
    "contrato limitado": "baja",
    "contrato regular": "media",
    "contrato consolidado": "alta",
}

def filter_by_contract_and_explicit(pool: pd.DataFrame, contract_type: str)->pd.DataFrame:
    tier = CONTRACT_TO_TIER.get(str(contract_type).strip().lower(), "media")
    lo, hi = POPULARITY_THRESHOLDS[tier]
    df = pool.copy()
    if "explicit" in df.columns: df = df[df["explicit"] == False]
    if "popularity" in df.columns: df = df[df["popularity"].between(lo, hi, inclusive='both')]
    else: df = df.iloc[0:0]
    return df

def pick_tracks_for_mix(pool_df, mix_weights, n_tracks=60, max_per_artist=2, max_per_album=2, relax=True):
    df = pool_df.copy()
    if df.empty: return df
    if "score" not in df.columns: df["score"]=0.0
    df["w_gen"] = df["GENERO_CAN"].map(mix_weights).fillna(0.0)
    df["rank_score"] = 0.6*df["score"] + 0.4*df["w_gen"]
    df = df.sort_values("rank_score", ascending=False)
    def run(a_lim, al_lim):
        out = []; ac={}; alc={}
        for _, r in df.iterrows():
            a=str(r.get("artist","")).lower().strip(); al=str(r.get("album","")).lower().strip()
            if ac.get(a,0)>=a_lim or alc.get(al,0)>=al_lim: continue
            out.append(r); ac[a]=ac.get(a,0)+1; alc[al]=alc.get(al,0)+1
            if len(out)>=n_tracks: break
        return pd.DataFrame(out)
    for a_lim,al_lim in [(2,2),(3,3),(5,5),(999,999)]:
        pl=run(a_lim,al_lim)
        if len(pl)>=n_tracks or a_lim==999: return pl
    return pd.DataFrame()

app = FastAPI()

@app.post("/api/recommend")
async def recommend(
    excel_file: UploadFile,
    sheet_name: str = Form("Sheet 1"),
    cadena: str = Form("WALMART"),
    estado: str = Form("CIUDAD DE MEXICO"),
    n_tracks: int = Form(60),
    per_genre_pool: int = Form(300),
    market: str = Form("MX"),
    daypart_override: str = Form(None),
    otros_cap: float = Form(0.08),
    floors_pop: str = Form(""),
    floors_cumbia: str = Form(""),
    contract_override: str = Form(""),
):
    content = await excel_file.read()
    tiendas = load_tiendas_from_bytes(content, sheet_name=sheet_name)
    by_estado, by_cadena, by_global = build_aggregates(tiendas)
    mix = get_mix_for_location(cadena, estado, by_estado, by_cadena, by_global, (0.6,0.3,0.1))

    ot = mix.get("otros",0.0)
    if ot>otros_cap:
        excess = ot-otros_cap
        mix["otros"]=otros_cap
        keys=[k for k in mix if k!="otros"]
        rest=sum(mix[k] for k in keys) or 1.0
        for k in keys: mix[k]+=excess*(mix[k]/rest)
        s=sum(mix.values()); mix={k:v/s for k,v in mix.items()} if s>0 else mix

    floors={}
    try:
        if floors_pop: floors["pop"]=float(floors_pop)
        if floors_cumbia: floors["cumbia"]=float(floors_cumbia)
    except: pass
    if floors:
        for g,f in floors.items(): mix[g]=max(mix.get(g,0.0),f)
        s=sum(mix.values()); mix={k:v/s for k,v in mix.items()} if s>0 else mix

    if daypart_override:
        daypart=daypart_override
    else:
        mask=(tiendas["CADENA"]==cadena.upper()) & (tiendas["ESTADO"]==estado.upper())
        row=tiendas.loc[mask].head(1)
        if row.empty: row=tiendas.head(1)
        daypart=compute_daypart_from_row(row.iloc[0])

    contract_type=(contract_override or get_contract_for_location(cadena, estado, tiendas, "contrato regular")).lower()

    sp=get_spotify_client()
    pool=build_spotify_pool(sp, mix, per_genre=per_genre_pool, market=market, seed=42)
    if pool.empty:
        return JSONResponse({"error":"Pool vacío"}, status_code=400)
    pool=filter_by_contract_and_explicit(pool, contract_type)
    if pool.empty:
        return JSONResponse({"error":"Sin candidatos tras filtros"}, status_code=400)

    # scoring
    def score_row(r):
        t=DAYPART_TARGETS.get(daypart, DAYPART_TARGETS["midday"])
        from math import isfinite
        def tri(x, lo, hi):
            mid=(lo+hi)/2.0
            if x<=lo or x>=hi: return 0.0
            return 1.0 - abs(x-mid)/(hi-lo)*2.0
        return 0.4*tri(r["energy"], *t["energy"]) + 0.3*tri(r["valence"], *t["valence"]) + 0.3*tri(r["tempo"], *t["tempo"])
    pool["score"]=pool.apply(score_row, axis=1)

    playlist=pick_tracks_for_mix(pool, mix, n_tracks=n_tracks, max_per_artist=2, max_per_album=2, relax=True)

    keep=["uri","artist","track","album","GENERO_CAN","score","energy","valence","tempo","popularity","explicit"]
    for c in keep:
        if c not in playlist.columns: playlist[c]=np.nan
    playlist=playlist.loc[:, keep].reset_index(drop=True)

    csv_bytes=playlist.to_csv(index=False).encode("utf-8")
    b64=base64.b64encode(csv_bytes).decode("ascii")
    filename=f"playlist_{cadena}_{estado}_{daypart}_{CONTRACT_TO_TIER.get(contract_type,'media')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    dataurl=f"data:text/csv;base64,{b64}"

    return {
        "cadena": cadena,
        "estado": estado,
        "daypart": daypart,
        "contract": contract_type,
        "columns": keep,
        "rows": playlist.values.tolist(),
        "csv_filename": filename,
        "csv_dataurl": dataurl,
    }
