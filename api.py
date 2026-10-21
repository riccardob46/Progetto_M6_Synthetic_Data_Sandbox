"""Analytic Backend: API FastAPI per caricamento, addestramento, generazione e analisi."""
import io
import os
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors

from data_engine import DataEngine, infer_decimals, suggest_config
from vae import VAE, train_vae, estimate_noise_std, generate_synthetic
from llm_analyst import analyze, PIMA_DESCRIPTIONS

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
REAL_CSV = os.path.join(DATA_DIR, "real.csv")
SYNTH_CSV = os.path.join(DATA_DIR, "synthetic.csv")

app = FastAPI(title="Synthetic Data Sandbox API", version="1.0")

# Stato dell'applicazione (in memoria: si azzera al riavvio del server)
STATE = {
    "df_real": None, "engine": None, "X": None, "model": None, "noise_std": None, "df_synth": None,
    "status": "idle", "epoch": 0, "epochs": 0, "history": None, "error": None,
}


def to_json(df):
    """Converte un DataFrame in oggetti JSON nativi (gestisce i tipi numpy e i NaN)."""
    return json.loads(df.to_json())


# ------------------------------- Modelli delle richieste -------------------------------
class TrainRequest(BaseModel):
    categorical_cols: List[str] = []
    zero_as_nan_cols: List[str] = []
    epochs: int = Field(500, ge=10, le=3000)
    hidden_dim: int = 128
    latent_dim: int = 8
    beta: float = 0.005
    batch_size: int = 64
    lr: float = 1e-3


class AskRequest(BaseModel):
    question: str


# ------------------------------- Endpoint -------------------------------
@app.get("/")
def root():
    return {"message": "Synthetic Data Sandbox API attiva. Documentazione su /docs"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Carica il CSV con i dati reali."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Il file deve essere in formato CSV.")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"Impossibile leggere il CSV: {e}")
    df = df.dropna(axis=1, how="all")
    if df.shape[0] < 50 or df.shape[1] < 2:
        raise HTTPException(400, "Il dataset deve avere almeno 50 righe e 2 colonne.")

    df.to_csv(REAL_CSV, index=False)
    STATE.update(df_real=df, engine=None, X=None, model=None, noise_std=None, df_synth=None,
                 status="idle", epoch=0, epochs=0, history=None, error=None)
    categorical, zero_as_nan = suggest_config(df)
    return {
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "suggested_categorical": categorical,
        "suggested_zero_as_nan": zero_as_nan,
    }


def _train_job(cfg: TrainRequest):
    """Addestramento eseguito in background."""
    try:
        df = STATE["df_real"]
        numeric_cols = [c for c in df.columns if c not in cfg.categorical_cols]
        engine = DataEngine(cfg.categorical_cols, cfg.zero_as_nan_cols, infer_decimals(df, numeric_cols))
        X = engine.fit_transform(df)
        model = VAE(engine.n_features, len(engine.numeric_cols), engine.cat_group_sizes,
                    cfg.hidden_dim, cfg.latent_dim)

        def progress(epoch, history):
            STATE["epoch"] = epoch
            STATE["history"] = history

        history = train_vae(model, X, cfg.epochs, cfg.batch_size, cfg.lr, cfg.beta, callback=progress)
        noise_std = estimate_noise_std(model, X, len(engine.numeric_cols))
        STATE.update(engine=engine, X=X, model=model, noise_std=noise_std,
                     history=history, status="done", df_synth=None)
    except Exception as e:
        STATE.update(status="error", error=str(e))


@app.post("/train")
def train(cfg: TrainRequest, background_tasks: BackgroundTasks):
    """Avvia l'addestramento del VAE (in background)."""
    if STATE["df_real"] is None:
        raise HTTPException(400, "Nessun dataset caricato: usa prima /upload.")
    if STATE["status"] == "training":
        raise HTTPException(409, "Addestramento già in corso.")
    columns = set(STATE["df_real"].columns)
    unknown = (set(cfg.categorical_cols) | set(cfg.zero_as_nan_cols)) - columns
    if unknown:
        raise HTTPException(400, f"Colonne inesistenti: {sorted(unknown)}")

    STATE.update(status="training", epoch=0, epochs=cfg.epochs, history=None, error=None)
    background_tasks.add_task(_train_job, cfg)
    return {"message": "Addestramento avviato", "epochs": cfg.epochs}


@app.get("/status")
def status():
    """Stato dell'addestramento (per la barra di avanzamento della UI)."""
    h = STATE["history"]
    return {
        "status": STATE["status"],
        "epoch": STATE["epoch"],
        "epochs": STATE["epochs"],
        "error": STATE["error"],
        "history": h if STATE["status"] == "done" else None,
        "last_loss": {k: v[-1] for k, v in h.items()} if h and h["total"] else None,
    }


@app.get("/generate")
def generate(n_samples: int = 1000, seed: Optional[int] = 42):
    """Genera il dataset sintetico e lo restituisce come file CSV da scaricare."""
    if STATE["model"] is None:
        raise HTTPException(400, "Modello non addestrato: usa prima /train.")
    if not 1 <= n_samples <= 100_000:
        raise HTTPException(400, "n_samples deve essere tra 1 e 100000.")

    df_synth = generate_synthetic(STATE["model"], STATE["engine"], n_samples, STATE["noise_std"], seed)
    df_synth = df_synth[STATE["df_real"].columns]          # stesso ordine di colonne del reale
    df_synth.to_csv(SYNTH_CSV, index=False)
    STATE["df_synth"] = df_synth

    return StreamingResponse(
        io.StringIO(df_synth.to_csv(index=False)),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=synthetic_data.csv"},
    )


@app.get("/report")
def report():
    """Metriche di qualità e privacy: statistiche, correlazioni e Distance to Closest Record."""
    if STATE["df_synth"] is None:
        raise HTTPException(400, "Nessun dataset sintetico: usa prima /generate.")
    engine = STATE["engine"]
    df_real = engine.inverse_transform(STATE["X"])[STATE["df_real"].columns]   # reale "pulito"
    df_synth = STATE["df_synth"]

    stats = pd.concat({"Reale": df_real.describe().T[["mean", "std", "min", "max"]],
                       "Sintetico": df_synth.describe().T[["mean", "std", "min", "max"]]}, axis=1).round(3)
    stats.columns = [f"{a} {b}" for a, b in stats.columns]

    corr_real = df_real.corr(numeric_only=True)
    corr_synth = df_synth.corr(numeric_only=True)
    upper = np.triu_indices_from(corr_real, k=1)
    corr_diff = (corr_real - corr_synth).abs().values[upper]

    real_set = set(map(tuple, df_real.values.tolist()))
    exact_copies = sum(tuple(r) in real_set for r in df_synth.values.tolist())
    X_real, X_synth = engine.transform(df_real), engine.transform(df_synth)
    nn_real = NearestNeighbors(n_neighbors=2).fit(X_real)
    dcr_synth = nn_real.kneighbors(X_synth, n_neighbors=1)[0][:, 0]
    dcr_real = nn_real.kneighbors(X_real, n_neighbors=2)[0][:, 1]

    return {
        "stats": to_json(stats),
        "corr_real": to_json(corr_real.round(3)),
        "corr_synth": to_json(corr_synth.round(3)),
        "corr_diff_mean": round(float(np.nanmean(corr_diff)), 4),
        "privacy": {
            "exact_copies": int(exact_copies),
            "n_synthetic": int(len(df_synth)),
            "dcr_median_synth_real": round(float(np.median(dcr_synth)), 4),
            "dcr_median_real_real": round(float(np.median(dcr_real)), 4),
            "dcr_min_synth_real": round(float(dcr_synth.min()), 4),
        },
    }


@app.post("/ask")
def ask(req: AskRequest):
    """Chatbot: domanda in linguaggio naturale sul dataset SINTETICO."""
    if STATE["df_synth"] is None:
        raise HTTPException(400, "Nessun dataset sintetico: usa prima /generate.")
    if not req.question.strip():
        raise HTTPException(400, "La domanda è vuota.")
    df_synth = STATE["df_synth"]
    descriptions = PIMA_DESCRIPTIONS if set(PIMA_DESCRIPTIONS).issubset(df_synth.columns) else None
    try:
        return analyze(req.question, df_synth, SYNTH_CSV, descriptions)
    except Exception as e:
        raise HTTPException(502, f"Errore nella comunicazione con l'LLM: {e}")
