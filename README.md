# Synthetic Data Sandbox (VAE Edition)

Piattaforma per caricare un dataset tabellare sensibile, addestrare un Variational Autoencoder (PyTorch)
e generare un dataset sintetico anonimo, interrogabile in linguaggio naturale tramite LLM.

## Struttura
| File | Componente |
|---|---|
| `synthetic_data_sandbox.ipynb` | Sviluppo completo e analisi (Fasi 1–5) |
| `data_engine.py` | Data Engine: gestione nulli, MinMaxScaler, One-Hot Encoding |
| `vae.py` | Generative Core: Encoder, Reparameterization Trick, Decoder, Loss |
| `llm_analyst.py`, `sandbox_runner.py` | Natural Language Analyst con esecuzione sicura del codice |
| `api.py` | Analytic Backend (FastAPI) |
| `app.py` | User Interface (Streamlit) |

## Installazione
```bash
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Copia `.env.example` in `.env` e inserisci la chiave Groq (gratuita su console.groq.com).

## Avvio
Terminale 1 – backend:
```bash
python -m uvicorn api:app
```
Terminale 2 – interfaccia:
```bash
python -m streamlit run app.py
```
Documentazione API: http://127.0.0.1:8000/docs — Interfaccia: http://localhost:8501

## Endpoint
| Metodo | Endpoint | Descrizione |
|---|---|---|
| POST | `/upload` | Carica il CSV con i dati reali |
| POST | `/train` | Avvia l'addestramento del VAE (in background) |
| GET | `/status` | Avanzamento dell'addestramento |
| GET | `/generate` | Genera e scarica il dataset sintetico (CSV) |
| GET | `/report` | Metriche di qualità e privacy |
| POST | `/ask` | Domanda in linguaggio naturale sui dati sintetici |

## Dataset
Pima Indians Diabetes (UCI / Kaggle): `pima_real.csv` è generato dal notebook (cella 43).