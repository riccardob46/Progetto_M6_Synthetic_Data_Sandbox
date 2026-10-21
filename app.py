"""User Interface: dashboard Streamlit che comunica con il backend FastAPI."""
import io
import os
import time
import base64
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Synthetic Data Sandbox", page_icon="🧬", layout="wide")
st.title("🧬 Synthetic Data Sandbox — VAE Edition")
st.caption("Carica un dataset sensibile, addestra un VAE e genera un dataset sintetico anonimo da interrogare in linguaggio naturale.")

# Stato della sessione (sopravvive ai refresh di Streamlit)
for key, default in {"upload_info": None, "df_real": None, "trained": False, "history": None,
                     "df_synth": None, "report": None, "messages": []}.items():
    st.session_state.setdefault(key, default)


def api_error(response):
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text


# ------------------------------- Sidebar -------------------------------
with st.sidebar:
    st.header("⚙️ Backend")
    try:
        requests.get(API_URL, timeout=3)
        st.success(f"API connessa\n\n{API_URL}")
    except requests.exceptions.RequestException:
        st.error(f"API non raggiungibile su {API_URL}.\n\nAvvia il backend con:\n\n`uvicorn api:app`")
        st.stop()

    st.header("📋 Avanzamento")
    st.write("✅ Dataset caricato" if st.session_state.upload_info else "⬜ Dataset caricato")
    st.write("✅ VAE addestrato" if st.session_state.trained else "⬜ VAE addestrato")
    st.write("✅ Dati sintetici generati" if st.session_state.df_synth is not None else "⬜ Dati sintetici generati")

tab_data, tab_compare, tab_chat = st.tabs(["📤 Dati e addestramento", "📊 Reali vs Sintetici", "💬 Chatbot"])

# =============================== TAB 1: upload e training ===============================
with tab_data:
    st.subheader("1. Carica il dataset")
    uploaded = st.file_uploader("Trascina qui il file CSV", type=["csv"])

    if uploaded is not None and (st.session_state.upload_info is None
                                 or st.session_state.upload_info["filename"] != uploaded.name):
        r = requests.post(f"{API_URL}/upload", files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")})
        if r.ok:
            st.session_state.upload_info = r.json()
            st.session_state.df_real = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            st.session_state.update(trained=False, history=None, df_synth=None, report=None, messages=[])
        else:
            st.error(api_error(r))

    info = st.session_state.upload_info
    if info:
        c1, c2 = st.columns(2)
        c1.metric("Righe", info["rows"])
        c2.metric("Colonne", len(info["columns"]))
        st.dataframe(st.session_state.df_real.head(10))

        st.subheader("2. Configura e addestra il VAE")
        c1, c2 = st.columns(2)
        categorical = c1.multiselect("Colonne categoriche (One-Hot Encoding)", info["columns"],
                                     default=info["suggested_categorical"])
        zero_as_nan = c2.multiselect("Colonne in cui 0 = valore mancante", info["columns"],
                                     default=info["suggested_zero_as_nan"])
        epochs = st.slider("Epoche di addestramento", 50, 1500, 500, step=50)

        if st.button("🚀 Avvia addestramento", type="primary"):
            r = requests.post(f"{API_URL}/train", json={"categorical_cols": categorical,
                                                        "zero_as_nan_cols": zero_as_nan,
                                                        "epochs": epochs})
            if not r.ok:
                st.error(api_error(r))
            else:
                bar = st.progress(0.0, text="Addestramento in corso...")
                while True:
                    s = requests.get(f"{API_URL}/status").json()
                    if s["status"] == "training":
                        loss = f" — loss {s['last_loss']['total']:.4f}" if s["last_loss"] else ""
                        bar.progress(s["epoch"] / s["epochs"], text=f"Epoca {s['epoch']}/{s['epochs']}{loss}")
                        time.sleep(0.5)
                    elif s["status"] == "done":
                        st.session_state.update(trained=True, history=s["history"], df_synth=None,
                                                report=None, messages=[])
                        st.rerun()          # aggiorna la sidebar e mostra le curve
                    else:
                        st.error(f"Errore durante l'addestramento: {s['error']}")
                        break

        if st.session_state.history:
            st.markdown("**Curve di apprendimento**")
            h = pd.DataFrame(st.session_state.history)
            c1, c2, c3 = st.columns(3)
            c1.caption("Loss totale"); c1.line_chart(h["total"])
            c2.caption("Reconstruction Loss (MSE)"); c2.line_chart(h["recon"])
            c3.caption("KL Divergence"); c3.line_chart(h["kl"])

# =============================== TAB 2: confronto ===============================
with tab_compare:
    if not st.session_state.trained:
        st.info("Addestra prima il VAE nella scheda **Dati e addestramento**.")
    else:
        st.subheader("3. Genera il dataset sintetico")
        c1, c2 = st.columns([1, 3])
        n_samples = c1.number_input("Numero di record", 10, 100_000, value=st.session_state.upload_info["rows"])
        if c1.button("🧪 Genera", type="primary"):
            r = requests.get(f"{API_URL}/generate", params={"n_samples": int(n_samples)})
            if r.ok:
                st.session_state.df_synth = pd.read_csv(io.StringIO(r.text))
                st.session_state.report = requests.get(f"{API_URL}/report").json()
                st.session_state.messages = []
                st.rerun()                  # aggiorna la sidebar
            else:
                st.error(api_error(r))

        df_synth = st.session_state.df_synth
        rep = st.session_state.report
        if df_synth is not None and rep:
            c2.download_button("⬇️ Scarica il dataset sintetico (CSV)", df_synth.to_csv(index=False),
                               file_name="synthetic_data.csv", mime="text/csv")
            c2.dataframe(df_synth.head(10))

            # --- Privacy ---
            st.subheader("🔒 Privacy Check")
            p = rep["privacy"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Record identici ai reali", f"{p['exact_copies']} / {p['n_synthetic']}")
            c2.metric("DCR mediana sintetico→reale", p["dcr_median_synth_real"],
                      delta=round(p["dcr_median_synth_real"] - p["dcr_median_real_real"], 4))
            c3.metric("DCR mediana reale→reale (baseline)", p["dcr_median_real_real"])
            if p["exact_copies"] == 0 and p["dcr_median_synth_real"] >= 0.8 * p["dcr_median_real_real"]:
                st.success("Nessuna copia dei record reali: i dati sintetici sono distanti dai reali quanto i reali tra loro.")
            else:
                st.warning("Possibile memorizzazione dei dati reali: valuta meno epoche o un beta più alto.")

            # --- Tabella comparativa ---
            st.subheader("📋 Tabella comparativa Dati Reali vs Sintetici")
            st.dataframe(pd.DataFrame(rep["stats"]))

            # --- Correlazioni ---
            st.subheader("🔗 Matrici di correlazione")
            st.caption(f"Differenza media assoluta tra le correlazioni: **{rep['corr_diff_mean']}**")
            corr_real, corr_synth = pd.DataFrame(rep["corr_real"]), pd.DataFrame(rep["corr_synth"])
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            kw = dict(annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, cbar=False)
            sns.heatmap(corr_real, ax=axes[0], **kw); axes[0].set_title("Dati Reali")
            sns.heatmap(corr_synth, ax=axes[1], **kw); axes[1].set_title("Dati Sintetici")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # --- Distribuzioni ---
            st.subheader("📈 Distribuzioni")
            df_real = st.session_state.df_real.copy()
            for col in zero_as_nan:
                df_real[col] = df_real[col].replace(0, np.nan)        # zeri = mancanti, non li disegniamo
            numeric = [c for c in df_synth.select_dtypes("number").columns if c in df_real.columns]
            default = [c for c in ["Age", "Glucose", "BMI", "Insulin"] if c in numeric] or numeric[:4]
            chosen = st.multiselect("Variabili da confrontare", numeric, default=default)
            if chosen:
                fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 4), squeeze=False)
                for ax, col in zip(axes[0], chosen):
                    bins = np.histogram_bin_edges(df_real[col].dropna(), bins=30)
                    ax.hist(df_real[col].dropna(), bins=bins, alpha=0.5, density=True, label="Reale")
                    ax.hist(df_synth[col], bins=bins, alpha=0.5, density=True, label="Sintetico")
                    ax.set_title(col); ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# =============================== TAB 3: chatbot ===============================
with tab_chat:
    if st.session_state.df_synth is None:
        st.info("Genera prima il dataset sintetico nella scheda **Reali vs Sintetici**.")
    else:
        st.caption("Fai domande sul dataset **sintetico**: l'LLM genera codice Pandas + Matplotlib, "
                   "il backend lo esegue in modo sicuro e mostra il risultato. I dati reali non vengono mai inviati all'LLM.")

        def render(msg):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                    return
                if msg.get("explanation"):
                    st.markdown(msg["explanation"])
                if msg.get("output"):
                    st.code(msg["output"], language="text")
                if msg.get("image"):
                    st.image(base64.b64decode(msg["image"]))
                if msg.get("error"):
                    st.error(msg["error"])
                if msg.get("code"):
                    with st.expander(f"Codice generato (tentativi: {msg.get('attempts', 1)})"):
                        st.code(msg["code"], language="python")

        for msg in st.session_state.messages:
            render(msg)

        question = st.chat_input("Es. Qual è la correlazione tra glicemia e BMI?")
        if question:
            user_msg = {"role": "user", "content": question}
            st.session_state.messages.append(user_msg)
            render(user_msg)
            with st.spinner("L'analista sta lavorando..."):
                r = requests.post(f"{API_URL}/ask", json={"question": question})
            answer = {"role": "assistant", **r.json()} if r.ok else {"role": "assistant", "error": api_error(r)}
            st.session_state.messages.append(answer)
            render(answer)
