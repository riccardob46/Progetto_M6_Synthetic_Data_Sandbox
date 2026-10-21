# Eseguito in un processo separato: riceve il codice via stdin e restituisce un JSON
import sys, io, json, base64, builtins, contextlib
import matplotlib
matplotlib.use("Agg")                     # backend senza finestre: disegna in memoria
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
df = pd.read_csv(payload["csv_path"])
safe_builtins = {n: getattr(builtins, n) for n in payload["builtins"]}
namespace = {"__builtins__": safe_builtins, "df": df, "pd": pd, "np": np, "plt": plt}

result = {"ok": True, "output": "", "image": None, "error": None}
buffer = io.StringIO()
try:
    with contextlib.redirect_stdout(buffer):              # cattura i print()
        exec(compile(payload["code"], "<codice_llm>", "exec"), namespace)
    if plt.get_fignums():                                  # se è stato creato un grafico
        img = io.BytesIO()
        plt.gcf().savefig(img, format="png", dpi=110, bbox_inches="tight")
        result["image"] = base64.b64encode(img.getvalue()).decode("ascii")
except Exception as e:
    result["ok"] = False
    result["error"] = f"{type(e).__name__}: {e}"
result["output"] = buffer.getvalue()[:5000]
sys.stdout.buffer.write(json.dumps(result).encode("utf-8"))
