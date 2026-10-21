"""Natural Language Analyst: l'LLM genera codice Pandas/Matplotlib, il backend lo esegue in modo sicuro."""
import os, re, ast, sys, json, subprocess
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-120b"
RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_runner.py")

PIMA_DESCRIPTIONS = {
    "Pregnancies": "numero di gravidanze",
    "Glucose": "glicemia (mg/dL) dopo test di tolleranza al glucosio",
    "BloodPressure": "pressione diastolica (mm Hg)",
    "SkinThickness": "spessore della plica cutanea del tricipite (mm)",
    "Insulin": "insulina sierica a 2 ore (mu U/ml)",
    "BMI": "indice di massa corporea (kg/m^2)",
    "DiabetesPedigreeFunction": "indice di predisposizione genetica al diabete",
    "Age": "età (anni)",
    "Outcome": "diagnosi di diabete (0 = no, 1 = sì)",
}

# ------------------------- Esecuzione sicura -------------------------
ALLOWED_MODULES = {"pandas", "numpy", "matplotlib"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "memoryview", "os", "sys", "subprocess", "shutil", "pathlib", "builtins",
}
FORBIDDEN_ATTRS = {
    "savefig", "imread", "eval", "query", "load", "save", "savez", "savetxt",
    "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "system", "popen",
}
ALLOWED_TO_METHODS = {"to_frame", "to_numpy", "to_string", "to_dict", "to_list", "to_markdown"}
SAFE_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format",
    "int", "isinstance", "len", "list", "map", "max", "min", "print", "range",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "ValueError", "KeyError", "TypeError", "Exception",
]


def validate_code(code):
    """Analisi statica (AST): restituisce (codice_pulito, None) oppure (None, errore)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None, f"Errore di sintassi: {e}"

    new_body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in modules:
                if m.split(".")[0] not in ALLOWED_MODULES:
                    return None, f"Import non consentito: '{m}'. Usa solo df, pd, np e plt."
            continue
        new_body.append(node)
    tree.body = new_body

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None, "Gli import sono consentiti solo all'inizio del codice."
        if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef)):
            return None, f"Costrutto non consentito: {type(node).__name__}."
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return None, f"Nome non consentito: '{node.id}'."
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_"):
                return None, f"Accesso ad attributi privati non consentito: '{a}'."
            if a in FORBIDDEN_ATTRS or a.startswith("read_") or (a.startswith("to_") and a not in ALLOWED_TO_METHODS):
                return None, f"Metodo non consentito (accesso a file o esecuzione dinamica): '{a}'."
    return ast.unparse(tree), None


def run_code_safely(code, csv_path, timeout=20):
    """Valida il codice e lo esegue in un processo separato con timeout."""
    safe_code, error = validate_code(code)
    if error:
        return {"ok": False, "output": "", "image": None, "error": f"Codice rifiutato: {error}"}

    payload = json.dumps({"code": safe_code, "csv_path": os.path.abspath(csv_path), "builtins": SAFE_BUILTINS})
    try:
        proc = subprocess.run([sys.executable, RUNNER_PATH], input=payload.encode("utf-8"),
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "", "image": None, "error": f"Timeout: esecuzione oltre {timeout} secondi."}
    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "output": "", "image": None,
                "error": "Errore del processo di esecuzione: " + proc.stderr.decode("utf-8", "replace")[-500:]}


# ------------------------- Prompt e LLM -------------------------
def build_system_prompt(df, descriptions=None):
    """Costruisce il system prompt a partire dalla struttura REALE del DataFrame."""
    descriptions = descriptions or {}
    lines = []
    for col in df.columns:
        s = df[col]
        desc = descriptions.get(col, "")
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            stats = f"min={s.min():.4g}, max={s.max():.4g}, media={s.mean():.4g}"
        else:
            stats = "valori: " + ", ".join(map(str, sorted(s.unique())[:10]))
        lines.append(f"- {col} ({s.dtype}){': ' + desc if desc else ''} | {stats}")
    schema = "\n".join(lines)

    return f"""Sei un "Natural Language Analyst": un data analyst esperto in Python, Pandas e Matplotlib.
Rispondi alle domande dell'utente su un dataset SINTETICO (generato da un VAE) caricato nel DataFrame `df`.

STRUTTURA DEL DATASET ({len(df)} righe, {df.shape[1]} colonne):
{schema}

REGOLE PER IL CODICE:
1. Le variabili `df`, `pd` (pandas), `np` (numpy) e `plt` (matplotlib.pyplot) sono GIÀ disponibili: non serve importare nulla.
2. Usa SOLO le colonne elencate sopra, con i nomi esatti.
3. Stampa sempre il risultato numerico con print(), in modo leggibile.
4. Se un grafico aiuta a rispondere, crea UNA sola figura con plt (titolo e assi in italiano). Non chiamare plt.savefig().
5. Vietato: leggere o scrivere file, usare import diversi da pandas/numpy/matplotlib, usare eval, query, open, os, sys.
6. Il codice deve essere in UN SOLO blocco ```python ... ```.

FORMATO DELLA RISPOSTA:
- Una o due frasi in italiano che spiegano l'approccio.
- Il blocco di codice.

Se la domanda riguarda informazioni NON presenti nel dataset (ad esempio colonne che non esistono),
NON scrivere codice: spiega in italiano che quei dati non sono disponibili ed elenca le colonne utilizzabili."""


_client = None


def get_client():
    """Crea il client OpenAI (provider Groq) alla prima richiesta."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY non trovata: controlla il file .env")
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    return _client


def ask_llm(messages, temperature=0.1):
    """Unico punto di contatto con il provider LLM."""
    response = get_client().chat.completions.create(model=LLM_MODEL, messages=messages, temperature=temperature)
    return response.choices[0].message.content or ""


def extract_code(text):
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def analyze(question, df, csv_path, descriptions=None, max_attempts=3):
    """Domanda -> codice generato dall'LLM -> esecuzione sicura -> risultato (con auto-correzione)."""
    messages = [
        {"role": "system", "content": build_system_prompt(df, descriptions)},
        {"role": "user", "content": question},
    ]
    result = {"question": question, "explanation": "", "code": None, "output": "",
              "image": None, "error": None, "attempts": 0}

    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt
        reply = ask_llm(messages)
        code = extract_code(reply)
        result["explanation"] = re.sub(r"```.*?```", "", reply, flags=re.DOTALL).strip()

        if code is None:
            result["error"] = None
            return result

        result["code"] = code
        execution = run_code_safely(code, csv_path)
        result.update(output=execution["output"], image=execution["image"], error=execution["error"])
        if execution["ok"]:
            return result

        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"Il codice ha prodotto questo errore:\n{execution['error']}\n"
                                        "Correggilo rispettando le regole e restituisci il codice completo."},
        ]
    return result
