"""
comun.py — Utilidades compartidas del análisis adicional (Fase 1).

Este módulo NO importa ni modifica el código de Marco (trader_rl.py, experimentos.py,
graficas.py). Contiene réplicas independientes y verificadas de:

  * el entorno SimpleTradingEnv y los 4 generadores del enunciado (copia literal del
    notebook del profesor, sin dependencia de gymnasium),
  * la carga restringida de los .pkl de resultados,
  * la reconstrucción de políticas greedy a partir de las tablas Q,
  * utilidades estadísticas (descomposición de varianza, bootstrap, cuantil t con 4 g.l.).

Dependencias: numpy, matplotlib (sólo para figuras). Probado con Python 3.14.3,
numpy 2.4.2, matplotlib 3.10.8. No se usa scipy ni pandas.
"""
import csv
import json
import pickle
import pickletools
import sys
import types
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parent
RES_ORIG = RAIZ / "resultados"
SAL = CARPETA / "resultados"
FIG = CARPETA / "figuras"

GENERADORES = ["ciclico", "browniano", "reversion", "regimenes"]
ALGORITMOS = ["qlearning", "sarsa"]
ETIQUETA_GEN = {"ciclico": "Tendencia + ciclo", "browniano": "Browniano geométrico",
                "reversion": "Reversión a la media", "regimenes": "Regímenes alcista/bajista"}
ETIQUETA_AG = {"qlearning": "Q-Learning", "sarsa": "SARSA", "aleatorio": "Aleatorio",
               "buy&hold": "Buy & Hold", "efectivo": "Efectivo"}
# Protocolo original (experimentos.py: SEED_VAL, N_VAL, SEED_TEST, N_TEST)
SEED_VAL, N_VAL = 2024, 50
SEED_TEST, N_TEST = 7777, 100
N_STEPS = 500
SEMILLAS = range(5)
BOOT_SEED = 20261028  # semilla explícita de todos los procedimientos bootstrap
N_BOOT = 10000


def asegurar_carpetas():
    SAL.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)


# =============================================================================
# 1. Deserialización restringida
# =============================================================================
PERMITIDOS = {("numpy._core.numeric", "_frombuffer"), ("numpy", "dtype")}
OPCODES_PROHIBIDOS = {"INST", "OBJ", "GLOBAL", "EXT1", "EXT2", "EXT4"}


class _UnpicklerRestringido(pickle.Unpickler):
    """Sólo permite reconstruir arrays de NumPy; cualquier otro global lanza error."""

    def find_class(self, modulo, nombre):
        if (modulo, nombre) in PERMITIDOS:
            return super().find_class(modulo, nombre)
        raise pickle.UnpicklingError(f"global bloqueado: {modulo}.{nombre}")


def inspeccionar_pickle(ruta):
    """Recorre los opcodes SIN ejecutar nada. Devuelve el conjunto de opcodes usados."""
    with open(ruta, "rb") as f:
        datos = f.read()
    usados = {op.name for op, _, _ in pickletools.genops(datos)}
    prohibidos = usados & OPCODES_PROHIBIDOS
    if prohibidos:
        raise RuntimeError(f"{ruta}: opcodes no permitidos {prohibidos}")
    return usados


def cargar_pickle(nombre):
    ruta = RES_ORIG / f"{nombre}.pkl"
    inspeccionar_pickle(ruta)
    with open(ruta, "rb") as f:
        return _UnpicklerRestringido(f).load()


# =============================================================================
# 2. Generadores de precios (copia literal del notebook del profesor)
# =============================================================================
def generar_precios(n_steps=1000):
    t = np.linspace(0, 50, n_steps)
    drift = 0.0005 * t
    ciclo = np.sin(t) * 0.05
    ruido = np.random.normal(0, 0.1, n_steps)
    retornos = drift + ciclo + ruido
    return 100 * np.cumprod(1 + retornos)


def generar_precios_browniano(n_steps=1000, start_price=100.0, mu=0.0001, sigma=0.02):
    dt = 1.0
    retornos = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
    return start_price * np.cumprod(np.exp(retornos))


def generar_precios_reversion_media(n_steps=1000, start_price=100.0, mean_price=100.0, kappa=0.05, volatility=0.02):
    precios = np.zeros(n_steps)
    precios[0] = start_price
    for t in range(1, n_steps):
        delta = kappa * (mean_price - precios[t - 1]) + np.random.normal(0, volatility * mean_price)
        precios[t] = precios[t - 1] + delta
        precios[t] = max(precios[t], 0.1)
    return precios


def generar_precios_regimenes(n_steps=1000, start_price=100.0):
    precios = np.zeros(n_steps)
    precios[0] = start_price
    estado = 0
    retornos = []
    for _ in range(n_steps):
        if np.random.rand() < 0.02:
            estado = 1 - estado
        if estado == 0:
            ret = np.random.normal(0.001, 0.01)
        else:
            ret = np.random.normal(-0.0015, 0.025)
        retornos.append(ret)
    retornos = np.array(retornos)
    precios = start_price * np.cumprod(1 + retornos)
    return np.maximum(precios, 0.1)


GEN = {"ciclico": generar_precios, "browniano": generar_precios_browniano,
       "reversion": generar_precios_reversion_media, "regimenes": generar_precios_regimenes}


def generar_series(gen_nombre, n_series, seed, n_steps=N_STEPS):
    """Mismo procedimiento que el original: np.random.seed(seed) y n series consecutivas.
    El estado global de NumPy se fija justo antes de generar, para no depender de llamadas previas."""
    np.random.seed(seed)
    return [GEN[gen_nombre](n_steps) for _ in range(n_series)]


# =============================================================================
# 3. Entorno: réplica + verificación contra el código original del notebook
# =============================================================================
def calcular_bins(precios, cortes):
    """Bin de tendencia en cada paso de decisión (índice i <-> current_step = i + 5).
    Reproduce _get_obs: tendencia float64 -> recortada -> float32 (como la observación)."""
    v = np.lib.stride_tricks.sliding_window_view(precios, 6)
    m = v.mean(axis=1)
    tr = np.clip((v[:, -1] - m) / m, -1.0, 1.0).astype(np.float32)
    return np.digitize(tr, cortes)


def simular(precios, bins, decidir, visitas=None):
    """Réplica fiel de SimpleTradingEnv.step(). `decidir(bin, posicion) -> accion`.

    Devuelve (recompensa acumulada, balance final, operaciones efectivas)."""
    p = precios.tolist()
    n = len(p) - 1                      # max_steps
    pos, cash, shares = 0, 1000.0, 0.0
    paso, rec, ops, bal = 5, 0.0, 0, 1000.0
    while paso < n:
        b = int(bins[paso - 5])
        if visitas is not None:
            visitas[pos][b] += 1
        a = decidir(b, pos)
        prev = p[paso]
        paso += 1
        cur = p[paso]
        r = 0.0
        if a == 2 and pos == 0:
            comision = cash * 0.001
            shares = (cash - comision) / prev
            cash = 0.0
            pos = 1
            r -= 0.001
            ops += 1
        elif a == 0 and pos == 1:
            venta = shares * prev
            comision = venta * 0.001
            cash = venta - comision
            shares = 0.0
            pos = 0
            r -= 0.001
            ops += 1
        if pos == 1:
            r += (cur - prev) / prev
        bal = cash + shares * cur if pos == 1 else cash
        rec += r
    return rec, bal, ops


def cargar_entorno_original():
    """Extrae la clase SimpleTradingEnv de la celda 4 del notebook del enunciado y la ejecuta con
    stubs mínimos de gymnasium (no está instalado). Sirve para validar la réplica contra el
    código real del profesor."""
    with open(RAIZ / "taller_trader.ipynb", encoding="utf-8", errors="replace") as f:
        nb = json.load(f)
    fuente = "".join(nb["cells"][4]["source"])
    assert "class SimpleTradingEnv" in fuente

    class _Env:
        def reset(self, seed=None):
            pass

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Box:
        def __init__(self, **kw):
            pass

    gym = types.SimpleNamespace(Env=_Env)
    spaces = types.SimpleNamespace(Discrete=_Discrete, Box=_Box)
    ns = {"gym": gym, "spaces": spaces, "np": np}
    exec(compile(fuente, "notebook_celda4", "exec"), ns)
    return ns["SimpleTradingEnv"]


# =============================================================================
# 4. Políticas
# =============================================================================
def politica_desde_Q(Q):
    """Política greedy sobre acciones válidas. Devuelve (comprar[b], vender[b], empate0[b], empate1[b]).
    comprar[b]: con posición 0 elige COMPRAR (Q[b,0,2] > Q[b,0,1]); si hay empate exacto el original
    desempata al azar -> se marca en empate0 y aquí se asigna MANTENER."""
    comprar = Q[:, 0, 2] > Q[:, 0, 1]
    vender = Q[:, 1, 0] > Q[:, 1, 1]
    return comprar, vender, Q[:, 0, 2] == Q[:, 0, 1], Q[:, 1, 0] == Q[:, 1, 1]


def decisor_tabla(Q):
    comprar, vender, _, _ = politica_desde_Q(Q)
    c, v = comprar.tolist(), vender.tolist()
    return lambda b, pos: (2 if c[b] else 1) if pos == 0 else (0 if v[b] else 1)


def evaluar_tabla(Q, cortes, series, visitas=None):
    """Reconstruye la política de una tabla Q y la juega sobre las series. -> arrays (rec, bal, ops)."""
    dec = decisor_tabla(Q)
    out = [simular(p, calcular_bins(p, cortes), dec, visitas) for p in series]
    rec, bal, ops = (np.array(x) for x in zip(*out))
    return rec, bal, ops


def evaluar_decisor(decidir, series, cortes):
    out = [simular(p, calcular_bins(p, cortes), decidir) for p in series]
    rec, bal, ops = (np.array(x) for x in zip(*out))
    return rec, bal, ops


def ganadores(resultados):
    """Misma lógica que experimentos.ganadores(): variante con mayor log-retorno medio de VALIDACIÓN
    (media de las 5 semillas); en empate gana la primera encontrada. -> {(gen, algo): [runs]}"""
    grupos = {}
    for r in resultados:
        grupos.setdefault((r["generador"], r["algoritmo"], r["variante"]), []).append(r)
    mejores = {}
    for (g, a, _), runs in grupos.items():
        puntuacion = np.mean([r["val"]["logret"].mean() for r in runs])
        if (g, a) not in mejores or puntuacion > mejores[(g, a)][0]:
            mejores[(g, a)] = (puntuacion, runs)
    return {k: sorted(runs, key=lambda r: r["seed"]) for k, (_, runs) in mejores.items()}


# =============================================================================
# 5. Estadística
# =============================================================================
def t4_cdf(t):
    """CDF de la t de Student con 4 g.l. (forma cerrada): F = 1/2 + 3x/8 (1 - x²/12), x = t/sqrt(1+t²/4)."""
    x = t / np.sqrt(1.0 + t * t / 4.0)
    return 0.5 + 3.0 * x / 8.0 * (1.0 - x * x / 12.0)


def t4_ppf(p):
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t4_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t4_pvalor(t):
    return 2.0 * (1.0 - t4_cdf(abs(t)))


T4_95 = t4_ppf(0.975)  # 2.7764...


def resumen_brazo(L):
    """L: matriz (S semillas, J series). Devuelve media, varianzas y componentes.

    var_cond : varianza de la media condicional a las S políticas (sólo muestreo de series).
               = var_j(media por serie sobre semillas) / J
    var_pop  : varianza de la media si se repite TODO (nuevas semillas y nuevas series), estimador
               de momentos del modelo de efectos aleatorios cruzados semilla x serie sin réplica:
               Var = (MS_s + MS_j - MS_e) / (S J). Se trunca por debajo en var_cond.
    """
    L = np.atleast_2d(np.asarray(L, dtype=float))
    S, J = L.shape
    media = L.mean()
    por_serie = L.mean(axis=0)
    var_cond = por_serie.var(ddof=1) / J
    if S == 1:
        return dict(S=S, J=J, media=media, var_cond=var_cond, var_pop=var_cond, ms_s=0.0, ms_j=var_cond * J, ms_e=0.0)
    por_semilla = L.mean(axis=1)
    ms_s = J * por_semilla.var(ddof=1)
    ms_j = S * por_serie.var(ddof=1)
    resid = L - por_semilla[:, None] - por_serie[None, :] + media
    ms_e = (resid**2).sum() / ((S - 1) * (J - 1))
    var_pop = max((ms_s + ms_j - ms_e) / (S * J), var_cond)
    return dict(S=S, J=J, media=media, var_cond=var_cond, var_pop=var_pop, ms_s=ms_s, ms_j=ms_j, ms_e=ms_e)


def ic_bootstrap_series(x, rng, B=N_BOOT):
    """IC95 percentil de la media de x (por serie) remuestreando series con reemplazo."""
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    medias = x[idx].mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def ic_bootstrap_jerarquico(LA, LB, rng, B=N_BOOT):
    """Bootstrap jerárquico de la diferencia de medias A - B (B puede ser None -> sólo A).

    Nivel 1: remuestrea semillas de cada brazo (con reemplazo, independientemente).
    Nivel 2: remuestrea series (con reemplazo) y las MISMAS series se usan en ambos brazos (emparejamiento).
    Con pocas semillas (5) este intervalo es optimista: se reporta como contraste del IC t-4 g.l."""
    LA = np.atleast_2d(LA)
    S_A, J = LA.shape
    if LB is None:
        LB = np.zeros((1, J))
    LB = np.atleast_2d(LB)
    S_B = LB.shape[0]
    dif = np.empty(B)
    for b in range(B):
        sa = rng.integers(0, S_A, S_A)
        sb = rng.integers(0, S_B, S_B)
        js = rng.integers(0, J, J)
        dif[b] = LA[sa][:, js].mean() - LB[sb][:, js].mean()
    return np.percentile(dif, [2.5, 97.5])


def escribir_csv(nombre, filas, columnas=None):
    asegurar_carpetas()
    if columnas is None:
        columnas = list(filas[0].keys())
    with open(SAL / nombre, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in fila.items()})


def leer_csv(nombre):
    with open(SAL / nombre, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tabla_md(filas, columnas, formatos=None):
    formatos = formatos or {}
    out = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for fila in filas:
        celdas = []
        for c in columnas:
            v = fila[c]
            fmt = formatos.get(c)
            celdas.append(format(v, fmt) if fmt and isinstance(v, (float, int, np.floating, np.integer)) else str(v))
        out.append("| " + " | ".join(celdas) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    print(sys.version)
    print("T4_95 =", T4_95)
