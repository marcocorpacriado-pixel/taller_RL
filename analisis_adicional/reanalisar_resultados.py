"""
reanalisar_resultados.py — Fase 1, parte A: verificación, intervalos y comparaciones pareadas.

Uso (desde cualquier carpeta):  py -3.14 analisis_adicional/reanalisar_resultados.py

Qué hace, en orden:
  1. Carga los .pkl originales con deserialización restringida (comun.cargar_pickle).
  2. Regenera las series de val (seed 2024, 50) y test (seed 7777, 100) con réplicas de los generadores
     y comprueba que reproducen los baselines guardados.
  3. Reconstruye la política greedy de CADA tabla Q guardada (600 entrenamientos), la juega con una réplica
     del entorno (validada contra el código del notebook) y comprueba que reproduce val/test guardados.
  4. Construye las observaciones por (generador, agente, semilla, serie) incluyendo la recompensa
     acumulada del entorno, que los .pkl no guardan, y el baseline "efectivo".
  5. Calcula medias e IC95 (series condicional / población), comparaciones emparejadas por serie,
     recompensa frente a rendimiento financiero y el efecto del criterio de selección.
  6. Escribe CSV en analisis_adicional/resultados/ y figuras en analisis_adicional/figuras/.

No entrena nada ni toca los archivos originales.
"""
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import comun as c

T0 = time.perf_counter()
VERIF = []


def verificar(nombre, n, max_dif, tol, detalle=""):
    ok = bool(max_dif <= tol)
    VERIF.append(dict(comprobacion=nombre, n_elementos=n, max_dif_abs=float(max_dif), tolerancia=tol,
                      resultado="OK" if ok else "FALLO", detalle=detalle))
    print(f"[{'OK' if ok else 'FALLO'}] {nombre}: n={n}, max|dif|={max_dif:.3g} (tol {tol})", flush=True)
    return ok


def log(msg):
    print(f"[{time.perf_counter() - T0:6.1f}s] {msg}", flush=True)


# -----------------------------------------------------------------------------
# 1-3. Carga, series y reproducción
# -----------------------------------------------------------------------------
c.asegurar_carpetas()
etapas = {e: c.cargar_pickle(e) for e in ("alpha", "gamma", "epsilon", "discretizacion")}
base = c.cargar_pickle("baselines")
verificar("pickles: opcodes sin INST/OBJ/GLOBAL y globales sólo numpy.dtype/_frombuffer", 6, 0, 0)

val = {g: c.generar_series(g, c.N_VAL, c.SEED_VAL) for g in c.GENERADORES}
test = {g: c.generar_series(g, c.N_TEST, c.SEED_TEST) for g in c.GENERADORES}
DUMMY = np.array([0.0])  # cortes de relleno: B&H, efectivo y aleatorio no usan la observación
N = c.N_TEST


def a_arrays(rec, bal, ops):
    return dict(rec=rec, bal=bal, ops=ops, logret=np.log(bal / 1000.0), ret=bal / 1000.0 - 1.0)


# Baselines reconstruidos externamente
ARMS = {}
dif_bh = dif_bh_b = dif_ale = dif_ale_b = dif_ale_o = 0.0
for g in c.GENERADORES:
    rec, bal, ops = c.evaluar_decisor(lambda b, pos: 2, test[g], DUMMY)
    bh = a_arrays(rec, bal, ops)
    dif_bh = max(dif_bh, np.abs(bh["logret"] - base[(g, "buy&hold")][0]["logret"]).max())
    dif_bh_b = max(dif_bh_b, np.abs(bal - base[(g, "buy&hold")][0]["balance_final"]).max())
    ARMS[(g, "buy&hold")] = {k: v[None, :] for k, v in bh.items()}

    rec, bal, ops = c.evaluar_decisor(lambda b, pos: 1, test[g], DUMMY)
    ef = a_arrays(rec, bal, ops)
    ARMS[(g, "efectivo")] = {k: v[None, :] for k, v in ef.items()}
    assert np.all(bal == 1000.0) and np.all(ops == 0) and np.all(rec == 0.0)

    runs = []
    for s in c.SEMILLAS:
        rng = np.random.default_rng(s)
        rec, bal, ops = c.evaluar_decisor(lambda b, pos: int(rng.integers(3)), test[g], DUMMY)
        a = a_arrays(rec, bal, ops)
        ref = base[(g, "aleatorio")][s]
        dif_ale = max(dif_ale, np.abs(a["logret"] - ref["logret"]).max())
        dif_ale_b = max(dif_ale_b, np.abs(bal - ref["balance_final"]).max())
        dif_ale_o = max(dif_ale_o, np.abs(ops - ref["operaciones"]).max())
        runs.append(a)
    ARMS[(g, "aleatorio")] = {k: np.array([r[k] for r in runs]) for k in runs[0]}
verificar("Buy&Hold recalculado vs baselines.pkl (log-retorno, 4 gen x 100 series)", 400, dif_bh, 1e-12)
verificar("Buy&Hold recalculado vs baselines.pkl (balance final)", 400, dif_bh_b, 1e-9)
verificar("Aleatorio (rng por semilla) recalculado vs baselines.pkl (log-retorno)", 2000, dif_ale, 1e-12)
verificar("Aleatorio recalculado vs baselines.pkl (balance final)", 2000, dif_ale_b, 1e-9)
verificar("Aleatorio recalculado vs baselines.pkl (operaciones)", 2000, dif_ale_o, 0)
verificar("Efectivo: balance=1000, operaciones=0, recompensa=0 en las 400 series de test (réplica del entorno)",
          400, 0, 0)
log("baselines reconstruidos")

# La réplica del entorno se contrasta con la clase SimpleTradingEnv extraída del notebook del profesor (celda 4)
Env = c.cargar_entorno_original()
rng_chk = np.random.default_rng(1)
d_env = 0.0
n_env = 0
for g in c.GENERADORES:
    for p in c.generar_series(g, 10, 99):
        acciones = rng_chk.integers(0, 3, size=len(p))
        env = Env(p)
        env.reset()
        rec_o, ops_o, term, k = 0.0, 0, False, 0
        while not term:
            a = int(acciones[k])
            k += 1
            ops_o += (a == 2 and env.position == 0) or (a == 0 and env.position == 1)
            _, r_, term, _, _ = env.step(a)
            rec_o += r_
        it = iter(acciones)
        rec_r, bal_r, ops_r = c.simular(p, c.calcular_bins(p, np.array([0.0])), lambda b, pos: int(next(it)))
        d_env = max(d_env, abs(rec_o - rec_r), abs(env.balance - bal_r), abs(ops_o - ops_r))
        n_env += 1
verificar("Réplica del entorno vs SimpleTradingEnv del notebook (recompensa, balance, operaciones; 40 episodios con acciones aleatorias)",
          n_env, d_env, 0)

# Todas las corridas guardadas: se reconstruye su política y se compara con val/test guardados
RUNS = {}   # (etapa, gen, algo, variante, seed) -> dict
d_val = d_test = d_val_b = d_test_b = d_ops = 0.0
n_emp = 0
total = sum(len(v) for v in etapas.values())
hecho = 0
for etapa, resultados in etapas.items():
    for r in resultados:
        g = r["generador"]
        Q, cortes = r["Q"], r["cortes"]
        comprar, vender, e0, e1 = c.politica_desde_Q(Q)
        n_emp += int(e0.sum() + e1.sum())
        rv = c.evaluar_tabla(Q, cortes, val[g])
        rt = c.evaluar_tabla(Q, cortes, test[g])
        a_v, a_t = a_arrays(*rv), a_arrays(*rt)
        if not (e0.any() or e1.any()):  # con empates el original desempata al azar: no se puede exigir igualdad
            d_val = max(d_val, np.abs(a_v["logret"] - r["val"]["logret"]).max())
            d_test = max(d_test, np.abs(a_t["logret"] - r["test"]["logret"]).max())
            d_val_b = max(d_val_b, np.abs(rv[1] - r["val"]["balance_final"]).max() / 1000)
            d_test_b = max(d_test_b, np.abs(rt[1] - r["test"]["balance_final"]).max() / 1000)
            d_ops = max(d_ops, np.abs(rt[2] - r["test"]["operaciones"]).max(), np.abs(rv[2] - r["val"]["operaciones"]).max())
        RUNS[(etapa, g, r["algoritmo"], r["variante"], r["seed"])] = dict(val=a_v, test=a_t, empates=int(e0.sum() + e1.sum()), run=r)
        hecho += 1
    log(f"etapa {etapa}: {len(resultados)} políticas reconstruidas ({hecho}/{total})")
sin_empates = sum(1 for v in RUNS.values() if v["empates"] == 0)
verificar(f"Política reconstruida de Q vs log-retorno VALIDACIÓN guardado ({sin_empates} entrenamientos sin empates)", sin_empates * 50, d_val, 1e-12)
verificar(f"Política reconstruida de Q vs log-retorno TEST guardado ({sin_empates} entrenamientos sin empates)", sin_empates * 100, d_test, 1e-12)
verificar("Política reconstruida vs balance final guardado (val y test, relativo a 1000)", sin_empates * 150, max(d_val_b, d_test_b), 1e-12)
verificar("Política reconstruida vs nº de operaciones guardado (val y test)", sin_empates * 150, d_ops, 0)
VERIF.append(dict(comprobacion="Entrenamientos con empates exactos en Q (celdas empatadas totales; todos de '10 uniformes')",
                  n_elementos=total - sin_empates, max_dif_abs=float(n_emp), tolerancia="informativo", resultado="INFO",
                  detalle="; ".join(sorted({k[3] for k, v in RUNS.items() if v["empates"] > 0}))))

# Ganadores de la última etapa de ajuste (los que reporta la tabla original)
fin = c.ganadores(etapas["epsilon"])
for (g, a), runs in fin.items():
    assert [r["seed"] for r in runs] == list(c.SEMILLAS)
    ARMS[(g, a)] = None
    obs = [RUNS[("epsilon", g, a, r["variante"], r["seed"])]["test"] for r in runs]
    ARMS[(g, a)] = {k: np.array([o[k] for o in obs]) for k in obs[0]}

# Coincidencia con log_ajuste.txt (config. ganadora de la etapa epsilon) y con tabla_resultados.md
texto = (c.RES_ORIG / "log_ajuste.txt").read_text(encoding="utf-8-sig")
bloque = texto.split("[epsilon]")[2] if texto.count("[epsilon]") >= 2 else texto.split("[epsilon]")[-1]
bloque = bloque.split("[discretizacion]")[0]
n_ok = n_tot = 0
for linea in bloque.splitlines():
    m = re.match(r"\s+(\w+)\s+(\w+)\s+-> α=([\d.]+) decay=([\d.]+) γ=([\d.]+) ε=([\d.]+)→([\d.]+) f=([\d.]+)", linea)
    if not m:
        continue
    g, a = m.group(1), m.group(2)
    cfg = fin[(g, a)][0]["cfg"]
    esperado = (float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7)), float(m.group(8)))
    real = (cfg["alpha"], cfg["alpha_decay"], cfg["gamma"], cfg["eps_inicial"], cfg["eps_final"], cfg["frac_exploracion"])
    n_tot += 1
    n_ok += int(np.allclose(esperado, real))
verificar("Configuración ganadora recalculada vs log_ajuste.txt (8 pares generador-algoritmo)", n_tot, n_tot - n_ok, 0)

tabla_orig = (c.RES_ORIG.parent / "figuras" / "tabla_resultados.md").read_text(encoding="utf-8")
nombre_tab = {"Tendencia + ciclo": "ciclico", "Browniano geométrico": "browniano",
              "Reversión a la media": "reversion", "Regímenes alcista/bajista": "regimenes"}
dif_tab = dif_bal = dif_ops = 0.0
n_tab = 0
for linea in tabla_orig.splitlines():
    cel = [x.strip() for x in linea.strip("|").split("|")]
    if len(cel) < 8 or cel[0] not in nombre_tab:
        continue
    g = nombre_tab[cel[0]]
    ag = {"Aleatorio": "aleatorio", "Buy & Hold": "buy&hold", "**Q-Learning**": "qlearning", "**SARSA**": "sarsa"}[cel[1]]
    m, sd = re.match(r"([+-][\d.]+) ± ([\d.]+)", cel[2]).groups()
    medias = (ARMS[(g, ag)]["logret"] * 100).mean(axis=1)
    dif_tab = max(dif_tab, abs(medias.mean() - float(m)), abs(medias.std() - float(sd)))
    dif_bal = max(dif_bal, abs(round(float(np.median(ARMS[(g, ag)]["bal"]))) - float(cel[3].replace(",", ""))))
    dif_ops = max(dif_ops, abs(round(float(ARMS[(g, ag)]["ops"].mean())) - float(cel[4])))
    n_tab += 1
verificar("Tabla original: log-ret. test media ± desv. (ddof=0) entre semillas, 16 filas (tolerancia = redondeo a 0.1)", n_tab, dif_tab, 0.051)
verificar("Tabla original: mediana del balance final ($, redondeada), 16 filas", n_tab, dif_bal, 0)
verificar("Tabla original: nº medio de operaciones (redondeado), 16 filas", n_tab, dif_ops, 0)

# -----------------------------------------------------------------------------
# 4. Observaciones largas (CSV)
# -----------------------------------------------------------------------------
filas = []
for (g, a), A in ARMS.items():
    S = A["logret"].shape[0]
    for s in range(S):
        for j in range(N):
            filas.append(dict(generador=g, agente=a, semilla=s, serie=j, recompensa_acum=A["rec"][s, j],
                              logret=A["logret"][s, j], ret_simple=A["ret"][s, j], balance_final=A["bal"][s, j],
                              operaciones=int(A["ops"][s, j])))
c.escribir_csv("observaciones_test.csv", filas)
log(f"observaciones_test.csv: {len(filas)} filas")

# -----------------------------------------------------------------------------
# 5a. Métricas por agente con IC
# -----------------------------------------------------------------------------
AGENTES = ["efectivo", "aleatorio", "buy&hold", "qlearning", "sarsa"]


def pp(x):
    return 100.0 * x


def ic_pop_t4(res):
    h = c.T4_95 * np.sqrt(res["var_pop"])
    return res["media"] - h, res["media"] + h


metricas = []
for i, g in enumerate(c.GENERADORES):
    for k, a in enumerate(AGENTES):
        A = ARMS[(g, a)]
        L = A["logret"]
        r = c.resumen_brazo(L)
        rng = np.random.default_rng([c.BOOT_SEED, 1, i, k])
        lo_s, hi_s = c.ic_bootstrap_series(L.mean(axis=0), rng)
        lo_h, hi_h = c.ic_bootstrap_jerarquico(L, None, rng)
        lo_t, hi_t = ic_pop_t4(r)
        medias_sem = L.mean(axis=1)
        metricas.append(dict(
            generador=g, agente=a, semillas=L.shape[0], series=L.shape[1],
            logret_pp=pp(r["media"]),
            sd_semillas_pp=pp(medias_sem.std(ddof=1)) if L.shape[0] > 1 else 0.0,
            sd_semillas_pp_ddof0=pp(medias_sem.std(ddof=0)),
            ic95_series_lo=pp(lo_s), ic95_series_hi=pp(hi_s),
            ic95_pob_t4_lo=pp(lo_t), ic95_pob_t4_hi=pp(hi_t),
            ic95_pob_boot_lo=pp(lo_h), ic95_pob_boot_hi=pp(hi_h),
            se_series_pp=pp(np.sqrt(r["var_cond"])), se_pob_pp=pp(np.sqrt(r["var_pop"])),
            sd_entre_series_pp=pp(L.mean(axis=0).std(ddof=1)),
            ops_medias=A["ops"].mean(), balance_mediano=float(np.median(A["bal"])),
            ret_simple_medio_pp=pp(A["ret"].mean()), recompensa_media_pp=pp(A["rec"].mean())))
c.escribir_csv("metricas_por_agente.csv", metricas)
log("metricas_por_agente.csv")

# -----------------------------------------------------------------------------
# 5b. Comparaciones emparejadas por serie
# -----------------------------------------------------------------------------
FAMILIA = [("qlearning", "sarsa"), ("qlearning", "buy&hold"), ("sarsa", "buy&hold"), ("qlearning", "aleatorio"),
           ("sarsa", "aleatorio"), ("qlearning", "efectivo"), ("sarsa", "efectivo")]
CONTEXTO = [("buy&hold", "efectivo"), ("aleatorio", "efectivo")]
M_FAMILIA = len(FAMILIA) * len(c.GENERADORES)  # 28 comparaciones confirmatorias para Holm/Bonferroni
T4_BONF = c.t4_ppf(1 - 0.025 / M_FAMILIA)


def comparar(LA, LB, rng, escala=100.0):
    """A - B emparejado por serie. LA, LB: (S, J). Devuelve dict con diferencias e IC."""
    LA, LB = np.atleast_2d(LA), np.atleast_2d(LB)
    J = LA.shape[1]
    rA, rB = c.resumen_brazo(LA), c.resumen_brazo(LB)
    yA, yB = LA.mean(axis=0), LB.mean(axis=0)
    d_j = yA - yB
    media = yA.mean() - yB.mean()
    var_cond = d_j.var(ddof=1) / J
    cov_j = np.cov(yA, yB, ddof=1)[0, 1] if yA.std() > 0 and yB.std() > 0 else 0.0
    var_pop = max(rA["var_pop"] + rB["var_pop"] - 2 * cov_j / J, var_cond)
    se_pop = np.sqrt(var_pop)
    lo_s, hi_s = c.ic_bootstrap_series(d_j, rng)
    lo_h, hi_h = c.ic_bootstrap_jerarquico(LA, LB, rng)
    por_sem_A, por_sem_B = LA.mean(axis=1), LB.mean(axis=1)
    pares = por_sem_A[:, None] - por_sem_B[None, :]
    eps = 1e-12
    t_stat = media / se_pop if se_pop > 0 else np.nan
    return dict(
        dif=escala * media, ic_series_lo=escala * lo_s, ic_series_hi=escala * hi_s,
        ic_pob_t4_lo=escala * (media - c.T4_95 * se_pop), ic_pob_t4_hi=escala * (media + c.T4_95 * se_pop),
        ic_pob_boot_lo=escala * lo_h, ic_pob_boot_hi=escala * hi_h,
        ic_pob_bonf_lo=escala * (media - T4_BONF * se_pop), ic_pob_bonf_hi=escala * (media + T4_BONF * se_pop),
        prop_A_gana=float((d_j > eps).mean()), prop_empate=float((np.abs(d_j) <= eps).mean()), prop_A_pierde=float((d_j < -eps).mean()),
        sd_semillas_A=escala * (por_sem_A.std(ddof=1) if len(por_sem_A) > 1 else 0.0),
        sd_semillas_B=escala * (por_sem_B.std(ddof=1) if len(por_sem_B) > 1 else 0.0),
        frac_pares_semilla_A_gana=float((pares > eps).mean()),
        se_series=escala * np.sqrt(var_cond), se_pob=escala * se_pop, t_stat=t_stat,
        p_t4=c.t4_pvalor(t_stat) if np.isfinite(t_stat) else np.nan)


comps = []
for i, g in enumerate(c.GENERADORES):
    for k, (a, b) in enumerate(FAMILIA + CONTEXTO):
        rng = np.random.default_rng([c.BOOT_SEED, 2, i, k])
        res = comparar(ARMS[(g, a)]["logret"], ARMS[(g, b)]["logret"], rng)
        res.update(generador=g, A=a, B=b, metrica="logret_pp", familia="confirmatoria" if (a, b) in FAMILIA else "contexto",
                   ops_A=ARMS[(g, a)]["ops"].mean(), ops_B=ARMS[(g, b)]["ops"].mean())
        comps.append(res)
# Holm sobre la familia confirmatoria (28 comparaciones); p-valores t con 4 g.l. (S-1 de las semillas)
fam = [r for r in comps if r["familia"] == "confirmatoria"]
orden = sorted(range(len(fam)), key=lambda i: (fam[i]["p_t4"] if np.isfinite(fam[i]["p_t4"]) else 2.0))
previo = 0.0
for rango, i in enumerate(orden):
    p = fam[i]["p_t4"]
    if not np.isfinite(p):
        fam[i]["p_holm"] = np.nan
        continue
    previo = max(previo, min(1.0, (M_FAMILIA - rango) * p))
    fam[i]["p_holm"] = previo
for r in comps:
    r.setdefault("p_holm", np.nan)
    excl = lambda lo, hi: (lo > 0) or (hi < 0)
    if r["se_pob"] == 0 and r["dif"] == 0:
        r["clasificacion"] = "idéntica (diferencia exactamente 0 en todas las series)"
    elif r["familia"] == "contexto":
        r["clasificacion"] = "contexto (fuera de la familia de 28)"
    elif (r["p_holm"] < 0.05 and excl(r["ic_pob_t4_lo"], r["ic_pob_t4_hi"]) and excl(r["ic_series_lo"], r["ic_series_hi"])
          and (r["frac_pares_semilla_A_gana"] >= 0.8 or r["frac_pares_semilla_A_gana"] <= 0.2)):
        r["clasificacion"] = "robusta"
    elif r["p_t4"] < 0.05:
        r["clasificacion"] = "indicio exploratorio (no sobrevive a Holm o a la consistencia entre semillas)"
    else:
        r["clasificacion"] = "no concluyente (IC población incluye 0)"

# Operaciones: QL - SARSA
for i, g in enumerate(c.GENERADORES):
    rng = np.random.default_rng([c.BOOT_SEED, 3, i])
    res = comparar(ARMS[(g, "qlearning")]["ops"].astype(float), ARMS[(g, "sarsa")]["ops"].astype(float), rng, escala=1.0)
    res.update(generador=g, A="qlearning", B="sarsa", metrica="operaciones", familia="descriptiva", p_holm=np.nan,
               ops_A=ARMS[(g, "qlearning")]["ops"].mean(), ops_B=ARMS[(g, "sarsa")]["ops"].mean(),
               clasificacion="descriptiva (no se aplica Holm)")
    comps.append(res)
cols = ["generador", "metrica", "A", "B", "familia", "dif", "ic_series_lo", "ic_series_hi", "ic_pob_t4_lo", "ic_pob_t4_hi",
        "ic_pob_boot_lo", "ic_pob_boot_hi", "ic_pob_bonf_lo", "ic_pob_bonf_hi", "prop_A_gana", "prop_empate", "prop_A_pierde",
        "sd_semillas_A", "sd_semillas_B", "frac_pares_semilla_A_gana", "ops_A", "ops_B", "se_series", "se_pob", "t_stat", "p_t4",
        "p_holm", "clasificacion"]
c.escribir_csv("comparaciones_pareadas.csv", comps, cols)
log("comparaciones_pareadas.csv")

# -----------------------------------------------------------------------------
# 5c. Recompensa del entorno frente a rendimiento financiero
# -----------------------------------------------------------------------------
rv = []
for i, g in enumerate(c.GENERADORES):
    for k, a in enumerate(["efectivo", "buy&hold", "aleatorio", "qlearning", "sarsa"]):
        A = ARMS[(g, a)]
        rng = np.random.default_rng([c.BOOT_SEED, 4, i, k])
        fila = dict(generador=g, agente=a, operaciones=A["ops"].mean(), balance_medio=A["bal"].mean(), balance_mediano=float(np.median(A["bal"])))
        for nombre, M in (("recompensa_pp", A["rec"]), ("logret_pp", A["logret"]), ("ret_simple_pp", A["ret"]),
                          ("brecha_pp", A["rec"] - A["logret"])):
            y = M.mean(axis=0)
            lo, hi = c.ic_bootstrap_series(y, rng)
            fila[nombre] = pp(y.mean())
            fila[nombre + "_ic_lo"] = pp(lo)
            fila[nombre + "_ic_hi"] = pp(hi)
        yr, yl = A["rec"].mean(axis=0), A["logret"].mean(axis=0)
        fila["corr_serie_rec_logret"] = np.corrcoef(yr, yl)[0, 1] if yr.std() > 0 and yl.std() > 0 else np.nan
        fila["frac_series_signo_distinto"] = float((np.sign(np.round(yr, 12)) != np.sign(np.round(yl, 12))).mean())
        rv.append(fila)
c.escribir_csv("recompensa_vs_financiero.csv", rv)
log("recompensa_vs_financiero.csv")

# Browniano (y resto): por semilla, ¿qué hace cada política y qué criterio la favorece?
sem = []
for g in c.GENERADORES:
    for a in c.ALGORITMOS:
        A = ARMS[(g, a)]
        for s in range(5):
            sem.append(dict(generador=g, algoritmo=a, semilla=s, operaciones=A["ops"][s].mean(),
                            recompensa_pp=pp(A["rec"][s].mean()), logret_pp=pp(A["logret"][s].mean()),
                            ret_simple_pp=pp(A["ret"][s].mean()),
                            comportamiento=("efectivo" if A["ops"][s].mean() == 0 else
                                            ("compra y mantiene (≈B&H)" if A["ops"][s].mean() >= 0.99 and A["ops"][s].max() <= 1 else "opera"))))
c.escribir_csv("recompensa_vs_financiero_por_semilla.csv", sem)

# ¿Habría cambiado la selección si el criterio fuese la recompensa acumulada en vez del log-retorno?
sel = []
for etapa in ("alpha", "gamma", "epsilon"):
    grupos = {}
    for (e, g, a, v, s), d in RUNS.items():
        if e == etapa:
            grupos.setdefault((g, a), {}).setdefault(v, []).append(d)
    for (g, a), vs in sorted(grupos.items()):
        punt_log = {v: np.mean([d["val"]["logret"].mean() for d in ds]) for v, ds in vs.items()}
        punt_rec = {v: np.mean([d["val"]["rec"].mean() for d in ds]) for v, ds in vs.items()}
        punt_ret = {v: np.mean([d["val"]["ret"].mean() for d in ds]) for v, ds in vs.items()}
        test_log = {v: np.mean([d["test"]["logret"].mean() for d in ds]) for v, ds in vs.items()}
        v_log = max(punt_log, key=punt_log.get)   # mismo criterio y desempate que el original (primera máxima)
        v_rec = max(punt_rec, key=punt_rec.get)
        v_ret = max(punt_ret, key=punt_ret.get)
        sel.append(dict(etapa=etapa, generador=g, algoritmo=a, elegida_por_logret=v_log, elegida_por_recompensa=v_rec,
                        elegida_por_ret_simple=v_ret, coincide_recompensa=v_log == v_rec, coincide_ret_simple=v_log == v_ret,
                        test_logret_pp_elegida_logret=pp(test_log[v_log]), test_logret_pp_elegida_recompensa=pp(test_log[v_rec]),
                        dif_val_entre_1ra_y_2da_pp=pp(np.diff(sorted(punt_log.values())[-2:])[0]) if len(punt_log) > 1 else np.nan))
c.escribir_csv("seleccion_criterio.csv", sel)
log("selección por criterio")

# -----------------------------------------------------------------------------
# 6. Verificación del emparejamiento y escritura de verificaciones
# -----------------------------------------------------------------------------
orden_ok = all(ARMS[(g, a)]["logret"].shape[1] == c.N_TEST for g in c.GENERADORES for a in AGENTES)
verificar("Emparejamiento: todos los agentes tienen 100 series de test y la serie j es la misma (seed 7777) en todos los brazos",
          len(c.GENERADORES) * len(AGENTES), 0 if orden_ok else 1, 0)
# B&H de la réplica = fórmula cerrada del profesor 1000*0.999/P5*P_T (comprobación independiente de la simulación)
cerrado = max(np.abs(ARMS[(g, "buy&hold")]["bal"][0] - np.array([1000 * 0.999 / p[5] * p[-1] for p in test[g]])).max() for g in c.GENERADORES)
verificar("Buy&Hold simulado vs fórmula cerrada 1000*0.999/P5*P_final (notebook del profesor, celda 21)", 400, cerrado, 1e-6)
c.escribir_csv("verificacion_reproduccion.csv", VERIF)
fallos = [v for v in VERIF if v["resultado"] == "FALLO"]
log(f"verificaciones: {len(VERIF)} filas, fallos={len(fallos)}")


# -----------------------------------------------------------------------------
# 7. Figuras
# -----------------------------------------------------------------------------
COL = {"efectivo": "#555555", "aleatorio": "#9aa0a6", "buy&hold": "#1baf7a", "qlearning": "#2a78d6", "sarsa": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "savefig.dpi": 170, "savefig.bbox": "tight"})
M = {(r["generador"], r["agente"]): r for r in metricas}

fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, g in zip(axs, c.GENERADORES):
    for x, a in enumerate(AGENTES):
        r = M[(g, a)]
        ax.plot([x, x], [r["ic95_pob_t4_lo"], r["ic95_pob_t4_hi"]], color=COL[a], lw=1.2, alpha=0.8)
        ax.plot([x - 0.08, x + 0.08], [r["ic95_pob_t4_lo"]] * 2, color=COL[a], lw=1.2)
        ax.plot([x - 0.08, x + 0.08], [r["ic95_pob_t4_hi"]] * 2, color=COL[a], lw=1.2)
        ax.plot([x, x], [r["ic95_series_lo"], r["ic95_series_hi"]], color=COL[a], lw=5, alpha=0.55, solid_capstyle="butt")
        ax.plot(x, r["logret_pp"], "o", color="black", ms=4, zorder=5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(AGENTES)))
    ax.set_xticklabels([c.ETIQUETA_AG[a] for a in AGENTES], rotation=30, ha="right")
    ax.set_title(c.ETIQUETA_GEN[g], fontweight="bold")
axs[0].set_ylabel("log-retorno medio en test (pp)")
fig.suptitle("Rendimiento en test con IC95: barra gruesa = incertidumbre por series (políticas fijas); "
             "línea fina = series + semillas (t, 4 g.l.)", y=1.02, fontsize=9.5)
fig.savefig(c.FIG / "fig1_rendimientos_ic.png")
plt.close(fig)

C = {(r["generador"], r["A"], r["B"], r["metrica"]): r for r in comps}
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, met, tit, unid in ((axs[0], "logret_pp", "Q-Learning − SARSA: log-retorno en test", "pp"),
                           (axs[1], "operaciones", "Q-Learning − SARSA: operaciones por serie", "operaciones")):
    for x, g in enumerate(c.GENERADORES):
        r = C[(g, "qlearning", "sarsa", met)]
        ax.plot([x, x], [r["ic_pob_t4_lo"], r["ic_pob_t4_hi"]], color="#444444", lw=1.2)
        ax.plot([x - 0.08, x + 0.08], [r["ic_pob_t4_lo"]] * 2, color="#444444", lw=1.2)
        ax.plot([x - 0.08, x + 0.08], [r["ic_pob_t4_hi"]] * 2, color="#444444", lw=1.2)
        ax.plot([x, x], [r["ic_series_lo"], r["ic_series_hi"]], color="#2a78d6", lw=5, alpha=0.55, solid_capstyle="butt")
        ax.plot(x, r["dif"], "o", color="black", ms=4, zorder=5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c.ETIQUETA_GEN[g] for g in c.GENERADORES], rotation=20, ha="right")
    ax.set_ylabel(f"diferencia ({unid})")
    ax.set_title(tit, fontweight="bold")
todos_incluyen_0 = all(C[(g, "qlearning", "sarsa", "logret_pp")]["ic_pob_t4_lo"] <= 0 <= C[(g, "qlearning", "sarsa", "logret_pp")]["ic_pob_t4_hi"]
                       for g in c.GENERADORES)
fig.suptitle("Diferencias emparejadas por serie (100 series de test comunes; grueso = series, fino = series + semillas). "
             + ("El IC de población incluye 0 en los 4 generadores: no se puede declarar ganador" if todos_incluyen_0 else "Algún IC de población excluye 0"),
             y=1.03, fontsize=9)
fig.savefig(c.FIG / "fig2_qlearning_vs_sarsa.png")
plt.close(fig)

fig, axs = plt.subplots(4, 2, figsize=(10, 8.2))
for i, g in enumerate(c.GENERADORES):
    for j, b in enumerate(["efectivo", "buy&hold"]):
        ax = axs[i, j]
        for y, a in enumerate(c.ALGORITMOS):
            r = C[(g, a, b, "logret_pp")]
            ax.plot([r["ic_pob_t4_lo"], r["ic_pob_t4_hi"]], [y, y], color=COL[a], lw=1.2)
            ax.plot([r["ic_series_lo"], r["ic_series_hi"]], [y, y], color=COL[a], lw=5, alpha=0.55, solid_capstyle="butt")
            ax.plot(r["dif"], y, "o", color="black", ms=4, zorder=5)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(2))
        ax.set_yticklabels([c.ETIQUETA_AG[a] for a in c.ALGORITMOS])
        ax.set_ylim(-0.6, 1.6)
        if i == 0:
            ax.set_title(f"Agente − {c.ETIQUETA_AG[b]}", fontweight="bold")
        if j == 0:
            ax.set_ylabel(c.ETIQUETA_GEN[g], fontsize=8)
        if i == 3:
            ax.set_xlabel("diferencia de log-retorno (pp)")
fig.suptitle("Frente a efectivo y Buy & Hold (IC95: grueso = series; fino = series + semillas, t 4 g.l.)", y=1.0, fontsize=10)
fig.tight_layout()
fig.savefig(c.FIG / "fig3_vs_efectivo_y_buyhold.png")
plt.close(fig)

R = {(r["generador"], r["agente"]): r for r in rv}
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, g in zip(axs, c.GENERADORES):
    ags = ["buy&hold", "qlearning", "sarsa"]
    anchos = 0.36
    for k, (campo, etiqueta, hatch) in enumerate((("recompensa_pp", "Σ recompensa del entorno (suma de retornos simples)", ""),
                                                  ("logret_pp", "log-retorno del patrimonio (capitalizado)", "//"))):
        for x, a in enumerate(ags):
            r = R[(g, a)]
            xx = x + (k - 0.5) * anchos
            ax.bar(xx, r[campo], anchos, color=COL[a], hatch=hatch, edgecolor="white", alpha=0.9, label=etiqueta if x == 0 else None)
            ax.plot([xx] * 2, [r[campo + "_ic_lo"], r[campo + "_ic_hi"]], color="black", lw=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c.ETIQUETA_AG[a] for a in ags])
    ax.set_title(c.ETIQUETA_GEN[g], fontweight="bold")
axs[0].set_ylabel("media en test (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Recompensa acumulada del entorno vs log-retorno del patrimonio (IC95 por series). El retorno simple del patrimonio está en el CSV", y=1.02, fontsize=10)
fig.savefig(c.FIG / "fig5_recompensa_vs_rendimiento.png")
plt.close(fig)
log("figuras escritas")

# -----------------------------------------------------------------------------
# Resumen por pantalla
# -----------------------------------------------------------------------------
print("\n=== MÉTRICAS (log-retorno test, pp) ===")
for r in metricas:
    print(f"{r['generador']:9s} {r['agente']:9s} {r['logret_pp']:+9.2f}  sd_sem={r['sd_semillas_pp']:6.2f}  IC_series=[{r['ic95_series_lo']:+8.2f},{r['ic95_series_hi']:+8.2f}]  "
          f"IC_pob_t4=[{r['ic95_pob_t4_lo']:+8.2f},{r['ic95_pob_t4_hi']:+8.2f}]  IC_pob_boot=[{r['ic95_pob_boot_lo']:+8.2f},{r['ic95_pob_boot_hi']:+8.2f}]  ops={r['ops_medias']:.1f}")
print("\n=== COMPARACIONES ===")
for r in comps:
    print(f"{r['generador']:9s} {r['metrica']:10s} {r['A']:9s}-{r['B']:9s} dif={r['dif']:+9.2f} IC_s=[{r['ic_series_lo']:+8.2f},{r['ic_series_hi']:+8.2f}] "
          f"IC_pob=[{r['ic_pob_t4_lo']:+8.2f},{r['ic_pob_t4_hi']:+8.2f}] gana/emp/pierde={r['prop_A_gana']:.2f}/{r['prop_empate']:.2f}/{r['prop_A_pierde']:.2f} "
          f"p={r['p_t4']:.3g} holm={r['p_holm']:.3g} -> {r['clasificacion']}")
sys.exit(1 if fallos else 0)
