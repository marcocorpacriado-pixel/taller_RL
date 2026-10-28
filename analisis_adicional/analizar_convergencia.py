"""
analizar_convergencia.py — Fase 1, parte B: convergencia y estabilidad del rendimiento.

Uso:  py -3.14 analisis_adicional/analizar_convergencia.py

Reanaliza los historiales guardados (hist['eval_logret'], 51 puntos: episodio 1 y cada 100, política greedy
evaluada sobre las 10 primeras series de validación). NO reentrena.

Definiciones (ver INFORME_FASE1.md, §3):
  * Ancla inferior  = log-retorno del agente aleatorio (rng semilla 0) en esas mismas 10 series de validación.
  * Meseta          = media de los últimos 11 puntos de la curva de la propia corrida (episodios 4000-5000).
  * Umbral(q)       = ancla + q * (meseta - ancla),  q en {0.80, 0.90, 0.95}.
  * Curva suavizada = media móvil de 3 puntos (como el criterio original).
  * Estabilización permanente(q): primer punto desde el cual la curva suavizada permanece >= umbral(q)
    hasta el final. Cruce inicial(q): primer punto que alcanza el umbral (criterio equivalente al original).
  * Si una curva nunca queda por encima del umbral de forma permanente: "no estabiliza" (censurada a 5100).

Alcanzar un umbral de rendimiento NO demuestra convergencia matemática de Q: sólo describe cuándo la política
greedy deja de mejorar de forma apreciable en esta métrica ruidosa (10 series).
"""
import itertools
import json
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import comun as c

T0 = time.perf_counter()
CENSURA = 5100
UMBRALES = (0.80, 0.90, 0.95)
VERIF = []


def log(m):
    print(f"[{time.perf_counter() - T0:6.1f}s] {m}", flush=True)


def verificar(nombre, n, max_dif, tol, detalle=""):
    ok = bool(max_dif <= tol)
    VERIF.append(dict(comprobacion=nombre, n_elementos=n, max_dif_abs=float(max_dif), tolerancia=tol,
                      resultado="OK" if ok else "FALLO", detalle=detalle))
    print(f"[{'OK' if ok else 'FALLO'}] {nombre}: n={n}, max|dif|={max_dif:.3g} (tol {tol})", flush=True)
    return ok


c.asegurar_carpetas()
etapas = {e: c.cargar_pickle(e) for e in ("alpha", "gamma", "epsilon", "discretizacion")}


def descr_cfg(cfg):
    a = f"{cfg['alpha']}" if cfg["alpha_decay"] == 1 else f"{cfg['alpha']}→{cfg['alpha_min']}"
    e = f"{cfg['eps_inicial']}" if cfg["eps_inicial"] == cfg["eps_final"] else f"1→0.01(f={cfg['frac_exploracion']})"
    return f"α={a} γ={cfg['gamma']} ε={e} bins={cfg['n_bins']}/{cfg['discretizacion']}"


def clave_cfg(cfg):
    d = {k: v for k, v in cfg.items() if k != "algoritmo"}
    return json.dumps(d, sort_keys=True)


def epsilon(cfg, episodio_idx):
    d = (cfg["eps_final"] / cfg["eps_inicial"]) ** (1 / (cfg["frac_exploracion"] * cfg["n_episodios"]))
    return max(cfg["eps_final"], cfg["eps_inicial"] * d**episodio_idx)


# -----------------------------------------------------------------------------
# Corridas únicas (una misma config+semilla aparece en varias etapas) y comprobación de determinismo
# -----------------------------------------------------------------------------
unicas = {}
dup_dif = 0.0
n_dup = 0
for etapa, resultados in etapas.items():
    for r in resultados:
        k = (r["generador"], r["algoritmo"], clave_cfg(r["cfg"]), r["seed"])
        if k in unicas:
            o = unicas[k]
            dup_dif = max(dup_dif, float(np.abs(o["hist"]["eval_logret"] - r["hist"]["eval_logret"]).max()),
                          float(np.abs(o["Q"] - r["Q"]).max()))
            n_dup += 1
        else:
            unicas[k] = dict(r, etapa_origen=etapa)
log(f"{len(unicas)} corridas únicas; {n_dup} repeticiones exactas de (config, semilla) entre etapas")
verificar("Determinismo del entrenamiento: misma config+semilla en dos etapas distintas => misma Q y misma curva", n_dup, dup_dif, 0)

# Ancla inferior: agente aleatorio (semilla 0, como graficas.baselines_curva) sobre las 10 series de la curva
val10 = {g: c.generar_series(g, c.N_VAL, c.SEED_VAL)[:10] for g in c.GENERADORES}
ancla = {}
for g in c.GENERADORES:
    rng = np.random.default_rng(0)
    _, bal, _ = c.evaluar_decisor(lambda b, pos: int(rng.integers(3)), val10[g], np.array([0.0]))
    ancla[g] = float(np.log(bal / 1000.0).mean())

# El último punto de la curva debe coincidir con las 10 primeras series de la validación guardada
d_fin = 0.0
for r in unicas.values():
    d_fin = max(d_fin, abs(float(r["hist"]["eval_logret"][-1]) - float(r["val"]["logret"][:10].mean())))
verificar("Último punto de la curva guardada == media de las 10 primeras series de validación guardada (float32)", len(unicas), d_fin, 1e-5)


# -----------------------------------------------------------------------------
# Criterios de estabilización
# -----------------------------------------------------------------------------
def suavizar(y):
    return np.convolve(y, np.ones(3) / 3, "valid")


def criterio_original(ep, curva):
    """Réplica de graficas.episodios_convergencia (curva media entre semillas)."""
    suave = suavizar(curva)
    meseta = suave[-max(1, len(suave) // 5):].mean()
    umbral = suave[0] + 0.9 * (meseta - suave[0])
    alc = suave >= umbral if meseta >= suave[0] else suave <= umbral
    return int(ep[1:-1][np.argmax(alc)])


def estabilizacion(ep, y, bajo, q, modo):
    """Episodio de estabilización (o CENSURA / NaN si no hay mejora). Ver docstring del módulo."""
    meseta = float(y[-11:].mean())
    if meseta - bajo < 0.01:       # sin mejora apreciable (>1 pp) sobre el azar: criterio no aplicable
        return np.nan
    umbral = bajo + q * (meseta - bajo)
    s = suavizar(y)
    centros = ep[1:-1]
    ok = s >= umbral
    if modo == "inicial":
        return float(centros[np.argmax(ok)]) if ok.any() else CENSURA
    # permanente: primer índice desde el que todos los siguientes cumplen
    if not ok[-1]:
        return CENSURA
    malos = np.where(~ok)[0]
    k = 0 if len(malos) == 0 else malos[-1] + 1
    return float(centros[k])


filas = []
for (g, a, _, s), r in unicas.items():
    ep = r["hist"]["eval_episodio"].astype(float)
    y = r["hist"]["eval_logret"].astype(float)
    cfg = r["cfg"]
    fila = dict(generador=g, algoritmo=a, semilla=s, config=descr_cfg(cfg), variante=r["variante"], etapa_origen=r["etapa_origen"],
                clave=clave_cfg(cfg), ancla_azar_pp=100 * ancla[g], meseta_pp=100 * y[-11:].mean(), tiempo_s=r["tiempo"])
    for q in UMBRALES:
        for modo, pref in (("permanente", "perm"), ("inicial", "ini")):
            e = estabilizacion(ep, y, ancla[g], q, modo)
            fila[f"{pref}_{int(q * 100)}"] = e
    e90 = fila["perm_90"]
    fila["eps_en_perm_90"] = epsilon(cfg, e90 - 1) if np.isfinite(e90) and e90 < CENSURA else np.nan
    fila["fin_decaimiento_eps"] = (cfg["frac_exploracion"] * cfg["n_episodios"]) if cfg["eps_inicial"] != cfg["eps_final"] else 0
    filas.append(fila)
c.escribir_csv("convergencia_por_corrida.csv", filas, [k for k in filas[0] if k != "clave"])
log("convergencia_por_corrida.csv")

# -----------------------------------------------------------------------------
# A. Configuraciones finales optimizadas de cada algoritmo
# -----------------------------------------------------------------------------
fin = c.ganadores(etapas["epsilon"])
tabla_orig = (c.RES_ORIG.parent / "figuras" / "tabla_resultados.md").read_text(encoding="utf-8")
nombre_tab = {"Tendencia + ciclo": "ciclico", "Browniano geométrico": "browniano",
              "Reversión a la media": "reversion", "Regímenes alcista/bajista": "regimenes"}
dif_conv = 0
n_conv = 0
conv_orig = {}
for g in c.GENERADORES:
    for a in c.ALGORITMOS:
        runs = fin[(g, a)]
        ep = runs[0]["hist"]["eval_episodio"]
        conv_orig[(g, a)] = criterio_original(ep, np.mean([r["hist"]["eval_logret"] for r in runs], axis=0))
for linea in tabla_orig.splitlines():
    cel = [x.strip() for x in linea.strip("|").split("|")]
    if len(cel) >= 8 and cel[0] in nombre_tab and cel[1] in ("**Q-Learning**", "**SARSA**"):
        a = "qlearning" if "Q-" in cel[1] else "sarsa"
        dif_conv = max(dif_conv, abs(conv_orig[(nombre_tab[cel[0]], a)] - int(cel[5])))
        n_conv += 1
verificar("Criterio original de convergencia recalculado == columna 'Episodios hasta convergencia' de la tabla original", n_conv, dif_conv, 0)


def permutacion_exacta(x, y):
    """Test de permutación exacto (bilateral) para la diferencia de medias, n pequeño."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    todo = np.concatenate([x, y])
    obs = x.mean() - y.mean()
    n, tot = len(x), 0
    extremos = 0
    for idx in itertools.combinations(range(len(todo)), n):
        m = np.zeros(len(todo), bool)
        m[list(idx)] = True
        d = todo[m].mean() - todo[~m].mean()
        tot += 1
        extremos += abs(d) >= abs(obs) - 1e-9
    return obs, extremos / tot


def valores(grupo, campo):
    v = np.array([r[campo] for r in grupo], float)
    return np.where(np.isnan(v), np.nan, v)


por_corrida = {(f["generador"], f["algoritmo"], f["clave"], f["semilla"]): f for f in filas}
comp = []
for g in c.GENERADORES:
    for campo in ("perm_80", "perm_90", "perm_95", "ini_90"):
        xs = {}
        for a in c.ALGORITMOS:
            xs[a] = np.array([por_corrida[(g, a, clave_cfg(r["cfg"]), r["seed"])][campo] for r in fin[(g, a)]], float)
        if any(np.isnan(v).any() for v in xs.values()):
            comp.append(dict(bloque="A: configuraciones finales (distintas entre algoritmos)", generador=g, criterio=campo,
                             config_qlearning=descr_cfg(fin[(g, "qlearning")][0]["cfg"]), config_sarsa=descr_cfg(fin[(g, "sarsa")][0]["cfg"]),
                             mediana_qlearning=np.nan, mediana_sarsa=np.nan, dif_medias=np.nan, p_permutacion=np.nan,
                             no_estabiliza_qlearning=int((xs["qlearning"] >= CENSURA).sum()), no_estabiliza_sarsa=int((xs["sarsa"] >= CENSURA).sum()),
                             valores_qlearning="n/a (sin mejora sobre el azar)", valores_sarsa="n/a (sin mejora sobre el azar)"))
            continue
        obs, p = permutacion_exacta(xs["qlearning"], xs["sarsa"])
        comp.append(dict(bloque="A: configuraciones finales (distintas entre algoritmos)", generador=g, criterio=campo,
                         config_qlearning=descr_cfg(fin[(g, "qlearning")][0]["cfg"]), config_sarsa=descr_cfg(fin[(g, "sarsa")][0]["cfg"]),
                         mediana_qlearning=float(np.median(xs["qlearning"])), mediana_sarsa=float(np.median(xs["sarsa"])),
                         dif_medias=obs, p_permutacion=p,
                         no_estabiliza_qlearning=int((xs["qlearning"] >= CENSURA).sum()), no_estabiliza_sarsa=int((xs["sarsa"] >= CENSURA).sum()),
                         valores_qlearning=" ".join(f"{v:.0f}" for v in xs["qlearning"]), valores_sarsa=" ".join(f"{v:.0f}" for v in xs["sarsa"])))

# -----------------------------------------------------------------------------
# B. Hiperparámetros IDÉNTICOS en ambos algoritmos (misma config salvo 'algoritmo', 5 semillas cada uno)
# -----------------------------------------------------------------------------
grupos = {}
for (g, a, k, s), f in por_corrida.items():
    grupos.setdefault((g, k), {}).setdefault(a, []).append(f)
pares = []
for (g, k), d in sorted(grupos.items()):
    if len(d.get("qlearning", [])) == 5 and len(d.get("sarsa", [])) == 5:
        pares.append((g, k, d))
log(f"{len(pares)} configuraciones con hiperparámetros idénticos disponibles en QL y SARSA")
filas_id = []
for g, k, d in pares:
    cfg_txt = d["qlearning"][0]["config"]
    for campo in ("perm_90", "perm_80", "perm_95"):
        xq = np.array([f[campo] for f in d["qlearning"]], float)
        xs = np.array([f[campo] for f in d["sarsa"]], float)
        if np.isnan(xq).any() or np.isnan(xs).any():
            filas_id.append(dict(generador=g, config=cfg_txt, criterio=campo, mediana_qlearning=np.nan, mediana_sarsa=np.nan, dif_medias=np.nan,
                                 p_permutacion=np.nan, no_estab_q=np.nan, no_estab_s=np.nan, valores_qlearning="n/a", valores_sarsa="n/a",
                                 meseta_q_pp=np.nan, meseta_s_pp=np.nan, tiempo_medio_q_s=np.nan, tiempo_medio_s_s=np.nan))
            continue
        obs, p = permutacion_exacta(xq, xs)
        filas_id.append(dict(generador=g, config=cfg_txt, criterio=campo, mediana_qlearning=float(np.median(xq)), mediana_sarsa=float(np.median(xs)),
                             dif_medias=obs, p_permutacion=p, no_estab_q=int((xq >= CENSURA).sum()), no_estab_s=int((xs >= CENSURA).sum()),
                             valores_qlearning=" ".join(f"{v:.0f}" for v in xq), valores_sarsa=" ".join(f"{v:.0f}" for v in xs),
                             meseta_q_pp=float(np.mean([f["meseta_pp"] for f in d["qlearning"]])), meseta_s_pp=float(np.mean([f["meseta_pp"] for f in d["sarsa"]])),
                             tiempo_medio_q_s=float(np.mean([f["tiempo_s"] for f in d["qlearning"]])), tiempo_medio_s_s=float(np.mean([f["tiempo_s"] for f in d["sarsa"]]))))
c.escribir_csv("convergencia_hiperparametros_identicos.csv", filas_id)
c.escribir_csv("convergencia_comparacion_algoritmos.csv", comp)

# Resumen sobre pares válidos (criterio principal: permanente 90 %)
princ = [f for f in filas_id if f["criterio"] == "perm_90" and np.isfinite(f["dif_medias"])]
q_antes = sum(1 for f in princ if f["mediana_qlearning"] < f["mediana_sarsa"])
s_antes = sum(1 for f in princ if f["mediana_sarsa"] < f["mediana_qlearning"])
igual = len(princ) - q_antes - s_antes
sig = [f for f in princ if f["p_permutacion"] < 0.05]
log(f"B (perm_90): {len(princ)} pares válidos; QL antes en {q_antes}, SARSA antes en {s_antes}, empate {igual}; p<0.05 sin ajustar: {len(sig)} (de {len(princ)})")
por_gen = {}
for g in c.GENERADORES:
    fg = [f for f in princ if f["generador"] == g]
    por_gen[g] = dict(n=len(fg), ql_antes=sum(f["mediana_qlearning"] < f["mediana_sarsa"] for f in fg),
                      sarsa_antes=sum(f["mediana_sarsa"] < f["mediana_qlearning"] for f in fg),
                      empate=sum(f["mediana_sarsa"] == f["mediana_qlearning"] for f in fg),
                      p_menor_005=sum(f["p_permutacion"] < 0.05 for f in fg))
resumen_B = dict(n_pares_validos=len(princ), ql_antes=q_antes, sarsa_antes=s_antes, empate=igual, p_menor_005_sin_ajustar=len(sig),
                 minimo_p_posible_con_5_vs_5=2 / 252, por_generador=por_gen,
                 nota="los pares comparten generador y semillas y no son independientes: los recuentos son descriptivos")
with open(c.SAL / "convergencia_resumen_B.json", "w", encoding="utf-8") as f:
    json.dump(resumen_B, f, indent=1, default=int)

# Sensibilidad al criterio: ¿cambia el ordenamiento QL/SARSA al cambiar q? (finales)
sens = []
for g in c.GENERADORES:
    for campo in ("perm_80", "perm_90", "perm_95", "ini_90"):
        r = next((x for x in comp if x["generador"] == g and x["criterio"] == campo), None)
        sens.append((g, campo, r["mediana_qlearning"], r["mediana_sarsa"], r["p_permutacion"]))

# -----------------------------------------------------------------------------
# Figuras
# -----------------------------------------------------------------------------
COL = {"qlearning": "#2a78d6", "sarsa": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "savefig.dpi": 170, "savefig.bbox": "tight"})

fig, axs = plt.subplots(2, 4, figsize=(16, 7), gridspec_kw={"height_ratios": [1.3, 1]})
for j, g in enumerate(c.GENERADORES):
    ax = axs[0, j]
    for a in c.ALGORITMOS:
        runs = fin[(g, a)]
        ep = runs[0]["hist"]["eval_episodio"]
        Y = np.array([r["hist"]["eval_logret"] for r in runs]) * 100
        ax.plot(ep, Y.mean(0), color=COL[a], label=f"{c.ETIQUETA_AG[a]} (media de 5 semillas)")
        ax.fill_between(ep, Y.min(0), Y.max(0), color=COL[a], alpha=0.12, lw=0)
        e90 = np.array([por_corrida[(g, a, clave_cfg(r["cfg"]), r["seed"])]["perm_90"] for r in runs], float)
        if np.isfinite(e90).all():
            ax.axvline(np.median(e90), color=COL[a], ls=":", lw=1.4)
        cfg = runs[0]["cfg"]
        if cfg["eps_inicial"] != cfg["eps_final"]:
            ax.plot([cfg["frac_exploracion"] * cfg["n_episodios"]], [0.015], marker="^", color=COL[a], ms=6, clip_on=False, transform=ax.get_xaxis_transform())
    ax.axhline(100 * ancla[g], color="gray", ls="--", lw=1, label="agente aleatorio (mismas 10 series)")
    ax.set_title(c.ETIQUETA_GEN[g], fontweight="bold")
    ax.set_xlabel("episodio de entrenamiento")
    if j == 0:
        ax.set_ylabel("log-retorno greedy, validación (10 series), pp")
        ax.legend(fontsize=7, loc="lower right")
    ax2 = axs[1, j]
    for k, a in enumerate(c.ALGORITMOS):
        for m, q in enumerate(UMBRALES):
            v = np.array([por_corrida[(g, a, clave_cfg(r["cfg"]), r["seed"])][f"perm_{int(q * 100)}"] for r in fin[(g, a)]], float)
            x = m + (k - 0.5) * 0.3
            if np.isnan(v).any():
                ax2.text(x, 50, "n/a", color=COL[a], ha="center", fontsize=7)
                continue
            ax2.plot(np.full(len(v), x) + np.linspace(-0.04, 0.04, len(v)), v, "o", color=COL[a], ms=4, alpha=0.7)
            ax2.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color="black", lw=1.5)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["80 %", "90 %", "95 %"])
    ax2.set_xlabel("umbral q (% del recorrido azar → meseta)")
    ax2.set_ylim(-100, 5400)
    if j == 0:
        ax2.set_ylabel("episodio de estabilización\n(permanente; 5100 = no estabiliza)")
fig.suptitle("Configuraciones finales de cada algoritmo (distintas entre sí). Arriba: curva media y banda mín-máx entre semillas; punteada = mediana de estabilización al 90 %; ▲ = fin del decaimiento de ε.\n"
             "Abajo: episodio de estabilización por semilla según el umbral (raya negra = mediana)", y=1.03, fontsize=9)
fig.tight_layout()
fig.savefig(c.FIG / "fig4_convergencia_sensibilidad.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 5.2))
colg = {"ciclico": "#7b3fa0", "browniano": "#888888", "reversion": "#1baf7a", "regimenes": "#d6a31c"}
for f in princ:
    xq = [float(v) for v in f["valores_qlearning"].split()]
    xs = [float(v) for v in f["valores_sarsa"].split()]
    ax.errorbar(np.median(xq), np.median(xs), xerr=[[np.median(xq) - min(xq)], [max(xq) - np.median(xq)]],
                yerr=[[np.median(xs) - min(xs)], [max(xs) - np.median(xs)]], fmt="o", color=colg[f["generador"]], ms=5, alpha=0.75, lw=0.8)
ax.plot([0, 5200], [0, 5200], "k--", lw=0.8)
for g, col in colg.items():
    ax.plot([], [], "o", color=col, label=c.ETIQUETA_GEN[g])
ax.set_xlabel("Q-Learning: episodio de estabilización (mediana; barra = mín-máx)")
ax.set_ylabel("SARSA: episodio de estabilización")
ax.set_title(f"Hiperparámetros idénticos ({len(princ)} configuraciones)\nPor debajo de la diagonal: SARSA estabiliza antes", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.savefig(c.FIG / "fig4b_convergencia_hiperparametros_identicos.png")
plt.close(fig)
log("figuras escritas")

c.escribir_csv("verificacion_convergencia.csv", VERIF)

# -----------------------------------------------------------------------------
# Resumen por pantalla
# -----------------------------------------------------------------------------
print("\n=== Criterio original replicado (episodios) ===")
for (g, a), v in conv_orig.items():
    print(f"  {g:9s} {a:9s} {v}")
print("\n=== A. Finales: estabilización por criterio ===")
for r in comp:
    print(f"  {r['generador']:9s} {r['criterio']:8s} medianas QL={r['mediana_qlearning']} SARSA={r['mediana_sarsa']} dif={r['dif_medias']} p={r['p_permutacion']} "
          f"nunca QL/S={r['no_estabiliza_qlearning']}/{r['no_estabiliza_sarsa']} | Q:{r['valores_qlearning']} | S:{r['valores_sarsa']}")
print("\n=== B. Hiperparámetros idénticos (perm_90) ===")
for f in princ:
    print(f"  {f['generador']:9s} {f['config']:48s} QL med={f['mediana_qlearning']:6.0f} SARSA med={f['mediana_sarsa']:6.0f} p={f['p_permutacion']:.3f} | Q:{f['valores_qlearning']} | S:{f['valores_sarsa']} | meseta {f['meseta_q_pp']:.1f}/{f['meseta_s_pp']:.1f}")
print(resumen_B)
print("\nAncla azar (10 series val), pp:", {g: round(100 * v, 1) for g, v in ancla.items()})
print("tiempo medio por entrenamiento (s) QL vs SARSA:", np.mean([f["tiempo_s"] for f in filas if f["algoritmo"] == "qlearning"]), np.mean([f["tiempo_s"] for f in filas if f["algoritmo"] == "sarsa"]))
sys.exit(1 if any(v["resultado"] == "FALLO" for v in VERIF) else 0)
