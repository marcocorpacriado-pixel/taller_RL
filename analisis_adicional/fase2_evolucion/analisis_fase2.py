"""
analisis_fase2.py — Fase 2: reconstrucción experimental de la evolución (bloques 1-5 y figuras).

    python -B -X utf8 analisis_adicional/fase2_evolucion/analisis_fase2.py

Lee: resultados/*.pkl de Marco (deserialización restringida) y artefactos/*.npz de la Fase 2. No entrena nada.
Escribe sólo en analisis_adicional/fase2_evolucion/{resultados,figuras}.

Reglas de uso de los conjuntos de evaluación:
  * VALIDACIÓN (50 series, seed 2024): única base para describir/seleccionar. Aquí no se selecciona nada nuevo: todas las
    configuraciones estaban fijadas de antemano (utilidades.plan_entrenamientos o ganadores de Marco por validación).
  * TEST (100 series, seed 7777): sólo se evalúan configuraciones ya fijadas. Es el mismo test que Marco ya inspeccionó:
    esta fase es EXPLORATORIA, no una confirmación independiente.
"""
import sys

sys.dont_write_bytecode = True

import csv
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import estadistica as est
import utilidades as u

c1 = u.c1
G, A = u.GENERADORES, u.ALGORITMOS
NG, NA = c1.ETIQUETA_GEN, c1.ETIQUETA_AG
COL = {"qlearning": "#2a78d6", "sarsa": "#eb6834"}
U = u.asegurar()

etapas = u.marco_etapas()
cadena = u.cadena_marco(etapas)
GRUPOS = {g: u.cargar_grupo(g) for g in u.grupos_existentes()}
VAL = {g: c1.generar_series(g, u.N_VAL, u.SEED_VAL) for g in G}
NS = [500, 1000, 2000, 5000]
NMAX = {g: (10000 if g in u.GEN_SENAL_DEBIL else 5000) for g in G}
ALFAS = ["0.001", "0.002", "0.005", "0.01", "dec", "0.05"]
ETQ_ALFA = {"0.001": "α=0.001", "0.002": "α=0.002", "0.005": "α=0.005", "0.01": "α=0.01", "dec": "α=0.05→0.001", "0.05": "α=0.05 const."}
SALIDAS = []


def rng(clave):
    return np.random.default_rng([c1.BOOT_SEED, 9, zlib.crc32(clave.encode())])


def csv_out(nombre, filas, columnas=None):
    columnas = columnas or list(dict.fromkeys(k for f in filas for k in f))
    with open(u.RES / nombre, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in fila.items()})
    SALIDAS.append(nombre)


# -----------------------------------------------------------------------------
# Acceso a "brazos" (5 semillas de una configuración)
# -----------------------------------------------------------------------------
def _grupo(nombre, g, a):
    runs = sorted([r for r in GRUPOS[nombre] if r["generador"] == g and r["algoritmo"] == a], key=lambda r: r["seed"])
    assert [r["seed"] for r in runs] == u.SEMILLAS, (nombre, g, a)
    return runs


def _marco(etapa, g, a, variante):
    runs = sorted([r for r in etapas[etapa] if r["generador"] == g and r["algoritmo"] == a and r["variante"] == variante], key=lambda r: r["seed"])
    assert [r["seed"] for r in runs] == u.SEMILLAS, (etapa, variante, g, a)
    return runs


def arm(g, a, et):
    """Etiquetas: E0@N, F@N, E1/E2/E3, alfa=<valor|dec>@N, disc:<variante de Marco>."""
    if et.startswith("E0@"):
        return _grupo(f"E0_N{int(et[3:])}", g, a)
    if et == "E0eps0.1@5000":
        return _grupo("E0eps0p1_N5000", g, a)
    if et.startswith("F@"):
        n = int(et[2:])
        return cadena[(g, a)]["E3"] if n == 5000 else _grupo(f"F_N{n}", g, a)
    if et in ("E1", "E2", "E3"):
        return cadena[(g, a)][et]
    if et.startswith("alfa="):
        al, n = et[5:].split("@")
        n = int(n)
        if al == "0.05":
            return arm(g, a, f"E0@{n}")
        if al == "dec":
            assert n == 5000
            return _marco("alpha", g, a, "α=0.05→0.001")
        if n == 5000:
            return _marco("alpha", g, a, f"α={al}")
        return _grupo(f"alfa{al}_N{n}".replace(".", "p"), g, a)
    if et.startswith("disc:"):
        return _marco("discretizacion", g, a, et[5:])
    raise KeyError(et)


def existe(g, a, et):
    try:
        arm(g, a, et)
        return True
    except (KeyError, AssertionError):
        return False


def M(runs, campo, met="logret"):
    return np.array([r[campo][met] for r in runs], float)


def vec(r):
    comprar, vender, _, _ = c1.politica_desde_Q(r["Q"])
    return np.concatenate([comprar, vender])


def acuerdo_semillas(runs):
    V = [vec(r) for r in runs]
    return float(np.mean([(V[i] == V[j]).mean() for i in range(len(V)) for j in range(i + 1, len(V))]))


def n_politicas(runs):
    return len({tuple(vec(r)) for r in runs})


def nunca_opera(runs):
    return int(sum(1 for r in runs if r["val"]["operaciones"].sum() == 0 and r["test"]["operaciones"].sum() == 0))


_VIS = {}


def frec_visitas(clave, r, g):
    """Frecuencia relativa de visita a los 20 estados (pos0 bins 0-9, pos1 bins 0-9) al jugar la política sobre VALIDACIÓN."""
    if clave not in _VIS:
        visitas = [[0] * 10, [0] * 10]
        c1.evaluar_tabla(r["Q"], r["cortes"], VAL[g], visitas)
        v = np.array(visitas, float).reshape(-1)
        _VIS[clave] = v / v.sum()
    return _VIS[clave]


def acuerdo_ponderado(clave, rb, ra, g):
    """Acuerdo de la política ra con la de referencia rb, ponderado por las visitas (en validación) de rb."""
    return float(((vec(ra) == vec(rb)) * frec_visitas(clave, rb, g)).sum())


def descr_cfg(cfg):
    a = f"{cfg['alpha']}" if cfg["alpha_decay"] == 1 else f"{cfg['alpha']}→{cfg['alpha_min']}"
    e = f"{cfg['eps_inicial']}" if cfg["eps_inicial"] == cfg["eps_final"] else f"1→0.01 f={cfg['frac_exploracion']}"
    return f"α={a} γ={cfg['gamma']} ε={e} N={cfg['n_episodios']} bins={cfg['n_bins']}/{cfg['discretizacion']}"


def resumen_basico(runs, g, a, clave_rng):
    val = M(runs, "val")
    tst = M(runs, "test")
    v_mean = val.mean(axis=1) * 100
    ic = est.media_ic(tst, rng(clave_rng))
    return dict(val_media_pp=float(v_mean.mean()), val_sd_semillas_pp=float(v_mean.std(ddof=1)),
                val_ops=float(M(runs, "val", "operaciones").mean()), test_media_pp=ic["media"],
                test_ic_series_lo=ic["ic_series_lo"], test_ic_series_hi=ic["ic_series_hi"],
                test_ic_pob_lo=ic["ic_pob_lo"], test_ic_pob_hi=ic["ic_pob_hi"],
                test_ops=float(M(runs, "test", "operaciones").mean()), semillas_nunca_opera=nunca_opera(runs),
                politicas_distintas=n_politicas(runs), acuerdo_entre_semillas=acuerdo_semillas(runs),
                tiempo_medio_s=float(np.mean([r["tiempo"] for r in runs])))


# =============================================================================
# BLOQUE 1 — Punto de partida (E0, N=2000)
# =============================================================================
print("== Bloque 1: configuración inicial")
b1 = []
for g in G:
    for a in A:
        runs = arm(g, a, "E0@2000")
        fila = dict(generador=g, algoritmo=a, configuracion=descr_cfg(runs[0]["cfg"]))
        fila.update(resumen_basico(runs, g, a, f"b1|{g}|{a}"))
        e3 = cadena[(g, a)]["E3"]
        fila["ref_E3_val_media_pp"] = float(M(e3, "val").mean() * 100)
        fila["ref_E3_test_media_pp"] = float(M(e3, "test").mean() * 100)
        b1.append(fila)
csv_out("bloque1_e0_n2000.csv", b1)

# =============================================================================
# BLOQUE 2 — alpha (N=5000, base de Marco: γ=0.95, ε decay f=0.6, 10 cuantiles)
# =============================================================================
print("== Bloque 2: alpha")
b2 = []
for g in G:
    for a in A:
        ref = arm(g, a, "alfa=0.05@5000")
        for k in ALFAS:
            runs = arm(g, a, f"alfa={k}@5000")
            fila = dict(generador=g, algoritmo=a, alpha=ETQ_ALFA[k], origen="Fase 2 (nuevo)" if k == "0.05" else "Marco (alpha.pkl)")
            fila.update(resumen_basico(runs, g, a, f"b2|{g}|{a}|{k}"))
            if k != "0.05":
                cmp = est.comparar(M(runs, "val"), M(ref, "val"), rng(f"b2c|{g}|{a}|{k}"))
                fila.update({f"val_vs_alfa005_{x}": cmp[x] for x in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi", "prop_A_gana")})
            b2.append(fila)
csv_out("bloque2_alpha_n5000.csv", b2)

# 2b. ¿es el ruido de Q ∝ sqrt(alpha)? dispersión entre semillas de la ventaja de operar, por alpha constante
ruido = []
for g in G:
    for a in A:
        for k in ["0.001", "0.002", "0.005", "0.01", "0.05"]:
            runs = arm(g, a, f"alfa={k}@5000")
            D = np.array([np.concatenate([r["Q"][:, 0, 2] - r["Q"][:, 0, 1], r["Q"][:, 1, 0] - r["Q"][:, 1, 1]]) for r in runs])
            sd = D.std(axis=0, ddof=1)
            ruido.append(dict(generador=g, algoritmo=a, alpha=float(k), sd_dQ_entre_semillas=float(sd.mean()),
                              media_abs_dQ=float(np.abs(D.mean(axis=0)).mean()),
                              cociente_senal_ruido=float(np.abs(D.mean(axis=0)).mean() / sd.mean())))
pend = []
for g in G:
    for a in A:
        f = [r for r in ruido if r["generador"] == g and r["algoritmo"] == a]
        pend.append(np.polyfit(np.log([r["alpha"] for r in f]), np.log([r["sd_dQ_entre_semillas"] for r in f]), 1)[0])
        for r in f:
            r["pendiente_loglog_sd_vs_alpha"] = float(pend[-1])
csv_out("bloque2_ruido_Q.csv", ruido)

# 2c. visitas (grupos instrumentados): ¿se dejan de explorar las acciones de operar?
vis = []
for nombre, runs_g in GRUPOS.items():
    for g in G:
        for a in A:
            runs = [r for r in runs_g if r["generador"] == g and r["algoritmo"] == a]
            if not runs:
                continue
            for alpha in sorted({r["cfg"]["alpha"] for r in runs}):
                rr = [r for r in runs if r["cfg"]["alpha"] == alpha]
                filas_run = []
                for r in rr:
                    comprar = c1.politica_desde_Q(r["Q"])[0]
                    hold = ~comprar
                    late = r["visitas_tarde"][:, 0, 2]
                    tot = r["visitas_total"][:, 0, 2]
                    vende = c1.politica_desde_Q(r["Q"])[1]
                    late_hold_dentro = r["visitas_tarde"][:, 1, 1]
                    filas_run.append((float(hold.mean()),
                                      float(np.median(late[hold])) if hold.any() else np.nan,
                                      float((late[hold] < 50).mean()) if hold.any() else np.nan,
                                      float(r["cfg"]["alpha"] * tot.mean()),
                                      float(tot.mean()),
                                      float(vende.mean()),
                                      float(np.median(late_hold_dentro[vende])) if vende.any() else np.nan,
                                      float(np.mean([r["Q"][:, 1, 1].mean(), 0.0]) * 0 + r["Q"][:, 1, 1].mean()),
                                      float(r["Q"][:, 1, 0].mean())))
                arr = np.array(filas_run, float)
                vis.append(dict(grupo=nombre, generador=g, algoritmo=a, alpha=alpha, n_episodios=rr[0]["cfg"]["n_episodios"], semillas=len(rr),
                                frac_estados_sin_compra=float(np.nanmean(arr[:, 0])), mediana_actualizaciones_compra_tardias_en_esos_estados=float(np.nanmean(arr[:, 1])),
                                frac_esos_estados_con_menos_de_50_actualizaciones_tardias=float(np.nanmean(arr[:, 2])),
                                indice_aprendizaje_alpha_x_actualizaciones_compra=float(np.nanmean(arr[:, 3])),
                                actualizaciones_compra_por_estado=float(np.nanmean(arr[:, 4])),
                                frac_estados_dentro_que_vende=float(np.nanmean(arr[:, 5])),
                                mediana_act_tardias_mantener_dentro_en_esos_estados=float(np.nanmean(arr[:, 6])),
                                Q_medio_mantener_dentro=float(np.nanmean(arr[:, 7])), Q_medio_vender_dentro=float(np.nanmean(arr[:, 8]))))
csv_out("bloque2_visitas.csv", vis)

# 2d. alpha x N (mercados de señal débil)
b2n = []
for g in u.GEN_SENAL_DEBIL:
    for a in A:
        for k in ["0.002", "0.005", "0.05"]:
            for n in NS + [10000]:
                runs = arm(g, a, f"alfa={k}@{n}")
                ref = arm(g, a, f"alfa=0.05@{n}")
                fila = dict(generador=g, algoritmo=a, alpha=float(k), N=n)
                fila.update({k2: v for k2, v in resumen_basico(runs, g, a, f"b2n|{g}|{a}|{k}|{n}").items()
                             if k2 in ("val_media_pp", "val_sd_semillas_pp", "val_ops", "semillas_nunca_opera", "politicas_distintas", "acuerdo_entre_semillas")})
                if k != "0.05":
                    cmp = est.comparar(M(runs, "val"), M(ref, "val"), rng(f"b2nc|{g}|{a}|{k}|{n}"))
                    fila.update({f"val_vs_alfa005_{x}": cmp[x] for x in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi")})
                b2n.append(fila)
csv_out("bloque2_alpha_x_N.csv", b2n)

# 2e. intervención sobre el bloqueo: E0 con ε constante 0.1 (reversión y regímenes, N=5000)
b2e = []
for g in u.GEN_SENAL_DEBIL:
    for a in A:
        base_r, int_r = arm(g, a, "E0@5000"), arm(g, a, "E0eps0.1@5000")
        for nombre, runs in (("E0 (ε 1→0.01, f=0.6)", base_r), ("E0 con ε=0.1 constante", int_r)):
            fila = dict(generador=g, algoritmo=a, configuracion=nombre)
            fila.update({k: v for k, v in resumen_basico(runs, g, a, f"b2e|{g}|{a}|{nombre}").items()
                         if k in ("val_media_pp", "val_sd_semillas_pp", "val_ops", "semillas_nunca_opera", "politicas_distintas", "test_media_pp")})
            fila["frac_estados_dentro_que_vende"] = float(np.mean([c1.politica_desde_Q(r["Q"])[1].mean() for r in runs]))
            fila["Q_medio_mantener_dentro"] = float(np.mean([r["Q"][:, 1, 1].mean() for r in runs]))
            fila["Q_medio_vender_dentro"] = float(np.mean([r["Q"][:, 1, 0].mean() for r in runs]))
            fila["mediana_act_mantener_dentro_total"] = float(np.mean([np.median(r["visitas_total"][:, 1, 1]) for r in runs]))
            if runs is int_r:
                cmp = est.comparar(M(int_r, "val"), M(base_r, "val"), rng(f"b2ec|{g}|{a}"))
                fila.update({f"val_vs_E0_{x}": cmp[x] for x in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi")})
            b2e.append(fila)
csv_out("bloque2_intervencion_epsilon.csv", b2e)

# =============================================================================
# BLOQUE 3 — número de episodios (E0 y configuración final F), selección sólo en validación
# =============================================================================
print("== Bloque 3: episodios")
b3, b3n = [], []
for cfg_nom in ("E0", "F"):
    for g in G:
        for a in A:
            ns = NS + ([10000] if g in u.GEN_SENAL_DEBIL else [])
            arms = {n: arm(g, a, f"{cfg_nom}@{n}") for n in ns}
            nmax = ns[-1]
            ref = arms[nmax]
            filas_cfg = []
            for i, n in enumerate(ns):
                runs = arms[n]
                fila = dict(config=cfg_nom, generador=g, algoritmo=a, N=n, N_referencia=nmax)
                fila.update(resumen_basico(runs, g, a, f"b3|{cfg_nom}|{g}|{a}|{n}"))
                fila["origen"] = "Marco (epsilon.pkl)" if (cfg_nom == "F" and n == 5000) else "Fase 2 (nuevo)"
                cmp = est.comparar(M(runs, "val"), M(ref, "val"), rng(f"b3c|{cfg_nom}|{g}|{a}|{n}")) if n != nmax else None
                if cmp:
                    fila.update({f"val_vs_Nref_{x}": cmp[x] for x in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi")})
                    fila["acuerdo_con_politica_Nref"] = float(np.mean([(vec(ra) == vec(rb)).mean() for ra, rb in zip(runs, ref)]))
                    fila["acuerdo_con_Nref_ponderado_visitas"] = float(np.mean([acuerdo_ponderado((cfg_nom, g, a, nmax, rb["seed"]), rb, ra, g) for ra, rb in zip(runs, ref)]))
                else:
                    fila.update(val_vs_Nref_dif=0.0, val_vs_Nref_ic_series_lo=0.0, val_vs_Nref_ic_series_hi=0.0, acuerdo_con_politica_Nref=1.0,
                                acuerdo_con_Nref_ponderado_visitas=1.0)
                if i + 1 < len(ns):
                    sig = arms[ns[i + 1]]
                    fila["acuerdo_con_politica_siguiente_N"] = float(np.mean([(vec(ra) == vec(rb)).mean() for ra, rb in zip(runs, sig)]))
                else:
                    fila["acuerdo_con_politica_siguiente_N"] = 1.0
                filas_cfg.append(fila)
            b3.extend(filas_cfg)
            # N suficiente (empírico, con permanencia): cada criterio debe cumplirse en N y en todos los N mayores de la rejilla
            crit = {
                "C1_val_no_peor_que_Nref": [not (f["val_vs_Nref_ic_series_hi"] < 0) for f in filas_cfg],
                "C2_politica_acuerdo_con_Nref_>=0.9": [f["acuerdo_con_Nref_ponderado_visitas"] >= 0.9 for f in filas_cfg],
                "C3_politica_estable_al_alargar_>=0.9": [f["acuerdo_con_politica_siguiente_N"] >= 0.9 for f in filas_cfg],
                "C4_semillas_coherentes_>=0.9": [f["acuerdo_entre_semillas"] >= 0.9 for f in filas_cfg],
            }
            crit["TODOS"] = [all(crit[k][i] for k in list(crit)) for i in range(len(ns))]
            fila_n = dict(config=cfg_nom, generador=g, algoritmo=a, N_referencia=nmax)
            for k, v in crit.items():
                ok = None
                for i in range(len(ns) - 1, -1, -1):
                    if all(v[i:]):
                        ok = ns[i]
                    else:
                        break
                fila_n[f"N_suficiente_{k}"] = ok if ok is not None else f">{nmax}"
            b3n.append(fila_n)
csv_out("bloque3_episodios.csv", b3)
csv_out("bloque3_n_suficiente.csv", b3n)

# =============================================================================
# BLOQUE 4/5 — cadena E0 -> E1 -> E2 -> E3 (N=5000, comparación controlada) y descomposición N frente a hiperparámetros
# =============================================================================
print("== Bloque 4/5: cadena y comparaciones")
cero = np.zeros((1, u.N_TEST))
cero_val = np.zeros((1, u.N_VAL))
b4, b4d, fam_filas = [], [], []
ETAPA_ARM = [("E0 (N=5000, mismo N)", "E0@5000"), ("E1: α", "E1"), ("E2: γ", "E2"), ("E3: ε", "E3")]
for g in G:
    for a in A:
        prev = None
        e0n = arm(g, a, "E0@5000")
        for nombre, et in [("E0 histórica (N=2000)", "E0@2000")] + ETAPA_ARM:
            runs = arm(g, a, et)
            fila = dict(generador=g, algoritmo=a, etapa=nombre, etiqueta=et, configuracion=descr_cfg(runs[0]["cfg"]))
            fila.update(resumen_basico(runs, g, a, f"b4|{g}|{a}|{et}"))
            if et in ("E1", "E2", "E3"):
                prev_runs = arm(g, a, {"E1": "E0@5000", "E2": "E1", "E3": "E2"}[et])
                fila["sin_cambio_respecto_a_etapa_previa"] = all(np.array_equal(r1["Q"], r2["Q"]) for r1, r2 in zip(runs, prev_runs))
                cv = est.comparar(M(runs, "val"), M(prev_runs, "val"), rng(f"b4v|{g}|{a}|{et}"))
                ct = est.comparar(M(runs, "test"), M(prev_runs, "test"), rng(f"b4t|{g}|{a}|{et}"))
                for x in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi"):
                    fila[f"val_vs_previa_{x}"] = cv[x]
                    fila[f"test_vs_previa_{x}"] = ct[x]
                fila["val_vs_previa_prop_mejora"] = cv["prop_A_gana"]
            if et != "E0@5000":
                cv0 = est.comparar(M(runs, "val"), M(e0n, "val"), rng(f"b4v0|{g}|{a}|{et}"))
                ct0 = est.comparar(M(runs, "test"), M(e0n, "test"), rng(f"b4t0|{g}|{a}|{et}"))
                for x in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi", "prop_A_gana"):
                    fila[f"val_vs_E0N5000_{x}"] = cv0[x]
                    fila[f"test_vs_E0N5000_{x}"] = ct0[x]
            cef = est.comparar(M(runs, "test"), cero, rng(f"b4ef|{g}|{a}|{et}"))
            fila.update(test_vs_efectivo_dif=cef["dif"], test_vs_efectivo_ic_pob_lo=cef["ic_pob_lo"], test_vs_efectivo_ic_pob_hi=cef["ic_pob_hi"],
                        test_vs_efectivo_ic_series_lo=cef["ic_series_lo"], test_vs_efectivo_ic_series_hi=cef["ic_series_hi"])
            b4.append(fila)
        # descomposición: duración del entrenamiento frente a hiperparámetros (validación y test)
        pares = [("hiperparámetros a N=5000: E3(5000) − E0(5000)", "F@5000", "E0@5000"),
                 ("hiperparámetros a N=2000: E3 reentrenada(2000) − E0(2000)", "F@2000", "E0@2000"),
                 ("duración con hiperparámetros de E0: E0(5000) − E0(2000)", "E0@5000", "E0@2000"),
                 ("duración con hiperparámetros de E3: E3(5000) − E3(2000)", "F@5000", "F@2000"),
                 ("TOTAL histórico: E3(5000) − E0(2000)", "F@5000", "E0@2000")]
        for nombre, x, y in pares:
            for conj in ("val", "test"):
                c = est.comparar(M(arm(g, a, x), conj), M(arm(g, a, y), conj), rng(f"b4d|{g}|{a}|{nombre}|{conj}"))
                fila = dict(generador=g, algoritmo=a, contraste=nombre, conjunto=conj, **{k: c[k] for k in (
                    "dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi", "prop_A_gana", "prop_empate", "prop_A_pierde",
                    "frac_pares_semilla_A_gana", "p_t4")})
                b4d.append(fila)
                if conj == "test" and nombre.startswith(("hiperparámetros a N=5000", "hiperparámetros a N=2000", "TOTAL")):
                    fam_filas.append(fila)
# Holm sobre la familia declarada (3 contrastes x 8 pares generador-algoritmo = 24, en test)
for f, p in zip(fam_filas, est.holm([f["p_t4"] for f in fam_filas])):
    f["p_holm_familia24"] = p
    f["clasificacion"] = est.clasifica(dict(f, se_pob=1.0), p)
for f in b4d:
    f.setdefault("p_holm_familia24", np.nan)
    f.setdefault("clasificacion", "exploratoria (fuera de la familia de 24)")
csv_out("bloque4_cadena.csv", b4)
csv_out("bloque4_descomposicion.csv", b4d)

# E4: discretización (experimento posterior de sensibilidad, NO es una etapa secuencial)
e4 = []
for g in G:
    for a in A:
        ref = arm(g, a, "disc:10 cuantiles")
        for var in ("5 cuantiles", "10 cuantiles", "20 cuantiles", "10 uniformes"):
            runs = arm(g, a, f"disc:{var}")
            fila = dict(generador=g, algoritmo=a, variante=var)
            fila.update(resumen_basico(runs, g, a, f"e4|{g}|{a}|{var}"))
            if var != "10 cuantiles":
                cv = est.comparar(M(runs, "val"), M(ref, "val"), rng(f"e4v|{g}|{a}|{var}"))
                fila.update({f"val_vs_10cuantiles_{x}": cv[x] for x in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi")})
            e4.append(fila)
csv_out("bloque4_e4_discretizacion.csv", e4)

# Comparación de la inicial frente a la optimizada (tabla principal): E0(N=2000) frente a E3 (Marco) y frente a efectivo
b5 = []
for g in G:
    for a in A:
        for nombre, x in (("E0 (N=2000)", "E0@2000"), ("E0 (N=5000)", "E0@5000"), ("E3 final (N=5000)", "E3")):
            runs = arm(g, a, x)
            fila = dict(generador=g, algoritmo=a, configuracion=nombre)
            fila.update(resumen_basico(runs, g, a, f"b5|{g}|{a}|{x}"))
            b5.append(fila)
csv_out("bloque5_inicial_vs_optimizada.csv", b5)

# =============================================================================
# BLOQUE 6 — Q-Learning frente a SARSA y frente a Buy & Hold, por configuración (series de test idénticas)
# =============================================================================
print("== Bloque 6: Q-Learning vs SARSA y vs Buy & Hold")
TEST = {g: c1.generar_series(g, u.N_TEST, u.SEED_TEST) for g in G}
BH = {g: np.log(np.array([1000 * 0.999 / p[5] * p[-1] for p in TEST[g]]) / 1000.0)[None, :] for g in G}
b6, b6bh = [], []
CONFIGS6 = [("E0 (N=2000)", "E0@2000"), ("E0 (N=5000)", "E0@5000"), ("E1: α", "E1"), ("E3 final (N=5000)", "E3")]
for g in G:
    for nombre, et in CONFIGS6:
        q, s_ = arm(g, "qlearning", et), arm(g, "sarsa", et)
        for conj in ("val", "test"):
            cmp = est.comparar(M(q, conj), M(s_, conj), rng(f"b6|{g}|{et}|{conj}"))
            b6.append(dict(generador=g, configuracion=nombre, conjunto=conj, **{k: cmp[k] for k in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi", "prop_A_gana")},
                           ops_q=float(M(q, conj, "operaciones").mean()), ops_s=float(M(s_, conj, "operaciones").mean()),
                           nunca_opera_q=nunca_opera(q), nunca_opera_s=nunca_opera(s_)))
        for a in A:
            runs = arm(g, a, et)
            cmp = est.comparar(M(runs, "test"), BH[g], rng(f"b6bh|{g}|{a}|{et}"))
            b6bh.append(dict(generador=g, algoritmo=a, configuracion=nombre, **{"test_vs_BuyHold_" + k: cmp[k] for k in ("dif", "ic_series_lo", "ic_series_hi", "ic_pob_lo", "ic_pob_hi", "prop_A_gana")}))
csv_out("bloque6_qlearning_vs_sarsa.csv", b6)
csv_out("bloque6_vs_buyhold.csv", b6bh)

# =============================================================================
# FIGURAS
# =============================================================================
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "savefig.dpi": 170, "savefig.bbox": "tight"})


def barra_ic(ax, x, lo, hi, color, grueso=False, ancho=1.3):
    ax.plot([x, x], [lo, hi], color=color, lw=5 if grueso else ancho, alpha=0.5 if grueso else 0.95, solid_capstyle="butt")


# Figura 1: inicial frente a optimizada (test; IC series grueso e IC población fino)
fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.9))
cats = [("E0@2000", "E0\nN=2000"), ("E0@5000", "E0\nN=5000"), ("E3", "E3 final\nN=5000")]
for ax, g in zip(axs, G):
    for k, a in enumerate(A):
        for i, (et, _) in enumerate(cats):
            runs = arm(g, a, et)
            ic = est.media_ic(M(runs, "test"), rng(f"f1|{g}|{a}|{et}"))
            x = i + (k - 0.5) * 0.3
            barra_ic(ax, x, ic["ic_pob_lo"], ic["ic_pob_hi"], COL[a])
            barra_ic(ax, x, ic["ic_series_lo"], ic["ic_series_hi"], COL[a], grueso=True)
            ax.plot(x, ic["media"], "o", color="black", ms=4, zorder=5)
            ax.plot([x] * 5, M(runs, "test").mean(axis=1) * 100, ".", color=COL[a], ms=3, alpha=0.8, zorder=4)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[1] for c in cats])
    ax.set_title(NG[g], fontweight="bold")
axs[0].set_ylabel("log-retorno medio en test (pp)")
for a in A:
    axs[0].plot([], [], "s", color=COL[a], label=NA[a])
axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Configuración inicial (E0) frente a final (E3). Línea 0 = efectivo. Puntos pequeños = cada semilla; "
             "barra gruesa = IC95 series; fina = IC95 series + semillas (t, 4 g.l.)", y=1.03, fontsize=9)
fig.savefig(u.FIG / "fig1_inicial_vs_optimizada.png")
plt.close(fig)
SALIDAS.append("fig1_inicial_vs_optimizada.png")

# Figura 2: alpha (validación; N=5000)
fig, axs = plt.subplots(2, 4, figsize=(15.5, 6.2), sharex="col")
xs = {k: i for i, k in enumerate(ALFAS)}
for j, g in enumerate(G):
    for k, a in enumerate(A):
        medias, sds, opsv = [], [], []
        for al in ALFAS:
            runs = arm(g, a, f"alfa={al}@5000")
            v = M(runs, "val").mean(axis=1) * 100
            x = xs[al] + (k - 0.5) * 0.22
            axs[0, j].plot([x] * 5, v, ".", color=COL[a], alpha=0.55, ms=5)
            axs[0, j].plot([x - 0.07, x + 0.07], [v.mean()] * 2, color=COL[a], lw=2)
            axs[1, j].plot([x] * 5, M(runs, "val", "operaciones").mean(axis=1), ".", color=COL[a], alpha=0.55, ms=5)
            axs[1, j].plot([x - 0.07, x + 0.07], [M(runs, "val", "operaciones").mean()] * 2, color=COL[a], lw=2)
    axs[0, j].axhline(0, color="black", lw=0.8)
    if g == "browniano":
        axs[0, j].set_ylim(-1, 1)
        axs[1, j].set_ylim(-0.1, 1.1)
    axs[0, j].set_title(NG[g] + (" (eje ampliado ±1 pp)" if g == "browniano" else ""), fontweight="bold")
    axs[1, j].set_xticks(range(len(ALFAS)))
    axs[1, j].set_xticklabels([ETQ_ALFA[k].replace("α=", "").replace(" const.", "\n(E0)") for k in ALFAS], fontsize=7.5)
    axs[1, j].set_xlabel("α (N=5000)")
axs[0, 0].set_ylabel("log-retorno VALIDACIÓN (pp)\nraya = media de 5 semillas")
axs[1, 0].set_ylabel("operaciones por serie (validación)")
for a in A:
    axs[0, 0].plot([], [], "s", color=COL[a], label=NA[a])
axs[0, 0].legend(fontsize=8)
fig.suptitle("Sensibilidad a α con la base de Marco (γ=0.95, ε 1→0.01 f=0.6, N=5000). α=0.05 constante es nuevo (Fase 2); el resto son artefactos de Marco", y=1.01, fontsize=9)
fig.tight_layout()
fig.savefig(u.FIG / "fig2_sensibilidad_alpha.png")
plt.close(fig)
SALIDAS.append("fig2_sensibilidad_alpha.png")

# Figura 2b: ruido de Q frente a alpha (log-log) y alpha x N
fig, axs = plt.subplots(1, 3, figsize=(15, 3.9))
for a in A:
    for g, ls in (("reversion", "-"), ("regimenes", "--")):
        f = [r for r in ruido if r["generador"] == g and r["algoritmo"] == a]
        axs[0].plot([r["alpha"] for r in f], [r["sd_dQ_entre_semillas"] for r in f], ls, marker="o", color=COL[a], label=f"{NA[a]} · {NG[g].split()[0]}")
al = np.array([0.001, 0.05])
axs[0].plot(al, 0.02 * np.sqrt(al / 0.05), "k:", label="pendiente 1/2 (referencia; nivel arbitrario)")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel("α")
axs[0].set_ylabel("SD entre semillas de Q(operar) − Q(mantener)")
axs[0].legend(fontsize=7)
axs[0].set_title("Ruido de Q frente a α (N=5000)", fontweight="bold")
for ax, g in zip(axs[1:], u.GEN_SENAL_DEBIL):
    for a in A:
        for k, ls, mk in (("0.002", "-", "o"), ("0.005", "--", "s"), ("0.05", ":", "^")):
            f = sorted([r for r in b2n if r["generador"] == g and r["algoritmo"] == a and r["alpha"] == float(k)], key=lambda r: r["N"])
            ax.errorbar([r["N"] for r in f], [r["val_media_pp"] for r in f], yerr=[r["val_sd_semillas_pp"] for r in f], color=COL[a], ls=ls,
                        marker=mk, ms=4, capsize=2, lw=1.2, alpha=0.85, label=f"{NA[a]} α={k}")
    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("episodios N")
    ax.set_title(f"{NG[g]}: validación frente a N según α", fontweight="bold")
axs[1].set_ylabel("log-retorno validación (pp), media ± SD semillas")
axs[1].legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(u.FIG / "fig2b_ruido_Q_y_alpha_por_N.png")
plt.close(fig)
SALIDAS.append("fig2b_ruido_Q_y_alpha_por_N.png")

# Figura 3: número de episodios (validación) + estabilidad de la política
fig, axs = plt.subplots(2, 4, figsize=(15.5, 6.2), sharex=True)
for j, g in enumerate(G):
    for a in A:
        for cfg_nom, ls, mk in (("E0", "--", "s"), ("F", "-", "o")):
            f = sorted([r for r in b3 if r["config"] == cfg_nom and r["generador"] == g and r["algoritmo"] == a], key=lambda r: r["N"])
            axs[0, j].errorbar([r["N"] for r in f], [r["val_media_pp"] for r in f], yerr=[r["val_sd_semillas_pp"] for r in f], color=COL[a],
                               ls=ls, marker=mk, ms=4, capsize=2, lw=1.4, label=f"{NA[a]} · {'E0' if cfg_nom == 'E0' else 'E3 final'}")
            axs[1, j].plot([r["N"] for r in f], [r["acuerdo_con_politica_siguiente_N"] for r in f], color=COL[a], ls=ls, marker=mk, ms=4, lw=1.4)
    axs[0, j].axhline(0, color="black", lw=0.8)
    axs[0, j].set_title(NG[g], fontweight="bold")
    axs[1, j].set_xscale("log")
    axs[1, j].set_ylim(0.3, 1.02)
    axs[1, j].set_xlabel("episodios N (ε se recalibra con N)")
axs[0, 0].set_ylabel("log-retorno VALIDACIÓN (pp)\nmedia ± SD entre semillas")
axs[1, 0].set_ylabel("acuerdo de política con la del\nsiguiente N (mismas semillas)")
axs[0, 0].legend(fontsize=7)
for ax in axs.ravel():
    ax.set_xlim(400, 13000)
fig.suptitle("Efecto del número de episodios (ε se recalibra para cada N). E3 con N=5000 es el artefacto de Marco; el resto son reentrenamientos de la Fase 2. "
             "Abajo: acuerdo de la política con la obtenida con el siguiente N (1 = nada cambia al alargar)", y=1.01, fontsize=9)
fig.tight_layout()
fig.savefig(u.FIG / "fig3_numero_de_episodios.png")
plt.close(fig)
SALIDAS.append("fig3_numero_de_episodios.png")

# Figura 4: cadena E0 -> E3 a N=5000 (validación y test; todas las semillas)
fig, axs = plt.subplots(2, 4, figsize=(15.5, 6.2), sharex=True)
etq = ["E0\n(α=0.05)", "E1\n(+α)", "E2\n(+γ)", "E3\n(+ε)"]
for j, g in enumerate(G):
    for k, a in enumerate(A):
        for i2, (conj, fila_ax) in enumerate((("val", 0), ("test", 1))):
            ax = axs[fila_ax, j]
            ys = []
            for i, et in enumerate(["E0@5000", "E1", "E2", "E3"]):
                v = M(arm(g, a, et), conj).mean(axis=1) * 100
                x = i + (k - 0.5) * 0.18
                ax.plot([x] * 5, v, ".", color=COL[a], alpha=0.5, ms=5)
                ys.append(v.mean())
            ax.plot(np.arange(4) + (k - 0.5) * 0.18, ys, "-", color=COL[a], lw=1.5, marker="_", ms=14)
    for fila_ax in (0, 1):
        axs[fila_ax, j].axhline(0, color="black", lw=0.8)
    axs[0, j].set_title(NG[g], fontweight="bold")
    axs[1, j].set_xticks(range(4))
    axs[1, j].set_xticklabels(etq, fontsize=8)
axs[0, 0].set_ylabel("log-retorno VALIDACIÓN (pp)\n(base de la selección)")
axs[1, 0].set_ylabel("log-retorno TEST (pp)\n(configuraciones ya fijadas)")
for a in A:
    axs[0, 0].plot([], [], "s", color=COL[a], label=NA[a])
axs[0, 0].legend(fontsize=8)
fig.suptitle("Cadena E0 → E1 → E2 → E3, todas con N=5000 episodios (comparación controlada). Puntos = semillas; raya = media. Etapas sin cambio de configuración coinciden con la anterior", y=1.01, fontsize=9)
fig.tight_layout()
fig.savefig(u.FIG / "fig4_cadena_E0_E3.png")
plt.close(fig)
SALIDAS.append("fig4_cadena_E0_E3.png")

# Figura 5: política aprendida en reversión: E0 (N=2000) frente a E3, con la señal de Marco (senal.pkl) arriba
senal = c1.cargar_pickle("senal")["reversion"]
fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 2.2, 2.2], hspace=0.38, wspace=0.28)
ax0 = fig.add_subplot(gs[0, :])
media, se = senal
ax0.bar(range(10), media * 100, yerr=2 * se * 100, color=["#2a78d6" if m > 0 else "#e34948" for m in media], width=0.7, error_kw={"lw": 1})
for y in (0.2, -0.2):
    ax0.axhline(y, ls="--", lw=0.8, color="#52514e")
ax0.set_ylabel("log-ret. futuro a 10 pasos (%)")
ax0.set_title("Señal disponible por bin en reversión (de Marco, senal.pkl; ±2 e.s. optimista; líneas = coste ida y vuelta)", fontsize=9, fontweight="bold")
ax0.set_xticks(range(10))
for col, a in enumerate(A):
    for fila, (et, nom) in enumerate((("E0@2000", "E0 (N=2000, α=0.05)"), ("E3", "E3 final (N=5000)"))):
        ax = fig.add_subplot(gs[1 + fila, col])
        runs = arm("reversion", a, et)
        Mv = np.array([vec(r) for r in runs], float)
        ax.imshow(Mv, cmap="Blues", vmin=0, vmax=1.3, aspect="auto")
        ax.axvline(9.5, color="black", lw=1.2)
        for (i, k2), v in np.ndenumerate(Mv):
            if v:
                ax.text(k2, i, "C" if k2 < 10 else "V", ha="center", va="center", fontsize=7, color="white")
        ax.set_xticks(range(20))
        ax.set_xticklabels([str(k2 % 10) for k2 in range(20)], fontsize=6)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"s{r['seed']} ({r['test']['logret'].mean() * 100:+.0f}%)" for r in runs], fontsize=7)
        ax.set_title(f"{NA[a]} — {nom}", fontsize=8.5, fontweight="bold")
        ax.grid(False)
        if fila == 1:
            ax.set_xlabel("bin: sin posición (¿compra?) | con posición (¿vende?)", fontsize=7)
fig.suptitle("Reversión a la media: política por semilla (azul = opera). Entre paréntesis, log-retorno en test de esa semilla", y=0.98, fontsize=10)
fig.savefig(u.FIG / "fig5_politica_reversion.png")
plt.close(fig)
SALIDAS.append("fig5_politica_reversion.png")

print("Salidas:", SALIDAS)
