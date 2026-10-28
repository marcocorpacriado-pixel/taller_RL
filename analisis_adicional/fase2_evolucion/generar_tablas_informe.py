"""
generar_tablas_informe.py — Fase 2: convierte los CSV de resultados en tablas Markdown (resultados/tablas_informe.md).

    python -B -X utf8 analisis_adicional/fase2_evolucion/generar_tablas_informe.py

Las tablas del informe se pegan desde ese archivo: ninguna cifra se transcribe a mano.
"""
import sys

sys.dont_write_bytecode = True

import csv

import utilidades as u

c1 = u.c1
G, A = u.GENERADORES, u.ALGORITMOS
NG, NA = c1.ETIQUETA_GEN, c1.ETIQUETA_AG
CORTO = {"ciclico": "Cíclico", "browniano": "Browniano", "reversion": "Reversión", "regimenes": "Regímenes"}


def leer(n):
    with open(u.RES / n, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, d=1, signo=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    s = f"{v:+.{d}f}" if signo else f"{v:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


def pm(m, sd, d=1):
    return f"{f(m, d, True)} ± {f(sd, d)}"


def ic(lo, hi, d=1):
    return f"[{f(lo, d, True)}, {f(hi, d, True)}]"


def md(cols, filas):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(c) for c in fila) + " |" for fila in filas]
    return "\n".join(out)


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


T = []
b1 = leer("bloque1_e0_n2000.csv")
T.append(("T1", "E0 (configuración por defecto, N=2000). Validación = media ± SD entre 5 semillas; test con IC95 por series",
          md(["Generador", "Alg.", "Val. (pp)", "Test (pp) [IC series]", "Ops. val / test", "Semillas que nunca operan", "Políticas distintas", "Acuerdo entre semillas"],
             [[CORTO[r["generador"]], NA[r["algoritmo"]], pm(r["val_media_pp"], r["val_sd_semillas_pp"]),
               f"{f(r['test_media_pp'], 1, True)} {ic(r['test_ic_series_lo'], r['test_ic_series_hi'])}", f"{f(r['val_ops'])} / {f(r['test_ops'])}",
               f"{r['semillas_nunca_opera']}/5", f"{r['politicas_distintas']}/5", f(r["acuerdo_entre_semillas"], 2)] for r in b1])))

b5 = leer("bloque5_inicial_vs_optimizada.csv")
filas = []
for g in G:
    for a in A:
        celdas = []
        for cfgn in ("E0 (N=2000)", "E0 (N=5000)", "E3 final (N=5000)"):
            r = sel(b5, generador=g, algoritmo=a, configuracion=cfgn)[0]
            celdas.append(f"{f(r['test_media_pp'], 1, True)} {ic(r['test_ic_pob_lo'], r['test_ic_pob_hi'])}")
        filas.append([CORTO[g], NA[a]] + celdas)
T.append(("T2", "Inicial frente a optimizada: log-retorno medio en test (pp) con IC95 de población (series + semillas, t 4 g.l.). Efectivo = 0",
          md(["Generador", "Alg.", "E0 (N=2000)", "E0 (N=5000)", "E3 final (N=5000)"], filas)))

b4d = leer("bloque4_descomposicion.csv")
filas = []
for g in G:
    for a in A:
        celdas = []
        for nombre in ("hiperparámetros a N=5000: E3(5000) − E0(5000)", "hiperparámetros a N=2000: E3 reentrenada(2000) − E0(2000)",
                       "duración con hiperparámetros de E0: E0(5000) − E0(2000)", "duración con hiperparámetros de E3: E3(5000) − E3(2000)",
                       "TOTAL histórico: E3(5000) − E0(2000)"):
            r = sel(b4d, generador=g, algoritmo=a, contraste=nombre, conjunto="test")[0]
            celdas.append(f"{f(r['dif'], 1, True)} {ic(r['ic_pob_lo'], r['ic_pob_hi'])} ({f(100 * float(r['prop_A_gana']), 0)} %)")
        filas.append([CORTO[g], NA[a]] + celdas)
T.append(("T3", "Descomposición en TEST (pp): diferencia [IC95 de población] (% de series donde mejora). Primeras 2 columnas = efecto de los hiperparámetros; siguientes 2 = efecto de la duración; última = lo que mezcla la comparación histórica",
          md(["Generador", "Alg.", "Hiperpar. a N=5000", "Hiperpar. a N=2000", "Duración con E0", "Duración con E3", "TOTAL E3(5000) − E0(2000)"], filas)))
filas = []
for r in b4d:
    if r["conjunto"] == "test" and r["contraste"].startswith(("hiperparámetros a N=5000", "hiperparámetros a N=2000", "TOTAL")):
        filas.append([CORTO[r["generador"]], NA[r["algoritmo"]], r["contraste"].split(":")[0], f(r["dif"], 1, True), ic(r["ic_pob_lo"], r["ic_pob_hi"]),
                      f(r["p_t4"], 3), f(r["p_holm_familia24"], 3), r["clasificacion"]])
T.append(("T3b", "Familia declarada de 24 contrastes en test (Holm). Exploratoria: el test ya fue inspeccionado por Marco",
          md(["Generador", "Alg.", "Contraste", "Dif. (pp)", "IC95 pob.", "p (t4)", "p Holm", "Clasificación"], filas)))

b2 = leer("bloque2_alpha_n5000.csv")
filas = []
for g in G:
    for a in A:
        celdas = []
        for al in ("α=0.001", "α=0.002", "α=0.005", "α=0.01", "α=0.05→0.001", "α=0.05 const."):
            r = sel(b2, generador=g, algoritmo=a, alpha=al)[0]
            celdas.append(f"{pm(r['val_media_pp'], r['val_sd_semillas_pp'])} ({f(r['val_ops'], 0)} op.; {r['semillas_nunca_opera']}/5 no opera)")
        filas.append([CORTO[g], NA[a]] + celdas)
T.append(("T4", "Alpha (N=5000, base de Marco). Validación (pp): media ± SD entre semillas (operaciones por serie; semillas que nunca operan en val ni test)",
          md(["Generador", "Alg.", "α=0.001", "α=0.002", "α=0.005", "α=0.01", "α=0.05→0.001", "α=0.05 (E0, nuevo)"], filas)))

ru = leer("bloque2_ruido_Q.csv")
filas = []
for g in G:
    for a in A:
        f_ = sel(ru, generador=g, algoritmo=a)
        d = {float(r["alpha"]): r for r in f_}
        filas.append([CORTO[g], NA[a], f(f_[0]["pendiente_loglog_sd_vs_alpha"], 2),
                      f(d[0.002]["sd_dQ_entre_semillas"], 4), f(d[0.05]["sd_dQ_entre_semillas"], 4),
                      f(d[0.002]["cociente_senal_ruido"], 2), f(d[0.05]["cociente_senal_ruido"], 2)])
T.append(("T5", "Ruido de Q entre semillas: pendiente log-log de SD(ΔQ) frente a α (teoría 1/2), SD(ΔQ) y cociente señal/ruido en α=0.002 y α=0.05",
          md(["Generador", "Alg.", "Pendiente", "SD ΔQ (α=0.002)", "SD ΔQ (α=0.05)", "Señal/ruido (α=0.002)", "Señal/ruido (α=0.05)"], filas)))

vs = leer("bloque2_visitas.csv")
filas = []
for r in vs:
    if r["generador"] in u.GEN_SENAL_DEBIL and r["n_episodios"] == "5000" and r["grupo"] in ("REP", "E0_N5000"):
        filas.append([r["grupo"], CORTO[r["generador"]], NA[r["algoritmo"]], r["alpha"], f(r["frac_estados_sin_compra"], 2),
                      f(r["mediana_actualizaciones_compra_tardias_en_esos_estados"], 0), f(r["frac_esos_estados_con_menos_de_50_actualizaciones_tardias"], 2),
                      f(r["indice_aprendizaje_alpha_x_actualizaciones_compra"], 1), f(r["frac_estados_dentro_que_vende"], 2),
                      f(r["mediana_act_tardias_mantener_dentro_en_esos_estados"], 0), f(r["Q_medio_mantener_dentro"], 4), f(r["Q_medio_vender_dentro"], 4)])
T.append(("T6", "Visitas (N=5000, grupos instrumentados): estados sin compra en la política final, mediana de actualizaciones de 'comprar' tras ε≈0.01 en esos estados, fracción con < 50 y índice de aprendizaje α × actualizaciones totales de 'comprar'",
          md(["Grupo", "Generador", "Alg.", "α", "Estados sin compra", "Mediana act. tardías (comprar)", "Frac. < 50", "α × act. totales", "Estados 'dentro' que venden", "Mediana act. tardías (mantener dentro)", "Q medio mantener dentro", "Q medio vender dentro"], filas)))

ie = leer("bloque2_intervencion_epsilon.csv")
filas = []
for r in ie:
    extra = f"{f(r['val_vs_E0_dif'], 1, True)} {ic(r['val_vs_E0_ic_series_lo'], r['val_vs_E0_ic_series_hi'])}" if r.get("val_vs_E0_dif") else "—"
    filas.append([CORTO[r["generador"]], NA[r["algoritmo"]], r["configuracion"], pm(r["val_media_pp"], r["val_sd_semillas_pp"]), f(r["val_ops"], 1),
                  f"{r['semillas_nunca_opera']}/5", f(r["frac_estados_dentro_que_vende"], 2), f(r["Q_medio_mantener_dentro"], 4), f(r["Q_medio_vender_dentro"], 4),
                  f(r["mediana_act_mantener_dentro_total"], 0), extra])
T.append(("T6b", "Intervención sobre el bloqueo (N=5000, α=0.05): E0 con ε decreciente frente a ε=0.1 constante (validación)",
          md(["Generador", "Alg.", "Configuración", "Val. (pp)", "Ops. val", "Nunca operan", "Estados 'dentro' que venden", "Q mantener dentro", "Q vender dentro", "Mediana act. mantener dentro (total)", "Δ val vs E0 [IC series]"], filas)))

b2n = leer("bloque2_alpha_x_N.csv")
filas = []
for g in u.GEN_SENAL_DEBIL:
    for a in A:
        for al in ("0.002", "0.005", "0.05"):
            celdas = []
            for n in ("500", "1000", "2000", "5000", "10000"):
                r = [x for x in b2n if x["generador"] == g and x["algoritmo"] == a and float(x["alpha"]) == float(al) and x["N"] == n][0]
                celdas.append(f"{pm(r['val_media_pp'], r['val_sd_semillas_pp'])} ({r['semillas_nunca_opera']}/5)")
            filas.append([CORTO[g], NA[a], al] + celdas)
T.append(("T7", "α × N (validación, pp, media ± SD; entre paréntesis semillas que nunca operan)",
          md(["Generador", "Alg.", "α", "N=500", "N=1000", "N=2000", "N=5000", "N=10000"], filas)))

b3 = leer("bloque3_episodios.csv")
b3n = leer("bloque3_n_suficiente.csv")
filas = []
for cfgn in ("E0", "F"):
    for g in G:
        for a in A:
            celdas = []
            for n in ("500", "1000", "2000", "5000", "10000"):
                r = [x for x in b3 if x["config"] == cfgn and x["generador"] == g and x["algoritmo"] == a and x["N"] == n]
                celdas.append(pm(r[0]["val_media_pp"], r[0]["val_sd_semillas_pp"]) if r else "—")
            ns = sel(b3n, config=cfgn, generador=g, algoritmo=a)[0]
            filas.append([("E0" if cfgn == "E0" else "E3 final"), CORTO[g], NA[a]] + celdas + [ns["N_suficiente_TODOS"]])
T.append(("T8", "Número de episodios (validación, pp, media ± SD entre semillas) y N 'suficiente' empírico (cumple C1-C4 en N y en todos los mayores; ver informe)",
          md(["Config.", "Generador", "Alg.", "N=500", "N=1000", "N=2000", "N=5000", "N=10000", "N suficiente"], filas)))

b4 = leer("bloque4_cadena.csv")
filas = []
for g in G:
    for a in A:
        celdas = []
        for et in ("E0@2000", "E0@5000", "E1", "E2", "E3"):
            r = sel(b4, generador=g, algoritmo=a, etiqueta=et)[0]
            txt = pm(r["val_media_pp"], r["val_sd_semillas_pp"])
            if r.get("sin_cambio_respecto_a_etapa_previa") == "True":
                txt += " (=)"
            celdas.append(txt)
        filas.append([CORTO[g], NA[a]] + celdas)
T.append(("T9", "Cadena en VALIDACIÓN (pp, media ± SD semillas). '(=)' = la etapa no cambió la configuración respecto a la anterior (corridas idénticas)",
          md(["Generador", "Alg.", "E0 histórica (N=2000)", "E0 (N=5000)", "E1: α", "E2: γ", "E3: ε"], filas)))
filas = []
for g in G:
    for a in A:
        celdas = []
        for et in ("E0@2000", "E0@5000", "E1", "E2", "E3"):
            r = sel(b4, generador=g, algoritmo=a, etiqueta=et)[0]
            celdas.append(f"{f(r['test_media_pp'], 1, True)} {ic(r['test_ic_pob_lo'], r['test_ic_pob_hi'])}")
        filas.append([CORTO[g], NA[a]] + celdas)
T.append(("T9b", "Cadena en TEST (pp, IC95 población; configuraciones ya fijadas)", md(["Generador", "Alg.", "E0 (N=2000)", "E0 (N=5000)", "E1", "E2", "E3"], filas)))
filas = []
for g in G:
    for a in A:
        celdas = []
        for et in ("E1", "E2", "E3"):
            r = sel(b4, generador=g, algoritmo=a, etiqueta=et)[0]
            if r.get("sin_cambio_respecto_a_etapa_previa") == "True":
                celdas.append("sin cambio")
            else:
                celdas.append(f"val {f(r['val_vs_previa_dif'], 1, True)} {ic(r['val_vs_previa_ic_series_lo'], r['val_vs_previa_ic_series_hi'])} / test {f(r['test_vs_previa_dif'], 1, True)} {ic(r['test_vs_previa_ic_series_lo'], r['test_vs_previa_ic_series_hi'])}")
        filas.append([CORTO[g], NA[a]] + celdas)
T.append(("T9c", "Incremento de cada etapa respecto a la anterior (pp; IC95 por series). La validación fundamenta la decisión; el test sólo la evalúa",
          md(["Generador", "Alg.", "E1 − E0(5000)", "E2 − E1", "E3 − E2"], filas)))

e4 = leer("bloque4_e4_discretizacion.csv")
filas = []
for g in G:
    for a in A:
        celdas = []
        for v in ("5 cuantiles", "10 cuantiles", "20 cuantiles", "10 uniformes"):
            r = sel(e4, generador=g, algoritmo=a, variante=v)[0]
            txt = pm(r["val_media_pp"], r["val_sd_semillas_pp"])
            if v != "10 cuantiles":
                txt += f" Δ{f(r['val_vs_10cuantiles_dif'], 1, True)} {ic(r['val_vs_10cuantiles_ic_series_lo'], r['val_vs_10cuantiles_ic_series_hi'])}"
            celdas.append(txt)
        filas.append([CORTO[g], NA[a]] + celdas)
T.append(("T10", "E4: discretización (experimento de sensibilidad posterior, en validación; Δ = diferencia con 10 cuantiles, IC95 por series)",
          md(["Generador", "Alg.", "5 cuantiles", "10 cuantiles", "20 cuantiles", "10 uniformes"], filas)))

b6 = leer("bloque6_qlearning_vs_sarsa.csv")
filas = []
for g in G:
    celdas = []
    for cfgn in ("E0 (N=2000)", "E0 (N=5000)", "E1: α", "E3 final (N=5000)"):
        r = [x for x in b6 if x["generador"] == g and x["configuracion"] == cfgn and x["conjunto"] == "test"][0]
        celdas.append(f"{f(r['dif'], 1, True)} {ic(r['ic_pob_lo'], r['ic_pob_hi'])}; ops {f(r['ops_q'], 0)}/{f(r['ops_s'], 0)}")
    filas.append([CORTO[g]] + celdas)
T.append(("T11", "Q-Learning − SARSA en TEST (pp; IC95 de población) y operaciones (Q/S) por configuración",
          md(["Generador", "E0 (N=2000)", "E0 (N=5000)", "E1", "E3 final"], filas)))
bh = leer("bloque6_vs_buyhold.csv")
filas = []
for g in G:
    for a in A:
        celdas = []
        for cfgn in ("E0 (N=2000)", "E0 (N=5000)", "E3 final (N=5000)"):
            r = [x for x in bh if x["generador"] == g and x["algoritmo"] == a and x["configuracion"] == cfgn][0]
            celdas.append(f"{f(r['test_vs_BuyHold_dif'], 1, True)} {ic(r['test_vs_BuyHold_ic_pob_lo'], r['test_vs_BuyHold_ic_pob_hi'])} ({f(100 * float(r['test_vs_BuyHold_prop_A_gana']), 0)} %)")
        filas.append([CORTO[g], NA[a]] + celdas)
T.append(("T12", "Frente a Buy & Hold en TEST (pp; IC95 población; % de series donde el agente gana)", md(["Generador", "Alg.", "E0 (N=2000)", "E0 (N=5000)", "E3 final (N=5000)"], filas)))

salida = "\n\n".join(f"### {k} — {t}\n\n{tab}" for k, t, tab in T)
(u.RES / "tablas_informe.md").write_text(salida + "\n", encoding="utf-8")
print(f"{len(T)} tablas escritas en resultados/tablas_informe.md")
