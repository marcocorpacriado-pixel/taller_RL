"""
analizar_politicas.py — Fase 1, parte C: estabilidad de las políticas finales a partir de las tablas Q guardadas.

Uso:  py -3.14 analisis_adicional/analizar_politicas.py

Para cada (generador, algoritmo) toma las 5 políticas finales (ganadores de la etapa epsilon), las reconstruye
(acción greedy sobre acciones válidas) y compara:
  * estado a estado (20 estados = 10 bins x 2 posiciones): coincidencia, nº de políticas distintas, desacuerdos;
  * ponderando por las visitas reales de cada estado en las series de test (equivalencia de comportamiento);
  * en rendimiento: ¿políticas distintas dan resultados distintos?
Distingue estabilidad de la política (tabla de decisiones) de estabilidad del rendimiento (log-retorno en test).
"""
import itertools
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import comun as c

c.asegurar_carpetas()
etapas = {e: c.cargar_pickle(e) for e in ("alpha", "gamma", "epsilon", "discretizacion")}
fin = c.ganadores(etapas["epsilon"])
test = {g: c.generar_series(g, c.N_TEST, c.SEED_TEST) for g in c.GENERADORES}
NB = 10


def rangos(x):
    return np.argsort(np.argsort(x)).astype(float)


def corr_rangos(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rangos(np.asarray(x)), rangos(np.asarray(y)))[0, 1])


POL = {}   # (g, a) -> dict(vec (5,20) bool, visitas (5,2,10), logret (5,100), empates)
for g in c.GENERADORES:
    for a in c.ALGORITMOS:
        vecs, vis, lr, emp = [], [], [], 0
        for r in fin[(g, a)]:
            comprar, vender, e0, e1 = c.politica_desde_Q(r["Q"])
            emp += int(e0.sum() + e1.sum())
            vecs.append(np.concatenate([comprar, vender]))        # 0..9: ¿compra? (fuera) | 10..19: ¿vende? (dentro)
            visitas = [[0] * NB, [0] * NB]
            rec, bal, ops = c.evaluar_tabla(r["Q"], r["cortes"], test[g], visitas)
            V = np.array(visitas, float)
            vis.append(V / V.sum())
            lr.append(np.log(bal / 1000.0))
            assert np.allclose(lr[-1], r["test"]["logret"], atol=1e-12, rtol=0)  # reproduce lo guardado
        POL[(g, a)] = dict(vec=np.array(vecs), vis=np.array(vis), logret=np.array(lr), empates=emp)
print("Políticas reconstruidas y verificadas contra los log-retornos de test guardados (40/40).")

ESTADOS = [f"fuera/bin{b}:C" for b in range(NB)] + [f"dentro/bin{b}:V" for b in range(NB)]
resumen, pares_f, estados_f, detalle = [], [], [], []
for g in c.GENERADORES:
    for a in c.ALGORITMOS:
        P = POL[(g, a)]
        V, VIS, LR = P["vec"], P["vis"], P["logret"]
        distintas = {tuple(v) for v in V}
        acuerdos, acuerdos_w, difs, hams = [], [], [], []
        for i, j in itertools.combinations(range(5), 2):
            coinc = (V[i] == V[j])
            w = (VIS[i] + VIS[j]).reshape(-1) / 2.0   # visitas medias de ambas políticas, orden (pos0 bins, pos1 bins)
            ac, acw = coinc.mean(), float((w * coinc).sum() / w.sum())
            d = LR[i] - LR[j]
            acuerdos.append(ac)
            acuerdos_w.append(acw)
            hams.append(int((~coinc).sum()))
            difs.append(abs(LR[i].mean() - LR[j].mean()) * 100)
            pares_f.append(dict(generador=g, algoritmo=a, semilla_i=i, semilla_j=j, estados_distintos=int((~coinc).sum()),
                                acuerdo_estados=ac, acuerdo_ponderado_visitas=acw,
                                dif_abs_logret_medio_pp=abs(LR[i].mean() - LR[j].mean()) * 100,
                                frac_series_resultado_identico=float((np.abs(d) < 1e-12).mean()),
                                sd_dif_por_serie_pp=float(d.std(ddof=1) * 100)))
        modal = V.mean(axis=0)
        desacuerdo = np.minimum(modal, 1 - modal)
        for k, nombre in enumerate(ESTADOS):
            pos, b = divmod(k, NB)
            estados_f.append(dict(generador=g, algoritmo=a, estado=nombre, frac_semillas_opera=float(modal[k]),
                                  visitas_medias_pct=float(100 * VIS[:, pos, b].mean()), desacuerdo=float(desacuerdo[k])))
        top = np.argsort(-desacuerdo)[:3]
        sem_logret = LR.mean(axis=1) * 100
        resumen.append(dict(
            generador=g, algoritmo=a, politicas_distintas=len(distintas), de=5,
            acuerdo_medio_estados=float(np.mean(acuerdos)), acuerdo_medio_ponderado_visitas=float(np.mean(acuerdos_w)),
            estados_distintos_medio=float(np.mean(hams)),
            estados_con_desacuerdo=int((desacuerdo > 0).sum()),
            top_desacuerdo=" ; ".join(f"{ESTADOS[k]} ({modal[k]:.1f})" for k in top if desacuerdo[k] > 0),
            valores_distintos_logret_test=len({round(float(x), 6) for x in sem_logret}),
            sd_semillas_logret_pp=float(sem_logret.std(ddof=1)), min_logret_pp=float(sem_logret.min()), max_logret_pp=float(sem_logret.max()),
            corr_rangos_distancia_vs_dif_rendimiento=corr_rangos(hams, difs),
            empates_exactos_Q_en_acciones_validas=P["empates"]))
        for s in range(5):
            detalle.append(dict(generador=g, algoritmo=a, semilla=s,
                                fuera_compra="".join("C" if x else "." for x in V[s, :NB]),
                                dentro_venta="".join("V" if x else "." for x in V[s, NB:]),
                                logret_test_pp=float(LR[s].mean() * 100), operaciones_test=float(np.nan)))

# Acuerdo entre algoritmos (25 pares semilla QL x semilla SARSA)
entre = []
for g in c.GENERADORES:
    Vq, Vs = POL[(g, "qlearning")]["vec"], POL[(g, "sarsa")]["vec"]
    ac = [(Vq[i] == Vs[j]).mean() for i in range(5) for j in range(5)]
    dentro = [r["acuerdo_medio_estados"] for r in resumen if r["generador"] == g]
    entre.append(dict(generador=g, acuerdo_entre_algoritmos=float(np.mean(ac)), acuerdo_dentro_QL=dentro[0], acuerdo_dentro_SARSA=dentro[1]))

# ¿Políticas distintas con rendimiento similar? (global, sobre todos los pares con alguna diferencia)
pd_ = [p for p in pares_f if p["estados_distintos"] > 0]
mismo = [p for p in pd_ if p["dif_abs_logret_medio_pp"] < 0.5]
c.escribir_csv("estabilidad_politicas.csv", resumen)
c.escribir_csv("estabilidad_politicas_pares.csv", pares_f)
c.escribir_csv("estabilidad_politicas_estados.csv", estados_f)
c.escribir_csv("estabilidad_politicas_detalle.csv", detalle, ["generador", "algoritmo", "semilla", "fuera_compra", "dentro_venta", "logret_test_pp"])
c.escribir_csv("estabilidad_politicas_entre_algoritmos.csv", entre)

# Figura: mapa de acciones por política
fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.9))
for ax, g in zip(axs, c.GENERADORES):
    M = np.vstack([POL[(g, a)]["vec"] for a in c.ALGORITMOS]).astype(float)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1.3, aspect="auto")
    ax.axvline(NB - 0.5, color="black", lw=1.2)
    ax.axhline(4.5, color="black", lw=1.2)
    for (i, k), v in np.ndenumerate(M):
        if v:
            ax.text(k, i, "C" if k < NB else "V", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(2 * NB))
    ax.set_xticklabels([str(k % NB) for k in range(2 * NB)], fontsize=6)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"{'QL' if i < 5 else 'SA'} s{i % 5}" for i in range(10)], fontsize=7)
    ax.set_xlabel("bin de tendencia: sin posición (¿compra?) | con posición (¿vende?)", fontsize=6.5)
    ax.set_title(c.ETIQUETA_GEN[g], fontweight="bold", fontsize=9)
    ax.grid(False)
fig.suptitle("Políticas finales por semilla (azul = opera en ese estado). Q-Learning (arriba) y SARSA (abajo)", y=1.02, fontsize=10)
fig.savefig(c.FIG / "fig6_politicas_por_semilla.png", dpi=170, bbox_inches="tight")
plt.close(fig)

print("\n=== Resumen de estabilidad ===")
for r in resumen:
    print(f"{r['generador']:9s} {r['algoritmo']:9s} distintas={r['politicas_distintas']}/5  acuerdo={r['acuerdo_medio_estados']:.2f}  pond.visitas={r['acuerdo_medio_ponderado_visitas']:.2f}  "
          f"estados_en_desacuerdo={r['estados_con_desacuerdo']}  valores_distintos_logret={r['valores_distintos_logret_test']}  sd_sem={r['sd_semillas_logret_pp']:.2f}  "
          f"rango=[{r['min_logret_pp']:.2f},{r['max_logret_pp']:.2f}]  corr_rangos(dist,Δrend)={r['corr_rangos_distancia_vs_dif_rendimiento']:.2f}  empates={r['empates_exactos_Q_en_acciones_validas']}  top: {r['top_desacuerdo']}")
print("\nAcuerdo entre algoritmos:", [(e['generador'], round(e['acuerdo_entre_algoritmos'], 3), round(e['acuerdo_dentro_QL'], 3), round(e['acuerdo_dentro_SARSA'], 3)) for e in entre])
print(f"\nPares de políticas con alguna diferencia: {len(pd_)}; de ellos con |Δ rendimiento medio| < 0.5 pp: {len(mismo)}; "
      f"con resultado idéntico en el 100 % de las series: {sum(p['frac_series_resultado_identico'] == 1.0 for p in pd_)}")
sys.exit(0)
