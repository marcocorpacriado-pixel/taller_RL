"""
estadistica.py — Fase 2: comparaciones emparejadas entre configuraciones.

Reutiliza las primitivas verificadas de la Fase 1 (comun.resumen_brazo, ic_bootstrap_series, ic_bootstrap_jerarquico,
T4_95, t4_pvalor) y reproduce `comparar` de reanalisar_resultados.py (que es un script y no se puede importar sin
ejecutarlo). No se modifica ningún archivo de la Fase 1.

Convenciones (idénticas a la Fase 1):
  * L es una matriz (semillas S, series J) de log-retornos; las J series son las mismas en todos los brazos.
  * IC "series": bootstrap percentil sobre series (condicional a las S políticas).
  * IC "población": series + semillas, t de Student con 4 g.l., estimador de momentos de efectos cruzados.
  * Los p-valores t (4 g.l.) son aproximados; con Holm sólo dentro de familias declaradas.
"""
import numpy as np

import utilidades as u

c1 = u.c1


def media_ic(L, rng, escala=100.0):
    """Media e IC95 (series, población t4, población bootstrap jerárquico) de una matriz (S, J)."""
    L = np.atleast_2d(np.asarray(L, float))
    r = c1.resumen_brazo(L)
    lo_s, hi_s = c1.ic_bootstrap_series(L.mean(axis=0), rng)
    lo_h, hi_h = c1.ic_bootstrap_jerarquico(L, None, rng)
    h = c1.T4_95 * np.sqrt(r["var_pop"])
    medias_sem = L.mean(axis=1)
    return dict(media=escala * r["media"], sd_semillas=escala * (medias_sem.std(ddof=1) if len(medias_sem) > 1 else 0.0),
                ic_series_lo=escala * lo_s, ic_series_hi=escala * hi_s,
                ic_pob_lo=escala * (r["media"] - h), ic_pob_hi=escala * (r["media"] + h),
                ic_pob_boot_lo=escala * lo_h, ic_pob_boot_hi=escala * hi_h,
                se_series=escala * np.sqrt(r["var_cond"]), se_pob=escala * np.sqrt(r["var_pop"]))


def comparar(LA, LB, rng, escala=100.0):
    """A - B emparejado por serie (copia de la lógica de la Fase 1)."""
    LA, LB = np.atleast_2d(np.asarray(LA, float)), np.atleast_2d(np.asarray(LB, float))
    J = LA.shape[1]
    rA, rB = c1.resumen_brazo(LA), c1.resumen_brazo(LB)
    yA, yB = LA.mean(axis=0), LB.mean(axis=0)
    d_j = yA - yB
    media = yA.mean() - yB.mean()
    var_cond = d_j.var(ddof=1) / J
    cov_j = np.cov(yA, yB, ddof=1)[0, 1] if yA.std() > 0 and yB.std() > 0 else 0.0
    var_pop = max(rA["var_pop"] + rB["var_pop"] - 2 * cov_j / J, var_cond)
    se_pop = np.sqrt(var_pop)
    lo_s, hi_s = c1.ic_bootstrap_series(d_j, rng)
    lo_h, hi_h = c1.ic_bootstrap_jerarquico(LA, LB, rng)
    por_sem_A, por_sem_B = LA.mean(axis=1), LB.mean(axis=1)
    pares = por_sem_A[:, None] - por_sem_B[None, :]
    eps = 1e-12
    t_stat = media / se_pop if se_pop > 0 else np.nan
    return dict(
        dif=escala * media, ic_series_lo=escala * lo_s, ic_series_hi=escala * hi_s,
        ic_pob_lo=escala * (media - c1.T4_95 * se_pop), ic_pob_hi=escala * (media + c1.T4_95 * se_pop),
        ic_pob_boot_lo=escala * lo_h, ic_pob_boot_hi=escala * hi_h,
        prop_A_gana=float((d_j > eps).mean()), prop_empate=float((np.abs(d_j) <= eps).mean()), prop_A_pierde=float((d_j < -eps).mean()),
        sd_semillas_A=escala * (por_sem_A.std(ddof=1) if len(por_sem_A) > 1 else 0.0),
        sd_semillas_B=escala * (por_sem_B.std(ddof=1) if len(por_sem_B) > 1 else 0.0),
        frac_pares_semilla_A_gana=float((pares > eps).mean()),
        se_pob=escala * se_pop, t_stat=t_stat, p_t4=c1.t4_pvalor(t_stat) if np.isfinite(t_stat) else np.nan)


def holm(pvalores):
    """Holm sobre una lista de p-valores (NaN se ignoran). Devuelve lista de p ajustados."""
    p = np.array(pvalores, float)
    ajust = np.full(len(p), np.nan)
    validos = [i for i in range(len(p)) if np.isfinite(p[i])]
    orden = sorted(validos, key=lambda i: p[i])
    m, previo = len(orden), 0.0
    for rango, i in enumerate(orden):
        previo = max(previo, min(1.0, (m - rango) * p[i]))
        ajust[i] = previo
    return list(ajust)


def clasifica(r, holm_p=None):
    """Etiqueta prudente de una comparación (no es prueba de equivalencia cuando el IC incluye 0)."""
    excl = lambda lo, hi: lo > 0 or hi < 0
    ok_pob = excl(r["ic_pob_lo"], r["ic_pob_hi"])
    ok_ser = excl(r["ic_series_lo"], r["ic_series_hi"])
    if r["se_pob"] == 0 and r["dif"] == 0:
        return "idéntica"
    if holm_p is not None and np.isfinite(holm_p) and holm_p < 0.05 and ok_pob and ok_ser and (
            r["frac_pares_semilla_A_gana"] >= 0.8 or r["frac_pares_semilla_A_gana"] <= 0.2):
        return "robusta (Holm)"
    if ok_pob and ok_ser:
        return "indicio (IC excluye 0; sin corrección o sin consistencia entre semillas)"
    if ok_ser:
        return "sólo por series (IC población incluye 0)"
    return "no concluyente (IC incluye 0)"
