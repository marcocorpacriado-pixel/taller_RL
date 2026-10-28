### T1 — E0 (configuración por defecto, N=2000). Validación = media ± SD entre 5 semillas; test con IC95 por series

| Generador | Alg. | Val. (pp) | Test (pp) [IC series] | Ops. val / test | Semillas que nunca operan | Políticas distintas | Acuerdo entre semillas |
|---|---|---|---|---|---|---|---|
| Cíclico | Q-Learning | +429,2 ± 129,2 | +396,1 [+373,5, +418,8] | 118,1 / 117,4 | 0/5 | 5/5 | 0,73 |
| Cíclico | SARSA | +513,1 ± 73,7 | +487,5 [+461,8, +513,5] | 85,2 / 85,7 | 0/5 | 5/5 | 0,79 |
| Browniano | Q-Learning | −1,8 ± 4,0 | −1,7 [−2,2, −1,2] | 17,1 / 17,4 | 4/5 | 2/5 | 0,98 |
| Browniano | SARSA | −1,8 ± 4,0 | −1,7 [−2,2, −1,2] | 17,1 / 17,4 | 4/5 | 2/5 | 0,98 |
| Reversión | Q-Learning | +0,0 ± 0,0 | +0,0 [+0,0, +0,0] | 0,0 / 0,0 | 5/5 | 1/5 | 1,00 |
| Reversión | SARSA | +0,0 ± 0,0 | +0,0 [+0,0, +0,0] | 0,0 / 0,0 | 5/5 | 1/5 | 1,00 |
| Regímenes | Q-Learning | +0,0 ± 0,0 | +0,0 [+0,0, +0,0] | 0,0 / 0,0 | 5/5 | 1/5 | 1,00 |
| Regímenes | SARSA | +0,0 ± 0,0 | +0,0 [+0,0, +0,0] | 0,0 / 0,0 | 5/5 | 1/5 | 1,00 |

### T2 — Inicial frente a optimizada: log-retorno medio en test (pp) con IC95 de población (series + semillas, t 4 g.l.). Efectivo = 0

| Generador | Alg. | E0 (N=2000) | E0 (N=5000) | E3 final (N=5000) |
|---|---|---|---|---|
| Cíclico | Q-Learning | +396,1 [+231,2, +560,9] | +462,9 [+335,3, +590,6] | +610,4 [+566,2, +654,5] |
| Cíclico | SARSA | +487,5 [+388,8, +586,2] | +458,7 [+308,1, +609,3] | +611,5 [+568,0, +655,0] |
| Browniano | Q-Learning | −1,7 [−6,4, +3,0] | +0,0 [+0,0, +0,0] | −2,4 [−7,9, +3,1] |
| Browniano | SARSA | −1,7 [−6,4, +3,0] | +0,0 [+0,0, +0,0] | −0,9 [−3,5, +1,7] |
| Reversión | Q-Learning | +0,0 [+0,0, +0,0] | −2,0 [−7,7, +3,6] | +21,6 [+16,8, +26,5] |
| Reversión | SARSA | +0,0 [+0,0, +0,0] | +0,0 [+0,0, +0,0] | +21,4 [+16,5, +26,4] |
| Regímenes | Q-Learning | +0,0 [+0,0, +0,0] | −1,4 [−5,3, +2,5] | +0,8 [−6,2, +7,8] |
| Regímenes | SARSA | +0,0 [+0,0, +0,0] | +0,0 [+0,0, +0,0] | +2,6 [−1,4, +6,5] |

### T3 — Descomposición en TEST (pp): diferencia [IC95 de población] (% de series donde mejora). Primeras 2 columnas = efecto de los hiperparámetros; siguientes 2 = efecto de la duración; última = lo que mezcla la comparación histórica

| Generador | Alg. | Hiperpar. a N=5000 | Hiperpar. a N=2000 | Duración con E0 | Duración con E3 | TOTAL E3(5000) − E0(2000) |
|---|---|---|---|---|---|---|
| Cíclico | Q-Learning | +147,4 [+22,3, +272,5] (95 %) | +207,6 [+41,5, +373,6] (98 %) | +66,9 [−136,3, +270,0] (96 %) | +6,7 [−27,4, +40,8] (58 %) | +214,3 [+50,2, +378,3] (96 %) |
| Cíclico | SARSA | +152,8 [+3,9, +301,6] (96 %) | +111,5 [+11,0, +212,0] (90 %) | −28,8 [−202,0, +144,5] (21 %) | +12,5 [−24,3, +49,3] (66 %) | +124,0 [+28,9, +219,1] (92 %) |
| Browniano | Q-Learning | −2,4 [−7,9, +3,1] (45 %) | −0,4 [−7,2, +6,5] (52 %) | +1,7 [−3,0, +6,4] (71 %) | −0,3 [−3,8, +3,1] (34 %) | −0,7 [−7,8, +6,4] (51 %) |
| Browniano | SARSA | −0,9 [−3,5, +1,7] (45 %) | +0,6 [−5,0, +6,2] (53 %) | +1,7 [−3,0, +6,4] (71 %) | +0,2 [−2,2, +2,6] (42 %) | +0,8 [−4,5, +6,1] (56 %) |
| Reversión | Q-Learning | +23,7 [+16,2, +31,2] (93 %) | +20,4 [+15,6, +25,1] (89 %) | −2,0 [−7,7, +3,6] (23 %) | +1,3 [−0,6, +3,1] (59 %) | +21,6 [+16,8, +26,5] (91 %) |
| Reversión | SARSA | +21,4 [+16,5, +26,4] (90 %) | +18,0 [+12,3, +23,7] (88 %) | +0,0 [+0,0, +0,0] (0 %) | +3,5 [−0,6, +7,5] (67 %) | +21,4 [+16,5, +26,4] (90 %) |
| Regímenes | Q-Learning | +2,2 [−5,7, +10,2] (51 %) | +2,4 [−1,9, +6,6] (52 %) | −1,4 [−5,3, +2,5] (28 %) | −1,5 [−6,4, +3,3] (41 %) | +0,8 [−6,2, +7,8] (47 %) |
| Regímenes | SARSA | +2,6 [−1,4, +6,5] (55 %) | −0,3 [−6,4, +5,8] (49 %) | +0,0 [+0,0, +0,0] (0 %) | +2,9 [−1,1, +6,9] (57 %) | +2,6 [−1,4, +6,5] (55 %) |

### T3b — Familia declarada de 24 contrastes en test (Holm). Exploratoria: el test ya fue inspeccionado por Marco

| Generador | Alg. | Contraste | Dif. (pp) | IC95 pob. | p (t4) | p Holm | Clasificación |
|---|---|---|---|---|---|---|---|
| Cíclico | Q-Learning | hiperparámetros a N=5000 | +147,4 | [+22,3, +272,5] | 0,031 | 0,462 | indicio (IC excluye 0; sin corrección o sin consistencia entre semillas) |
| Cíclico | Q-Learning | hiperparámetros a N=2000 | +207,6 | [+41,5, +373,6] | 0,026 | 0,409 | indicio (IC excluye 0; sin corrección o sin consistencia entre semillas) |
| Cíclico | Q-Learning | TOTAL histórico | +214,3 | [+50,2, +378,3] | 0,022 | 0,400 | indicio (IC excluye 0; sin corrección o sin consistencia entre semillas) |
| Cíclico | SARSA | hiperparámetros a N=5000 | +152,8 | [+3,9, +301,6] | 0,046 | 0,604 | indicio (IC excluye 0; sin corrección o sin consistencia entre semillas) |
| Cíclico | SARSA | hiperparámetros a N=2000 | +111,5 | [+11,0, +212,0] | 0,037 | 0,517 | indicio (IC excluye 0; sin corrección o sin consistencia entre semillas) |
| Cíclico | SARSA | TOTAL histórico | +124,0 | [+28,9, +219,1] | 0,022 | 0,400 | indicio (IC excluye 0; sin corrección o sin consistencia entre semillas) |
| Browniano | Q-Learning | hiperparámetros a N=5000 | −2,4 | [−7,9, +3,1] | 0,297 | 1,000 | no concluyente (IC incluye 0) |
| Browniano | Q-Learning | hiperparámetros a N=2000 | −0,4 | [−7,2, +6,5] | 0,893 | 1,000 | no concluyente (IC incluye 0) |
| Browniano | Q-Learning | TOTAL histórico | −0,7 | [−7,8, +6,4] | 0,803 | 1,000 | no concluyente (IC incluye 0) |
| Browniano | SARSA | hiperparámetros a N=5000 | −0,9 | [−3,5, +1,7] | 0,374 | 1,000 | no concluyente (IC incluye 0) |
| Browniano | SARSA | hiperparámetros a N=2000 | +0,6 | [−5,0, +6,2] | 0,793 | 1,000 | no concluyente (IC incluye 0) |
| Browniano | SARSA | TOTAL histórico | +0,8 | [−4,5, +6,1] | 0,709 | 1,000 | no concluyente (IC incluye 0) |
| Reversión | Q-Learning | hiperparámetros a N=5000 | +23,7 | [+16,2, +31,2] | 0,001 | 0,019 | robusta (Holm) |
| Reversión | Q-Learning | hiperparámetros a N=2000 | +20,4 | [+15,6, +25,1] | 0,000 | 0,006 | robusta (Holm) |
| Reversión | Q-Learning | TOTAL histórico | +21,6 | [+16,8, +26,5] | 0,000 | 0,006 | robusta (Holm) |
| Reversión | SARSA | hiperparámetros a N=5000 | +21,4 | [+16,5, +26,4] | 0,000 | 0,006 | robusta (Holm) |
| Reversión | SARSA | hiperparámetros a N=2000 | +18,0 | [+12,3, +23,7] | 0,001 | 0,019 | robusta (Holm) |
| Reversión | SARSA | TOTAL histórico | +21,4 | [+16,5, +26,4] | 0,000 | 0,006 | robusta (Holm) |
| Regímenes | Q-Learning | hiperparámetros a N=5000 | +2,2 | [−5,7, +10,2] | 0,477 | 1,000 | no concluyente (IC incluye 0) |
| Regímenes | Q-Learning | hiperparámetros a N=2000 | +2,4 | [−1,9, +6,6] | 0,201 | 1,000 | no concluyente (IC incluye 0) |
| Regímenes | Q-Learning | TOTAL histórico | +0,8 | [−6,2, +7,8] | 0,757 | 1,000 | no concluyente (IC incluye 0) |
| Regímenes | SARSA | hiperparámetros a N=5000 | +2,6 | [−1,4, +6,5] | 0,148 | 1,000 | no concluyente (IC incluye 0) |
| Regímenes | SARSA | hiperparámetros a N=2000 | −0,3 | [−6,4, +5,8] | 0,889 | 1,000 | no concluyente (IC incluye 0) |
| Regímenes | SARSA | TOTAL histórico | +2,6 | [−1,4, +6,5] | 0,148 | 1,000 | no concluyente (IC incluye 0) |

### T4 — Alpha (N=5000, base de Marco). Validación (pp): media ± SD entre semillas (operaciones por serie; semillas que nunca operan en val ni test)

| Generador | Alg. | α=0.001 | α=0.002 | α=0.005 | α=0.01 | α=0.05→0.001 | α=0.05 (E0, nuevo) |
|---|---|---|---|---|---|---|---|
| Cíclico | Q-Learning | +615,6 ± 16,3 (77 op.; 0/5 no opera) | +627,6 ± 13,0 (70 op.; 0/5 no opera) | +635,2 ± 1,6 (63 op.; 0/5 no opera) | +602,8 ± 52,9 (85 op.; 0/5 no opera) | +632,5 ± 4,0 (51 op.; 0/5 no opera) | +490,5 ± 102,8 (87 op.; 0/5 no opera) |
| Cíclico | SARSA | +613,2 ± 16,5 (67 op.; 0/5 no opera) | +624,9 ± 12,3 (71 op.; 0/5 no opera) | +630,8 ± 3,6 (63 op.; 0/5 no opera) | +620,1 ± 22,3 (69 op.; 0/5 no opera) | +630,4 ± 4,3 (62 op.; 0/5 no opera) | +484,8 ± 117,6 (96 op.; 0/5 no opera) |
| Browniano | Q-Learning | +0,0 ± 0,0 (0 op.; 4/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) |
| Browniano | SARSA | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) |
| Reversión | Q-Learning | +19,2 ± 2,4 (62 op.; 0/5 no opera) | +22,3 ± 0,7 (53 op.; 0/5 no opera) | +17,8 ± 4,1 (56 op.; 0/5 no opera) | +11,9 ± 3,2 (46 op.; 0/5 no opera) | +16,9 ± 4,7 (49 op.; 0/5 no opera) | −1,7 ± 3,8 (18 op.; 4/5 no opera) |
| Reversión | SARSA | +21,6 ± 1,2 (50 op.; 0/5 no opera) | +20,6 ± 1,7 (58 op.; 0/5 no opera) | +16,2 ± 2,5 (50 op.; 0/5 no opera) | +12,8 ± 4,0 (53 op.; 0/5 no opera) | +14,8 ± 1,5 (50 op.; 0/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) |
| Regímenes | Q-Learning | +8,0 ± 3,4 (38 op.; 0/5 no opera) | +4,6 ± 4,2 (23 op.; 2/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | −1,6 ± 3,5 (16 op.; 4/5 no opera) |
| Regímenes | SARSA | +3,8 ± 5,3 (16 op.; 3/5 no opera) | +3,4 ± 5,0 (17 op.; 3/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) | +0,0 ± 0,0 (0 op.; 5/5 no opera) |

### T5 — Ruido de Q entre semillas: pendiente log-log de SD(ΔQ) frente a α (teoría 1/2), SD(ΔQ) y cociente señal/ruido en α=0.002 y α=0.05

| Generador | Alg. | Pendiente | SD ΔQ (α=0.002) | SD ΔQ (α=0.05) | Señal/ruido (α=0.002) | Señal/ruido (α=0.05) |
|---|---|---|---|---|---|---|
| Cíclico | Q-Learning | 0,45 | 0,0071 | 0,0445 | 5,26 | 0,91 |
| Cíclico | SARSA | 0,22 | 0,0128 | 0,0375 | 5,77 | 0,91 |
| Browniano | Q-Learning | 0,25 | 0,0005 | 0,0018 | 1,96 | 1,57 |
| Browniano | SARSA | 0,09 | 0,0010 | 0,0020 | 2,19 | 1,58 |
| Reversión | Q-Learning | 0,25 | 0,0007 | 0,0019 | 4,31 | 1,53 |
| Reversión | SARSA | 0,21 | 0,0008 | 0,0020 | 5,38 | 1,64 |
| Regímenes | Q-Learning | 0,23 | 0,0008 | 0,0017 | 2,48 | 1,60 |
| Regímenes | SARSA | 0,22 | 0,0010 | 0,0017 | 2,89 | 1,77 |

### T6 — Visitas (N=5000, grupos instrumentados): estados sin compra en la política final, mediana de actualizaciones de 'comprar' tras ε≈0.01 en esos estados, fracción con < 50 y índice de aprendizaje α × actualizaciones totales de 'comprar'

| Grupo | Generador | Alg. | α | Estados sin compra | Mediana act. tardías (comprar) | Frac. < 50 | α × act. totales | Estados 'dentro' que venden | Mediana act. tardías (mantener dentro) | Q medio mantener dentro | Q medio vender dentro |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0_N5000 | Reversión | Q-Learning | 0.05 | 0,98 | 680 | 0,00 | 763,4 | 1,00 | 6 | −0,0031 | −0,0008 |
| E0_N5000 | Reversión | SARSA | 0.05 | 1,00 | 692 | 0,00 | 763,2 | 1,00 | 5 | −0,0041 | −0,0011 |
| E0_N5000 | Regímenes | Q-Learning | 0.05 | 0,98 | 664 | 0,00 | 714,8 | 0,98 | 4 | −0,0034 | −0,0009 |
| E0_N5000 | Regímenes | SARSA | 0.05 | 1,00 | 660 | 0,00 | 715,0 | 1,00 | 3 | −0,0042 | −0,0012 |
| REP | Reversión | Q-Learning | 0.002 | 0,80 | 339 | 0,00 | 43,2 | 0,26 | 86 | 0,0092 | 0,0067 |
| REP | Reversión | Q-Learning | 0.005 | 0,84 | 621 | 0,00 | 107,9 | 0,42 | 3363 | 0,0081 | 0,0058 |
| REP | Reversión | SARSA | 0.002 | 0,80 | 336 | 0,00 | 43,7 | 0,34 | 84 | 0,0071 | 0,0050 |
| REP | Reversión | SARSA | 0.005 | 0,86 | 469 | 0,00 | 104,5 | 0,48 | 243 | 0,0064 | 0,0050 |
| REP | Regímenes | Q-Learning | 0.002 | 0,92 | 639 | 0,00 | 31,0 | 0,22 | 287 | 0,0026 | −0,0002 |
| REP | Regímenes | Q-Learning | 0.005 | 1,00 | 511 | 0,00 | 61,0 | 0,84 | 12 | −0,0008 | −0,0009 |
| REP | Regímenes | SARSA | 0.002 | 0,96 | 497 | 0,00 | 23,5 | 0,34 | 23 | 0,0001 | −0,0024 |
| REP | Regímenes | SARSA | 0.005 | 1,00 | 653 | 0,00 | 59,4 | 0,84 | 8 | −0,0033 | −0,0013 |

### T6b — Intervención sobre el bloqueo (N=5000, α=0.05): E0 con ε decreciente frente a ε=0.1 constante (validación)

| Generador | Alg. | Configuración | Val. (pp) | Ops. val | Nunca operan | Estados 'dentro' que venden | Q mantener dentro | Q vender dentro | Mediana act. mantener dentro (total) | Δ val vs E0 [IC series] |
|---|---|---|---|---|---|---|---|---|---|---|
| Reversión | Q-Learning | E0 (ε 1→0.01, f=0.6) | −1,7 ± 3,8 | 17,6 | 4/5 | 1,00 | −0,0031 | −0,0008 | 8233 | — |
| Reversión | Q-Learning | E0 con ε=0.1 constante | −8,6 ± 8,5 | 75,1 | 2/5 | 0,96 | −0,0017 | 0,0003 | 2250 | −6,9 [−8,5, −5,2] |
| Reversión | SARSA | E0 (ε 1→0.01, f=0.6) | +0,0 ± 0,0 | 0,0 | 5/5 | 1,00 | −0,0041 | −0,0011 | 8137 | — |
| Reversión | SARSA | E0 con ε=0.1 constante | −8,7 ± 8,7 | 72,8 | 2/5 | 0,90 | −0,0045 | −0,0025 | 2419 | −8,7 [−10,2, −7,3] |
| Regímenes | Q-Learning | E0 (ε 1→0.01, f=0.6) | −1,6 ± 3,5 | 16,2 | 4/5 | 0,98 | −0,0034 | −0,0009 | 7754 | — |
| Regímenes | Q-Learning | E0 con ε=0.1 constante | −3,4 ± 4,7 | 30,2 | 3/5 | 0,90 | −0,0022 | −0,0004 | 1928 | −1,8 [−2,5, −1,2] |
| Regímenes | SARSA | E0 (ε 1→0.01, f=0.6) | +0,0 ± 0,0 | 0,0 | 5/5 | 1,00 | −0,0042 | −0,0012 | 7868 | — |
| Regímenes | SARSA | E0 con ε=0.1 constante | −3,7 ± 5,1 | 31,4 | 3/5 | 0,96 | −0,0057 | −0,0034 | 2183 | −3,7 [−4,5, −2,8] |

### T7 — α × N (validación, pp, media ± SD; entre paréntesis semillas que nunca operan)

| Generador | Alg. | α | N=500 | N=1000 | N=2000 | N=5000 | N=10000 |
|---|---|---|---|---|---|---|---|
| Reversión | Q-Learning | 0.002 | +18,9 ± 2,4 (0/5) | +19,4 ± 2,3 (0/5) | +19,9 ± 2,4 (0/5) | +22,3 ± 0,7 (0/5) | +23,9 ± 1,9 (0/5) |
| Reversión | Q-Learning | 0.005 | +17,7 ± 3,0 (0/5) | +17,6 ± 2,9 (0/5) | +17,1 ± 2,0 (0/5) | +17,8 ± 4,1 (0/5) | +20,2 ± 3,7 (0/5) |
| Reversión | Q-Learning | 0.05 | +0,7 ± 1,7 (4/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | −1,7 ± 3,8 (4/5) | +0,0 ± 0,0 (5/5) |
| Reversión | SARSA | 0.002 | +22,0 ± 1,2 (0/5) | +20,8 ± 1,9 (0/5) | +21,6 ± 1,3 (0/5) | +20,6 ± 1,7 (0/5) | +19,0 ± 2,6 (0/5) |
| Reversión | SARSA | 0.005 | +16,9 ± 3,1 (0/5) | +17,9 ± 1,3 (0/5) | +18,2 ± 3,6 (0/5) | +16,2 ± 2,5 (0/5) | +18,9 ± 5,0 (0/5) |
| Reversión | SARSA | 0.05 | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) |
| Regímenes | Q-Learning | 0.002 | +0,0 ± 0,0 (5/5) | +1,0 ± 2,3 (4/5) | +4,3 ± 3,9 (2/5) | +4,6 ± 4,2 (2/5) | +7,8 ± 3,3 (0/5) |
| Regímenes | Q-Learning | 0.005 | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) |
| Regímenes | Q-Learning | 0.05 | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | −1,6 ± 3,5 (4/5) | +0,0 ± 0,0 (5/5) |
| Regímenes | SARSA | 0.002 | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +3,4 ± 5,0 (3/5) | +0,9 ± 2,1 (4/5) |
| Regímenes | SARSA | 0.005 | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) |
| Regímenes | SARSA | 0.05 | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) | +0,0 ± 0,0 (5/5) |

### T8 — Número de episodios (validación, pp, media ± SD entre semillas) y N 'suficiente' empírico (cumple C1-C4 en N y en todos los mayores; ver informe)

| Config. | Generador | Alg. | N=500 | N=1000 | N=2000 | N=5000 | N=10000 | N suficiente |
|---|---|---|---|---|---|---|---|---|
| E0 | Cíclico | Q-Learning | +399,9 ± 135,7 | +506,4 ± 114,1 | +429,2 ± 129,2 | +490,5 ± 102,8 | — | >5000 |
| E0 | Cíclico | SARSA | +472,6 ± 57,7 | +460,8 ± 140,6 | +513,1 ± 73,7 | +484,8 ± 117,6 | — | >5000 |
| E0 | Browniano | Q-Learning | −1,2 ± 2,7 | +0,0 ± 0,0 | −1,8 ± 4,0 | +0,0 ± 0,0 | — | 5000 |
| E0 | Browniano | SARSA | −1,2 ± 2,7 | +0,0 ± 0,0 | −1,8 ± 4,0 | +0,0 ± 0,0 | — | 5000 |
| E0 | Reversión | Q-Learning | +0,7 ± 1,7 | +0,0 ± 0,0 | +0,0 ± 0,0 | −1,7 ± 3,8 | +0,0 ± 0,0 | 10000 |
| E0 | Reversión | SARSA | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | 500 |
| E0 | Regímenes | Q-Learning | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | −1,6 ± 3,5 | +0,0 ± 0,0 | 10000 |
| E0 | Regímenes | SARSA | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | 500 |
| E3 final | Cíclico | Q-Learning | +618,7 ± 15,7 | +624,4 ± 11,8 | +631,7 ± 28,9 | +635,2 ± 1,6 | — | 2000 |
| E3 final | Cíclico | SARSA | +613,2 ± 15,2 | +626,4 ± 12,9 | +620,7 ± 21,6 | +636,6 ± 4,5 | — | 5000 |
| E3 final | Browniano | Q-Learning | +0,1 ± 0,3 | +0,2 ± 0,5 | +0,5 ± 0,9 | +0,3 ± 0,4 | — | >5000 |
| E3 final | Browniano | SARSA | +0,0 ± 0,0 | +0,2 ± 0,5 | +0,0 ± 0,0 | +0,4 ± 0,9 | — | >5000 |
| E3 final | Reversión | Q-Learning | +18,9 ± 2,4 | +19,4 ± 2,3 | +19,9 ± 2,4 | +22,3 ± 0,7 | +23,9 ± 1,9 | 5000 |
| E3 final | Reversión | SARSA | +19,1 ± 5,6 | +20,9 ± 4,9 | +18,3 ± 3,7 | +22,4 ± 1,5 | +21,7 ± 1,4 | 5000 |
| E3 final | Regímenes | Q-Learning | +0,0 ± 0,0 | +0,0 ± 0,0 | +6,3 ± 4,4 | +8,5 ± 2,4 | +8,9 ± 3,3 | 10000 |
| E3 final | Regímenes | SARSA | +0,0 ± 0,0 | +1,4 ± 3,2 | +4,1 ± 2,8 | +6,9 ± 4,4 | +7,1 ± 3,2 | >10000 |

### T9 — Cadena en VALIDACIÓN (pp, media ± SD semillas). '(=)' = la etapa no cambió la configuración respecto a la anterior (corridas idénticas)

| Generador | Alg. | E0 histórica (N=2000) | E0 (N=5000) | E1: α | E2: γ | E3: ε |
|---|---|---|---|---|---|---|
| Cíclico | Q-Learning | +429,2 ± 129,2 | +490,5 ± 102,8 | +635,2 ± 1,6 | +635,2 ± 1,6 (=) | +635,2 ± 1,6 (=) |
| Cíclico | SARSA | +513,1 ± 73,7 | +484,8 ± 117,6 | +630,8 ± 3,6 | +636,6 ± 4,5 | +636,6 ± 4,5 (=) |
| Browniano | Q-Learning | −1,8 ± 4,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,3 ± 0,3 | +0,3 ± 0,4 |
| Browniano | SARSA | −1,8 ± 4,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,2 ± 0,5 | +0,4 ± 0,9 |
| Reversión | Q-Learning | +0,0 ± 0,0 | −1,7 ± 3,8 | +22,3 ± 0,7 | +22,3 ± 0,7 (=) | +22,3 ± 0,7 (=) |
| Reversión | SARSA | +0,0 ± 0,0 | +0,0 ± 0,0 | +21,6 ± 1,2 | +21,6 ± 1,2 (=) | +22,4 ± 1,5 |
| Regímenes | Q-Learning | +0,0 ± 0,0 | −1,6 ± 3,5 | +8,0 ± 3,4 | +8,5 ± 2,4 | +8,5 ± 2,4 (=) |
| Regímenes | SARSA | +0,0 ± 0,0 | +0,0 ± 0,0 | +3,8 ± 5,3 | +4,7 ± 5,1 | +6,9 ± 4,4 |

### T9b — Cadena en TEST (pp, IC95 población; configuraciones ya fijadas)

| Generador | Alg. | E0 (N=2000) | E0 (N=5000) | E1 | E2 | E3 |
|---|---|---|---|---|---|---|
| Cíclico | Q-Learning | +396,1 [+231,2, +560,9] | +462,9 [+335,3, +590,6] | +610,4 [+566,2, +654,5] | +610,4 [+566,2, +654,5] | +610,4 [+566,2, +654,5] |
| Cíclico | SARSA | +487,5 [+388,8, +586,2] | +458,7 [+308,1, +609,3] | +601,1 [+557,0, +645,2] | +611,5 [+568,0, +655,0] | +611,5 [+568,0, +655,0] |
| Browniano | Q-Learning | −1,7 [−6,4, +3,0] | +0,0 [+0,0, +0,0] | −1,1 [−4,3, +2,0] | −3,5 [−11,2, +4,1] | −2,4 [−7,9, +3,1] |
| Browniano | SARSA | −1,7 [−6,4, +3,0] | +0,0 [+0,0, +0,0] | +0,0 [+0,0, +0,0] | −1,0 [−3,8, +1,8] | −0,9 [−3,5, +1,7] |
| Reversión | Q-Learning | +0,0 [+0,0, +0,0] | −2,0 [−7,7, +3,6] | +21,6 [+16,8, +26,5] | +21,6 [+16,8, +26,5] | +21,6 [+16,8, +26,5] |
| Reversión | SARSA | +0,0 [+0,0, +0,0] | +0,0 [+0,0, +0,0] | +21,4 [+16,8, +26,0] | +21,4 [+16,8, +26,0] | +21,4 [+16,5, +26,4] |
| Regímenes | Q-Learning | +0,0 [+0,0, +0,0] | −1,4 [−5,3, +2,5] | +2,9 [−2,1, +7,8] | +0,8 [−6,2, +7,8] | +0,8 [−6,2, +7,8] |
| Regímenes | SARSA | +0,0 [+0,0, +0,0] | +0,0 [+0,0, +0,0] | +1,2 [−1,2, +3,7] | −0,9 [−5,7, +3,9] | +2,6 [−1,4, +6,5] |

### T9c — Incremento de cada etapa respecto a la anterior (pp; IC95 por series). La validación fundamenta la decisión; el test sólo la evalúa

| Generador | Alg. | E1 − E0(5000) | E2 − E1 | E3 − E2 |
|---|---|---|---|---|
| Cíclico | Q-Learning | val +144,7 [+123,9, +165,7] / test +147,4 [+129,7, +165,0] | sin cambio | sin cambio |
| Cíclico | SARSA | val +146,0 [+121,8, +169,9] / test +142,4 [+123,3, +160,8] | val +5,9 [−2,5, +14,6] / test +10,4 [+4,3, +16,3] | sin cambio |
| Browniano | Q-Learning | val +0,0 [−2,8, +2,7] / test −1,1 [−2,7, +0,5] | val +0,3 [−5,4, +5,6] / test −2,4 [−5,8, +1,1] | val +0,1 [−2,5, +2,8] / test +1,2 [−0,5, +2,9] |
| Browniano | SARSA | val +0,0 [+0,0, +0,0] / test +0,0 [+0,0, +0,0] | val +0,2 [−2,5, +2,8] / test −1,0 [−2,6, +0,7] | val +0,2 [−0,3, +0,7] / test +0,1 [−0,2, +0,4] |
| Reversión | Q-Learning | val +24,0 [+18,4, +29,7] / test +23,7 [+20,1, +27,1] | sin cambio | sin cambio |
| Reversión | SARSA | val +21,6 [+16,4, +27,0] / test +21,4 [+18,1, +24,6] | sin cambio | val +0,8 [−1,2, +2,7] / test +0,0 [−1,3, +1,3] |
| Regímenes | Q-Learning | val +9,6 [+4,2, +15,2] / test +4,3 [+0,9, +7,7] | val +0,5 [−1,3, +2,3] / test −2,0 [−3,3, −0,7] | sin cambio |
| Regímenes | SARSA | val +3,8 [+1,6, +6,1] / test +1,2 [−0,2, +2,6] | val +0,9 [−2,1, +3,8] / test −2,1 [−4,0, −0,2] | val +2,2 [+0,2, +4,3] / test +3,4 [+1,9, +4,9] |

### T10 — E4: discretización (experimento de sensibilidad posterior, en validación; Δ = diferencia con 10 cuantiles, IC95 por series)

| Generador | Alg. | 5 cuantiles | 10 cuantiles | 20 cuantiles | 10 uniformes |
|---|---|---|---|---|---|
| Cíclico | Q-Learning | +613,7 ± 31,2 Δ−21,5 [−37,4, −6,2] | +635,2 ± 1,6 | +624,9 ± 21,9 Δ−10,2 [−19,2, −1,4] | +582,9 ± 28,6 Δ−52,3 [−69,5, −35,7] |
| Cíclico | SARSA | +627,2 ± 1,5 Δ−9,4 [−19,4, +0,3] | +636,6 ± 4,5 | +637,6 ± 7,7 Δ+1,0 [−4,5, +6,5] | +593,4 ± 23,4 Δ−43,2 [−57,4, −28,7] |
| Browniano | Q-Learning | −8,6 ± 15,8 Δ−8,9 [−10,9, −7,0] | +0,3 ± 0,4 | +0,2 ± 0,9 Δ−0,1 [−0,8, +0,6] | +0,3 ± 0,4 Δ−0,1 [−0,4, +0,2] |
| Browniano | SARSA | +0,2 ± 0,4 Δ−0,2 [−3,3, +2,7] | +0,4 ± 0,9 | +0,4 ± 0,9 Δ+0,0 [+0,0, +0,0] | +0,2 ± 0,4 Δ−0,2 [−0,8, +0,2] |
| Reversión | Q-Learning | +22,2 ± 1,9 Δ−0,1 [−1,5, +1,4] | +22,3 ± 0,7 | +20,7 ± 1,4 Δ−1,6 [−3,2, +0,1] | +9,1 ± 0,0 Δ−13,2 [−17,6, −8,6] |
| Reversión | SARSA | +20,5 ± 3,6 Δ−1,9 [−3,4, −0,5] | +22,4 ± 1,5 | +21,7 ± 3,0 Δ−0,8 [−2,4, +0,9] | +7,5 ± 3,6 Δ−14,9 [−19,2, −10,3] |
| Regímenes | Q-Learning | +0,3 ± 0,6 Δ−8,2 [−13,1, −3,7] | +8,5 ± 2,4 | +8,6 ± 2,6 Δ+0,1 [−2,8, +3,0] | +0,0 ± 0,0 Δ−8,5 [−14,4, −2,9] |
| Regímenes | SARSA | +0,9 ± 1,4 Δ−6,0 [−9,2, −2,8] | +6,9 ± 4,4 | +9,2 ± 1,7 Δ+2,3 [−0,5, +5,0] | +0,0 ± 0,0 Δ−6,9 [−12,0, −2,2] |

### T11 — Q-Learning − SARSA en TEST (pp; IC95 de población) y operaciones (Q/S) por configuración

| Generador | E0 (N=2000) | E0 (N=5000) | E1 | E3 final |
|---|---|---|---|---|
| Cíclico | −91,4 [−277,6, +94,8]; ops 117/86 | +4,2 [−187,2, +195,7]; ops 87/96 | +9,3 [−2,6, +21,1]; ops 62/63 | −1,1 [−13,3, +11,0]; ops 62/53 |
| Browniano | +0,0 [−6,6, +6,6]; ops 17/17 | +0,0 [+0,0, +0,0]; ops 0/0 | −1,1 [−4,3, +2,0]; ops 0/0 | −1,4 [−5,3, +2,4]; ops 0/0 |
| Reversión | +0,0 [+0,0, +0,0]; ops 0/0 | −2,0 [−7,7, +3,6]; ops 17/0 | +0,3 [−0,6, +1,1]; ops 52/50 | +0,2 [−1,8, +2,2]; ops 52/45 |
| Regímenes | +0,0 [+0,0, +0,0]; ops 0/0 | −1,4 [−5,3, +2,5]; ops 16/0 | +1,7 [−2,0, +5,3]; ops 39/16 | −1,7 [−6,7, +3,3]; ops 33/28 |

### T12 — Frente a Buy & Hold en TEST (pp; IC95 población; % de series donde el agente gana)

| Generador | Alg. | E0 (N=2000) | E0 (N=5000) | E3 final (N=5000) |
|---|---|---|---|---|
| Cíclico | Q-Learning | +58,6 [−110,2, +227,4] (62 %) | +125,5 [−5,9, +256,8] (78 %) | +272,9 [+230,3, +315,4] (96 %) |
| Cíclico | SARSA | +150,0 [+46,6, +253,4] (80 %) | +121,2 [−32,9, +275,3] (77 %) | +274,0 [+233,5, +314,5] (98 %) |
| Browniano | Q-Learning | +4,5 [−8,2, +17,1] (51 %) | +6,2 [−5,8, +18,2] (55 %) | +3,8 [−4,0, +11,6] (55 %) |
| Browniano | SARSA | +4,5 [−8,2, +17,1] (51 %) | +6,2 [−5,8, +18,2] (55 %) | +5,2 [−4,4, +14,9] (55 %) |
| Reversión | Q-Learning | +1,1 [−1,0, +3,1] (56 %) | −0,9 [−6,9, +5,0] (42 %) | +22,7 [+17,3, +28,1] (89 %) |
| Reversión | SARSA | +1,1 [−1,0, +3,1] (56 %) | +1,1 [−1,0, +3,1] (56 %) | +22,5 [+17,2, +27,8] (90 %) |
| Regímenes | Q-Learning | +21,9 [+8,6, +35,2] (68 %) | +20,5 [+6,8, +34,1] (67 %) | +22,7 [+11,5, +34,0] (72 %) |
| Regímenes | SARSA | +21,9 [+8,6, +35,2] (68 %) | +21,9 [+8,6, +35,2] (68 %) | +24,5 [+13,1, +35,8] (74 %) |
