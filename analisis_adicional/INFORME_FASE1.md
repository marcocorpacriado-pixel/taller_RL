# Informe Fase 1 — Reanálisis estadístico de los artefactos del Taller RL

Registro independiente de investigación. Aportación adicional al trabajo de Marco (rama `feature/analisis-estadistico-rl`, partiendo de `5a6e1ee`). No se ha modificado ningún archivo original ni se ha entrenado nada: todo sale de los `.pkl` existentes y de las series de test regeneradas con las semillas originales.

---

## 1. Objetivo e hipótesis

**Objetivo.** Aprovechar los artefactos existentes para (a) separar la incertidumbre debida a las series de precios de la debida a las semillas de entrenamiento, (b) comparar agentes y baselines por serie, incluido un nuevo baseline de efectivo, (c) contrastar recompensa del entorno con rendimiento financiero, (d) reanalizar la convergencia con un criterio reproducible y (e) medir la estabilidad de las políticas.

**Hipótesis de partida (formuladas en la auditoría, antes de calcular).**

| # | Hipótesis | Qué la refutaría |
|---|---|---|
| H1 | Los "± entre semillas" de la tabla original subestiman la incertidumbre real | IC por series y de población similares a ese ± |
| H2 | Q-Learning y SARSA no son distinguibles en rendimiento | Algún IC de la diferencia que excluya 0 |
| H3 | En browniano y regímenes el RL no supera a mantener efectivo | IC de (agente − efectivo) que excluya 0 por arriba |
| H4 | La suma de recompensas del entorno diverge del log-retorno, hasta cambiar de signo en el browniano | Diferencia sin cambio de signo |
| H5 | La comparación de convergencia depende del umbral y de los hiperparámetros | Orden QL/SARSA estable al variar umbral y configuración |
| H6 | Las políticas de distintas semillas son pocas y casi iguales donde hay señal | Muchas políticas distintas con bajo acuerdo |

---

## 2. Artefactos reutilizados

| Artefacto | Uso | Seguridad |
|---|---|---|
| `resultados/alpha.pkl`, `gamma.pkl`, `epsilon.pkl`, `discretizacion.pkl` (600 entrenamientos: Q, cortes, cfg, curvas, val/test) | Reconstrucción de políticas, curvas, selección | Opcodes inspeccionados sin ejecutar; deserialización con lista blanca (`numpy.dtype`, `numpy._core.numeric._frombuffer`); sin `INST/OBJ/GLOBAL` |
| `resultados/baselines.pkl` | Verificación de aleatorio y Buy & Hold | Ídem |
| `resultados/log_ajuste.txt`, `figuras/tabla_resultados.md` | Verificar que se reproduce la tabla original | Texto |
| `taller_trader.ipynb` (celda 4) | Verificar la réplica del entorno | Texto |
| `senal.pkl` | No usado en esta fase | — |

Qué se pudo hacer directamente y qué requirió reconstruir:

* **Directo:** curvas (`hist`), métricas por semilla de val/test, configuraciones.
* **Reconstruido:** la recompensa acumulada del entorno (los `.pkl` no la guardan), el número de visitas por estado y las observaciones por serie. Para ello se reconstruyó la política greedy desde `Q` y `cortes` y se jugó con una réplica del entorno.

**No se asumió que la reconstrucción reproduce los resultados: se verificó** (§7).

---

## 3. Metodología estadística

**Unidad y estructura.** Para cada generador y agente: matriz `L[semilla, serie]` de log-retornos en test (5 × 100; las 100 series son las mismas, `seed 7777`, para todos los agentes). Las observaciones no son independientes: las 5 políticas se evalúan sobre las mismas 100 series. No se tratan 500 observaciones como independientes.

**Dos preguntas distintas, dos intervalos.**

1. **IC por series (condicional a estas 5 políticas).** Bootstrap percentil (10 000 réplicas) remuestreando series con reemplazo sobre la media por serie de las 5 semillas. Responde: *¿cuánto cambiaría el resultado de estas cinco políticas con otras 100 series?*
2. **IC de población (series + semillas).** Estimador de momentos de un modelo de efectos aleatorios cruzados semilla × serie sin réplica: `Var(media) = (MS_semilla + MS_serie − MS_residual)/(S·J)`, truncado por debajo en la varianza condicional; intervalo `media ± t(0,975; 4 g.l.)·√Var`. Responde: *¿cuánto cambiaría si se entrenase con otras semillas y se evaluase en otras series?* Con S = 5 los grados de libertad (4) son pocos, así que este intervalo es aproximado.
3. **Contraste:** bootstrap jerárquico (remuestreo de semillas por brazo y, después, de series compartidas entre brazos). Con 5 semillas queda sistemáticamente más estrecho que el t de 4 g.l.; se incluye como columna de control (`ic95_pob_boot_*`), no como resultado principal.

**Comparaciones emparejadas.** Para A − B: `d_j = media_semillas(A_j) − media_semillas(B_j)` por serie. La varianza de población es `Var_A + Var_B − 2·Cov_series/J` (series comunes; semillas de A y B tratadas como independientes, lo que es conservador porque comparten la secuencia de entrenamiento). Se reportan: diferencia en pp, IC95 por series y de población, proporción de series en que A supera / empata / pierde frente a B, SD entre semillas de cada brazo, fracción de pares de semillas con A > B y operaciones medias.

**Múltiples comparaciones.** Familia confirmatoria definida a priori: 7 comparaciones × 4 generadores = **28**. Se aplica Holm sobre los p-valores t (4 g.l.) y se da el IC con corrección de Bonferroni. Una comparación se llama **robusta** solo si cumple a la vez: p de Holm < 0,05, IC de población que excluye 0, IC por series que excluye 0 y al menos 4 de 5 semillas en el mismo sentido (aquí, ≥ 80 % de pares de semillas). El resto son **indicios exploratorios** o **no concluyentes**. Un IC que incluye 0 **no** se interpreta como equivalencia.

**Supuestos.** (i) Series de test independientes y representativas del generador; (ii) las 5 semillas son una muestra aleatoria de la población de entrenamientos; (iii) la media por serie es aproximadamente simétrica (en el cíclico el log-retorno es razonable, no así el patrimonio); (iv) los efectos semilla y serie son aditivos (la interacción se agrupa en el residual). La reproducibilidad del bootstrap está garantizada con `np.random.default_rng([20261028, etiqueta, i, k])`.

**Convergencia.** Ver §5.

---

## 4. Nuevas métricas, tablas y figuras

### 4.1 Rendimiento en test con incertidumbre separada

Log-retorno medio en test, pp. "SD sem." = desviación entre las 5 medias por semilla (ddof = 1; la tabla original usaba ddof = 0). Fuente: `resultados/metricas_por_agente.csv`.

| Generador | Agente | Media | SD sem. | IC95 series | IC95 series + semillas (t, 4 g.l.) | Ops. medias |
|---|---|---|---|---|---|---|
| Cíclico | Efectivo | 0,0 | 0,0 | [0, 0] | [0, 0] | 0 |
| | Aleatorio | +145,2 | 11,7 | [+121,9, +168,4] | [+112,7, +177,8] | 165 |
| | Buy & Hold | +337,5 | — | [+294,6, +379,7] | [+277,1, +397,9] | 1 |
| | Q-Learning | +610,4 | 7,1 | [+579,7, +641,7] | [+566,2, +654,5] | 61,6 |
| | SARSA | +611,5 | 7,2 | [+581,1, +641,7] | [+568,0, +655,0] | 53,1 |
| Browniano | Aleatorio | −20,8 | 1,8 | [−25,2, −16,4] | [−27,1, −14,4] | 165 |
| | Buy & Hold | −6,2 | — | [−14,7, +2,3] | [−18,2, +5,8] | 1 |
| | Q-Learning | −2,4 | 3,3 | [−5,7, +1,0] | [−7,9, +3,1] | 0,4 |
| | SARSA | −0,9 | 2,1 | [−2,6, +0,7] | [−3,5, +1,7] | 0,2 |
| Reversión | Aleatorio | −15,7 | 0,7 | [−17,9, −13,5] | [−18,9, −12,6] | 165 |
| | Buy & Hold | −1,1 | — | [−2,5, +0,4] | [−3,2, +1,0] | 1 |
| | Q-Learning | +21,6 | 0,04 | [+18,3, +25,2] | [+16,8, +26,5] | 52,2 |
| | SARSA | +21,4 | 0,6 | [+18,0, +24,9] | [+16,5, +26,4] | 44,9 |
| Regímenes | Aleatorio | −28,3 | 1,7 | [−33,4, −23,4] | [−35,4, −21,2] | 165 |
| | Buy & Hold | −21,9 | — | [−31,1, −12,7] | [−35,2, −8,6] | 1 |
| | Q-Learning | +0,8 | 3,5 | [−3,1, +4,8] | [−6,2, +7,8] | 33,1 |
| | SARSA | +2,6 | 1,5 | [−0,1, +5,2] | [−1,4, +6,5] | 28,3 |

**Lectura de la separación de fuentes.** La media se calcula sobre 100 series, de modo que casi toda la incertidumbre viene de las series. Las semillas aportan una fracción apreciable solo en el browniano (≈ 26 % de la varianza para Q-Learning, ≈ 14 % para SARSA) y en regímenes con Q-Learning (≈ 34 %); en reversión y cíclico es despreciable. El "+21,6 ± 0,0" de reversión escondía un intervalo real de ≈ ±5 pp: la desviación entre semillas (0,04 pp) mide cuántas políticas distintas aparecieron (2 de 5), no la incertidumbre sobre el rendimiento.

Figura: [`fig1_rendimientos_ic.png`](figuras/fig1_rendimientos_ic.png).

### 4.2 Comparaciones emparejadas por serie

Diferencia A − B en pp de log-retorno. Entre corchetes, IC95 de población (t, 4 g.l.); entre paréntesis, % de series en que A supera a B. Fuente: `resultados/comparaciones_pareadas.csv` (incluye IC por series y bootstrap jerárquico, Bonferroni, p-valores y operaciones).

| Generador | Q − SARSA | Q − B&H | SARSA − B&H | Q − Aleat. | SARSA − Aleat. | Q − Efectivo | SARSA − Efectivo |
|---|---|---|---|---|---|---|---|
| Cíclico | −1,1 [−13,3, +11,0] (51 %) | +272,9 [+230,4, +315,4] (96 %) | +274,0 [+233,5, +314,5] (98 %) | +465,2 [+427,7, +502,6] (100 %) | +466,3 [+430,5, +502,1] (100 %) | +610,4 [+566,2, +654,5] (100 %) | +611,5 [+568,0, +655,0] (100 %) |
| Browniano | −1,4 [−5,3, +2,4] (42 %) | +3,8 [−4,0, +11,6] (55 %) | +5,2 [−4,4, +14,9] (55 %) | +18,4 [+14,3, +22,5] (96 %) | +19,8 [+15,3, +24,3] (90 %) | −2,4 [−7,9, +3,1] (45 %) | −0,9 [−3,5, +1,7] (45 %) |
| Reversión | +0,2 [−1,8, +2,2] (55 %) | +22,7 [+17,4, +28,1] (89 %) | +22,5 [+17,2, +27,8] (90 %) | +37,3 [+31,5, +43,2] (96 %) | +37,1 [+31,1, +43,1] (95 %) | +21,6 [+16,8, +26,5] (91 %) | +21,4 [+16,5, +26,4] (90 %) |
| Regímenes | −1,7 [−6,7, +3,3] (40 %) | +22,7 [+11,5, +34,0] (72 %) | +24,5 [+13,1, +35,8] (74 %) | +29,1 [+22,3, +36,0] (93 %) | +30,9 [+25,2, +36,5] (93 %) | +0,8 [−6,2, +7,8] (47 %) | +2,6 [−1,4, +6,5] (55 %) |

**Clasificación (familia de 28; Holm).**

* **17 robustas:** todo frente a aleatorio (8: dos algoritmos × 4 generadores), cíclico frente a B&H y efectivo (4), reversión frente a B&H y efectivo (4) y regímenes SARSA − B&H (1).
* **1 indicio exploratorio:** regímenes Q-Learning − B&H (p = 0,0049; Holm 0,054, por muy poco). Regímenes SARSA − B&H es robusta con Holm 0,0468, es decir, **en el límite**: depende de la corrección elegida.
* **10 no concluyentes** (IC de población incluye 0): las 4 de Q − SARSA, todas las de browniano excepto frente a aleatorio (4: vs B&H ×2 y vs efectivo ×2) y las dos de regímenes frente a efectivo.

**Contexto (fuera de la familia).** Buy & Hold − Efectivo: cíclico +337,5 [+277,1, +397,9]; browniano −6,2 [−18,2, +5,8]; reversión −1,1 [−3,2, +1,0]; regímenes −21,9 [−35,2, −8,6]. Aleatorio − Efectivo es negativo y significativo en los cuatro.

**Operaciones, Q-Learning − SARSA** (descriptivo): cíclico +8,5 [−5,5, +22,5], browniano +0,2 [−0,7, +1,1], reversión +7,3 [−1,6, +16,2], regímenes +4,8 [−19,0, +28,6]. Q-Learning opera más en los cuatro, pero ningún IC de población excluye 0.

Figuras: [`fig2_qlearning_vs_sarsa.png`](figuras/fig2_qlearning_vs_sarsa.png), [`fig3_vs_efectivo_y_buyhold.png`](figuras/fig3_vs_efectivo_y_buyhold.png).

### 4.3 Baseline de efectivo

Implementado externamente (acción "mantener" en todos los pasos con la réplica del entorno) y verificado en las 400 series de test: balance final 1 000 $, 0 operaciones, recompensa 0, log-retorno y rentabilidad simple 0 %. No se toca el entorno original.

* **Browniano — ¿aporta valor el RL frente a no operar?** No hay evidencia: Q-Learning −2,4 [−7,9, +3,1] y SARSA −0,9 [−3,5, +1,7] frente a efectivo. La política "no opera" de la bitácora es en realidad una mezcla (ver §4.5): 3 de 5 semillas de Q-Learning y 4 de 5 de SARSA nunca compran; las otras compran una vez y mantienen (≈ Buy & Hold).
* **Regímenes — ¿rentabilidad positiva defendible?** No para Q-Learning (+0,8 [−6,2, +7,8]) y, para SARSA, solo como indicio débil (+2,6; IC por series [−0,1, +5,2], de población [−1,4, +6,5]; p = 0,15 sin ajustar). El agente **evita parte de las pérdidas de Buy & Hold** (diferencia de ≈ +23 a +24 pp), que es otra cosa que generar una estrategia rentable. Que Buy & Hold pierda −21,9 es una característica del generador (deriva con pérdidas en el régimen bajista); superarlo en un escenario bajista no es ganancia absoluta.
* **Reversión y cíclico:** el agente supera a efectivo de forma robusta (91 % y 90 % de las series en reversión; 100 % en cíclico).

### 4.4 Recompensa RL frente a rendimiento financiero

Medias en test (pp). Σ R = suma de recompensas del entorno por episodio; Brecha = Σ R − log-retorno. Fuente: `resultados/recompensa_vs_financiero.csv` (incluye IC por series, retorno simple del patrimonio, patrimonio medio y mediano, correlación entre series).

| Generador | Agente | Σ R (IC series) | Log-retorno | Ret. simple | Brecha | Mediana patrimonio ($) |
|---|---|---|---|---|---|---|
| Cíclico | Buy & Hold | +612,9 [+571,8, +656,0] | +337,5 | +28 031 | +275,4 | 32 600 |
| | Q-Learning | +784,8 [+753,2, +817,0] | +610,4 | +157 707 | +174,5 | 432 288 |
| | SARSA | +790,4 [+758,9, +821,8] | +611,5 | +158 140 | +178,9 | 429 966 |
| Browniano | Buy & Hold | **+3,6 [−4,6, +11,9]** | **−6,2** | +3,1 | +9,7 | 958 |
| | Q-Learning | +1,5 [−1,8, +4,8] | −2,4 | +1,3 | +3,9 | 1 000 |
| | SARSA | +0,9 [−0,8, +2,6] | −0,9 | +1,0 | +1,9 | 1 000 |
| Reversión | Buy & Hold | +9,0 [+7,6, +10,5] | −1,1 | −0,8 | +10,1 | 991 |
| | Q-Learning | +26,4 [+23,0, +29,7] | +21,6 | +26,2 | +4,7 | 1 246 |
| | SARSA | +26,9 [+23,4, +30,4] | +21,4 | +26,1 | +5,5 | 1 242 |
| Regímenes | Buy & Hold | −12,8 [−22,0, −3,8] | −21,9 | −10,2 | +9,1 | 826 |
| | Q-Learning | +3,7 [−0,3, +7,8] | +0,8 | +3,8 | +2,9 | 1 008 |
| | SARSA | **+4,6 [+1,9, +7,3]** | **+2,6 [−0,1, +5,2]** | +4,3 | +2,0 | 1 000 |

**Verificado.**

* Σ R y log-retorno están muy correlacionados entre series (≥ 0,995 en todos los casos con variación), así que ordenan bien las series dentro de un mismo agente. Pero **difieren en nivel**: Σ R sobreestima sistemáticamente, y la brecha crece con la volatilidad del generador (cíclico, σ = 10 % por paso: +175; el resto, σ ≈ 2 %: +2 a +10).
* La contabilidad del entorno es coherente: la variación relativa del balance por paso y la recompensa difieren como mucho en ≈ 4·10⁻⁴ (término `comisión × retorno`). Esta cifra procede de la comprobación de la auditoría anterior; los scripts de la Fase 1 no la recalculan.
* **Browniano:** Buy & Hold tiene Σ R **positiva** (+3,6) y log-retorno **negativo** (−6,2). Las 3 políticas que compran y mantienen tienen Σ R entre +3,6 y +4,7 y log-retorno entre −6,0 y −4,7. El signo cambia según el criterio. Es un **hecho observado**, no solo un argumento teórico.
* **Regímenes:** con la métrica del entorno SARSA parece rentable de forma defendible (Σ R +4,6 [+1,9, +7,3]); con el log-retorno que usa la bitácora, no lo es. La conclusión "rentable" depende de qué métrica se use.

**¿Puede la discrepancia explicar decisiones del agente?** Se rehízo la selección de cada etapa (α, γ, ε) usando la recompensa acumulada de validación en lugar del log-retorno (`resultados/seleccion_criterio.csv`, 24 selecciones):

* Cambiaría **4 de 24** (3 en cíclico y 1 en browniano Q-Learning, ε: f = 0,3 → f = 0,6). Con retorno simple, 5 de 24.
* En browniano cambia **1 de 6**; los márgenes entre la primera y segunda variante en validación son de 0,01 a 0,24 pp, es decir, ruido. En cíclico los márgenes son de 2 a 8 pp frente a un error estándar de validación de ≈ 25 pp.
* **No hay evidencia de que el criterio de selección (log-retorno) sea el responsable de que el browniano "no opere".** Lo observado es que, entre políticas casi empatadas, la elección es arbitraria.
* Tampoco se puede afirmar causalidad sobre el comportamiento del agente: que 3 de 10 políticas del browniano sean Buy & Hold-like es compatible con una recompensa esperada ligeramente positiva (+0,0001 por paso), pero no lo demuestra. **Explicación causal no demostrada.**

Figura: [`fig5_recompensa_vs_rendimiento.png`](figuras/fig5_recompensa_vs_rendimiento.png).

### 4.5 Estabilidad de las políticas

Fuente: `resultados/estabilidad_politicas*.csv`. Cada política tiene 20 decisiones binarias (10 bins × {compra si fuera, vende si dentro}). No hay **ningún empate exacto** entre acciones válidas en las 40 políticas finales (en las 600 corridas, los 533 empates están en las 40 de "10 uniformes": bins sin visitar).

| Generador | Alg. | Políticas distintas | Acuerdo estados | Acuerdo ponderado por visitas | Rango log-ret. test (pp) | Estados en desacuerdo |
|---|---|---|---|---|---|---|
| Cíclico | Q | 3/5 | 0,95 | 0,97 | [605,1, 618,1] | 2 (bins 4 y 5, compra) |
| | S | 4/5 | 0,92 | 0,95 | [602,8, 618,1] | 3 |
| Browniano | Q | 5/5 | 0,63 | 0,71 | [−6,0, 0,0] | 14 |
| | S | 5/5 | 0,71 | 0,86 | [−4,7, 0,0] | 11 |
| Reversión | Q | 2/5 | 0,97 | 0,98 | [21,61, 21,68] | 1 |
| | S | 2/5 | 0,98 | 0,98 | [20,40, 21,68] | 1 |
| Regímenes | Q | 4/5 | 0,88 | 0,92 | [−3,6, +3,6] | 4 |
| | S | 4/5 | 0,91 | 0,93 | [0,0, +3,6] | 4 |

* **Estabilidad de la política ≠ estabilidad del rendimiento.** En browniano hay 5 políticas distintas pero solo 2 a 3 resultados (cero o Buy & Hold): las diferencias están en estados de "vender" que nunca se visitan si el agente no entra. En regímenes, políticas casi iguales (acuerdo 0,88–0,91) dan resultados que van de −3,6 a +3,6 pp: pocas diferencias de estado pesan mucho (p. ej., comprar solo en el bin 5 o 6).
* De los **65 pares** de políticas con alguna diferencia, **21** difieren menos de 0,5 pp en rendimiento medio y **9** dan resultado idéntico en el 100 % de las series. Dos políticas distintas **pueden** producir resultados financieros similares.
* **Entre algoritmos** el acuerdo medio es parecido al de dentro de cada uno (cíclico 0,94, browniano 0,71, reversión 0,96, regímenes 0,90), coherente con que no se distingan en rendimiento.
* En reversión y cíclico las políticas coinciden con la señal de la bitácora (H8): compra en bins 0–1 y vende en 8–9 (reversión); momentum (cíclico).

Figura: [`fig6_politicas_por_semilla.png`](figuras/fig6_politicas_por_semilla.png).

---

## 5. Convergencia y estabilidad (sin reentrenar)

**Datos.** `hist['eval_logret']` (51 puntos: episodio 1 y cada 100) de 480 entrenamientos únicos. Las 120 repeticiones exactas de (configuración, semilla) entre etapas dan **idénticas Q y curvas** (diferencia 0): el entrenamiento es determinista y reproducible.

**Limitaciones de los datos de partida.** (i) La curva usa las 10 primeras series de validación, no las 50; su último punto coincide con la media de esas 10 series guardadas, de ahí que la meseta de reversión (≈ 32 %) difiera del ≈ 22 % sobre las 50. (ii) Ruido de muestreo no cuantificable por punto. (iii) Solo hay una curva por entrenamiento.

**Criterio propuesto.** Ancla inferior = agente aleatorio (semilla 0) en esas mismas 10 series; meseta = media de los últimos 11 puntos de la propia curva; umbral(q) = ancla + q·(meseta − ancla) con q ∈ {0,80; 0,90; 0,95}; curva suavizada con media móvil de 3 puntos. **Estabilización permanente** = primer punto desde el que la curva suavizada se mantiene por encima del umbral hasta el final; **cruce inicial** = primer punto que lo alcanza (equivalente al criterio original). Si nunca queda de forma permanente: "no estabiliza" (censura 5 100).

**Limitaciones del criterio.** Alcanzar un umbral de rendimiento **no demuestra convergencia matemática** de Q: describe cuándo la política greedy deja de mejorar de forma apreciable en una métrica ruidosa. Es circular (la meseta sale de la misma curva), depende del suavizado y, cuando la política óptima es "no operar" (browniano), mide simplemente cuándo el agente deja de entrar al mercado.

### A. Configuraciones finales (distintas entre algoritmos)

Se replicó el criterio original: **8 de 8** coinciden con la columna "Episodios hasta convergencia" de la tabla (cíclico 500/500, browniano 200/200, reversión 2 500/800, regímenes 1 000/1 300). Con el criterio nuevo, por semilla (mediana; test de permutación exacto, 5 frente a 5; p mínimo posible 0,008):

| Generador | q = 80 % (Q / S) | q = 90 % (Q / S) | q = 95 % (Q / S) | Cruce inicial 90 % |
|---|---|---|---|---|
| Cíclico | 100 / 100 (p = 1,0) | 200 / 200 (p = 1,0) | 1 900 / 200 (p = 0,29) | 200 / 200 |
| Browniano | 200 / 200 | 200 / 200 | 200 / 200 | 200 / 200 |
| Reversión | 1 000 / 200 (p = 0,079) | 2 000 / 2 000 (p = 0,89) | 2 700 / 3 400 (p = 0,87) | 500 / 300 (p = 0,87) |
| Regímenes | 200 / 800 (**p = 0,040**, Q antes) | 1 700 / 1 000 (p = 0,70, S antes) | 1 700 / 1 000 (p = 0,40) | 200 / 200 |

**El ordenamiento cambia con el umbral.** En regímenes, a q = 80 % Q-Learning estabiliza antes (p = 0,04 sin ajustar) y a q = 90 % lo hace SARSA. En reversión, el "SARSA converge antes" de la bitácora aparece solo con q = 80 % o con el cruce inicial, y desaparece con 90 % (medianas iguales) y 95 % (invertido). Ninguna diferencia sobrevive a una corrección por comparaciones múltiples.

### B. Hiperparámetros idénticos (33 configuraciones donde ambos algoritmos comparten todo salvo `algoritmo`)

Incluye los 20 pares de la etapa α, más los de γ, ε y discretización donde los ganadores previos coincidían. Resumen con q = 90 % permanente (`resultados/convergencia_resumen_B.json`):

| | n | SARSA antes | Q-Learning antes | Empate | p < 0,05 sin ajustar |
|---|---|---|---|---|---|
| Total | 33 | 19 | 4 | 10 | 3 |
| Cíclico | 7 | 4 | 0 | 3 | 0 |
| Browniano | 12 | 6 | 2 | 4 | 1 |
| Reversión | 5 | 2 | 2 | 1 | 1 |
| Regímenes | 9 | 7 | 0 | 2 | 1 |

* **Tendencia descriptiva:** SARSA estabiliza antes en más configuraciones (19 frente a 4). Pero (i) solo 3 de 33 pares tienen p < 0,05 sin ajustar, que es del orden de lo esperable por azar (≈ 1,7), y las tres apuntan en sentidos mixtos (uno con Q-Learning antes, dos con SARSA antes); (ii) los 33 pares **no son independientes** (mismos generadores y semillas, hiperparámetros parecidos), así que no se calcula un p global; (iii) 12 de los 33 son de browniano, donde "estabilizar" significa dejar de operar; (iv) la ventaja descriptiva parece concentrarse en configuraciones lentas para Q-Learning (α alto, ε constante), lectura visual de `convergencia_hiperparametros_identicos.csv` que no se ha contrastado.
* **Conclusión:** **no se puede concluir qué algoritmo converge antes.** Hay una tendencia que merecería un experimento diseñado a propósito (§11), no una afirmación.
* **Tiempo:** el tiempo medio por entrenamiento es 113 s (Q-Learning) y 103 s (SARSA), medido con 32 procesos en paralelo y 5 000 episodios completos. Es tiempo de entrenamiento total bajo contención de recursos, no tiempo de convergencia; no se extrapola. Los dos algoritmos difieren solo en el orden en que se elige `a'`, así que cualquier diferencia de tiempo no aporta información sobre convergencia.

Figuras: [`fig4_convergencia_sensibilidad.png`](figuras/fig4_convergencia_sensibilidad.png), [`fig4b_convergencia_hiperparametros_identicos.png`](figuras/fig4b_convergencia_hiperparametros_identicos.png).

---

## 6. Contraste con las conclusiones de Marco

| Conclusión original | Nueva evidencia | Veredicto | Justificación |
|---|---|---|---|
| Resultados con "± desviación entre semillas" (p. ej. +21,6 ± 0,0 en reversión) | IC por series [+18,3, +25,2]; de población [+16,8, +26,5]; la desviación entre semillas es 0,04 pp | **Se rechaza** como medida de incertidumbre | Mide cuántas políticas distintas hay (2 de 5), no el error de estimación |
| Cíclico: +610 % (Q) / +612 % (S), muy por encima de B&H (+338 %) | Diferencia +273 [+230, +315] frente a B&H; 96–98 % de series | **Se mantiene** (robusta) | La interpretación como artefacto del generador y del horizonte (auditoría) no se ha reanalizado aquí |
| Q-Learning ≈ SARSA en rendimiento (H7/H9) | Diferencias entre −1,7 y +0,2 pp; IC de población incluye 0 en los 4 | **Se mantiene** como "no se detecta diferencia" | IC amplios (±12 pp en cíclico): no es prueba de equivalencia |
| SARSA converge antes en reversión (1 400/800 frente a 2 500) | Con hiperparámetros idénticos y criterio permanente, ordenamiento cambia con q; 19/33 frente a 4/33 solo descriptivo | **Se rechaza** como afirmación | Confusión de hiperparámetros, de umbral y de muestra |
| Browniano: "no opera" y "lo óptimo es no operar" (H3/H4) | Mezcla de efectivo y compra-y-mantén; Q −2,4 [−7,9, +3,1], S −0,9 [−3,5, +1,7] frente a efectivo; Σ R positiva y log-retorno negativo para B&H | **Se matiza** | No es "no operar": 3/10 políticas compran y mantienen; no se distingue de efectivo ni de B&H; el óptimo depende del criterio |
| Reversión supera claramente a B&H y aprende "reversión a la media de manual" | +22,7 [+17,4, +28,1]; 89–90 % de series; 2 políticas distintas con acuerdo 0,97 | **Se mantiene** (robusta) | El resultado más sólido del trabajo |
| Regímenes: el valor del agente es evitar pérdidas (+1–3 % frente a −22 %) | Frente a B&H +22,7 (exploratorio, Holm 0,054) y +24,5 (robusto en el límite, Holm 0,047); frente a efectivo +0,8 [−6,2, +7,8] y +2,6 [−1,4, +6,5] | **Se mantiene en el matiz**: "evita pérdidas", no "es rentable" | Superar B&H en un escenario bajista no es rentabilidad absoluta |
| Log-retorno como métrica principal (D11) | Σ R y log-retorno correlan ≥ 0,995 entre series pero difieren en nivel y signo (browniano) | **Se mantiene** con salvedad | Debe explicarse que el agente optimiza una recompensa distinta de la métrica que se reporta |
| La selección α → γ → ε elige lo mejor (D13) | Cambiar el criterio a Σ R alteraría 4 de 24 selecciones; márgenes de 0,06–0,24 pp en browniano | **Se matiza** | En variantes casi empatadas la elección es ruido; no es un hallazgo |
| "Tiempo (s)" de la tabla como tiempo de convergencia | Es tiempo total de 5 000 episodios con 32 procesos en paralelo | **Se rechaza** como tiempo de convergencia | Debe reportarse en episodios hasta estabilización con intervalo |
| Políticas finales coinciden con la señal (H8) | Reversión y cíclico: acuerdo 0,92–0,98 entre semillas; regímenes: 4 políticas distintas, rendimiento −3,6 a +3,6 | **Se mantiene** donde hay señal fuerte | En regímenes la ventaja depende de pocos estados |
| Evaluación greedy determinista (preocupación de la auditoría) | 0 empates exactos en las 40 políticas finales; reproducción exacta de val/test | **Confirmado (sin problema en resultados finales)** | El riesgo es teórico; los 533 empates están en la variante "10 uniformes" |

**Resultado de las hipótesis de partida.** H1 confirmada (el ± original subestima: p. ej. 0,04 pp frente a un IC de ≈ ±5 pp). H2 no refutada (los 4 IC de población incluyen 0), sin que eso pruebe equivalencia. H3 no refutada en browniano; en regímenes tampoco para Q-Learning y solo como indicio débil para SARSA. H4 confirmada, incluido el cambio de signo en el browniano. H5 confirmada (el orden cambia con el umbral y la tendencia con hiperparámetros idénticos es solo descriptiva). H6 confirmada en reversión y cíclico, parcial en browniano y regímenes (políticas distintas, pocos resultados distintos).

---

## 7. Validaciones ejecutadas

Fuente: `resultados/verificacion_reproduccion.csv` y `verificacion_convergencia.csv`. Todas OK.

| Comprobación | Resultado |
|---|---|
| Pickles: sin opcodes `INST/OBJ/GLOBAL`; globales solo numpy | OK |
| Réplica del entorno frente a `SimpleTradingEnv` extraído del notebook (40 episodios aleatorios) | Diferencia 0 |
| Buy & Hold y Aleatorio recalculados frente a `baselines.pkl` (log-retorno, balance, operaciones) | Diferencia 0 |
| Efectivo: balance 1 000, 0 operaciones, recompensa 0 (400 series) | OK |
| Política reconstruida de Q frente a val/test guardados: **560 entrenamientos × 150 series** (log-retorno, balance, operaciones) | Diferencia 0 |
| Configuración ganadora frente a `log_ajuste.txt` (8 pares) | OK |
| Tabla original (media ± desv., mediana de balance, operaciones; 16 filas) | OK (≤ redondeo) |
| Criterio original de convergencia frente a la tabla (8 de 8) | OK |
| Determinismo: 120 repeticiones (config, semilla) entre etapas | Q y curva idénticas |
| Último punto de la curva guardada = media de las 10 primeras series de validación | OK (2·10⁻⁷, float32) |
| Series de test (`seed 7777`) y validación (`seed 2024`) regeneradas correctamente (B&H cerrado frente a simulado) | OK |
| Emparejamiento: misma serie *j* en todos los brazos | OK |
| Reejecutar los tres scripts da CSV byte a byte idénticos (bootstrap con semilla fija) | OK (17 archivos) |
| Nada modificado fuera de `analisis_adicional/`; ningún archivo original tocado | `git status` / `git diff` |

**Excepción documentada.** 40 entrenamientos ("10 uniformes") tienen empates exactos en Q (533 celdas): su política greedy original desempata al azar y no puede reproducirse de forma determinista. Se excluyen de la exigencia de igualdad exacta y se informa de ello; ninguno pertenece a las configuraciones finales.

**Dependencias.** Python 3.14.3, NumPy 2.4.2, Matplotlib 3.10.8 (solo para figuras). No se usó SciPy, pandas ni gymnasium; no se instaló nada. Ejecución desde la raíz del repositorio: `py -3.14 -B -X utf8 analisis_adicional/<script>.py` (≈ 30 s el primero; segundos los demás). `-B` evita crear `__pycache__`; `-X utf8` evita errores de codificación en la consola de Windows.

---

## 8. Limitaciones estadísticas

1. **Solo 5 semillas.** El IC de población usa 4 g.l.; la parte atribuible a semillas es incierta y los resultados dependen de la aproximación de efectos aleatorios. El bootstrap jerárquico sale más estrecho y no debe usarse como principal.
2. **Un único conjunto de test (100 series, `seed 7777`)**, ya usado en el desarrollo y reutilizado aquí para todas las comparaciones. No es una confirmación independiente. La potencia es baja para diferencias pequeñas (p. ej., Q − SARSA tiene IC de ±12 pp en cíclico).
3. **Familia de 28 comparaciones** con Holm; los p-valores t con 4 g.l. son aproximados. SARSA − B&H en regímenes es robusto por margen estrecho (0,047 frente a 0,05); Q-Learning − B&H en regímenes queda en 0,054. No deben presentarse como dicotómicos.
4. **Log-retorno medio** como estimando: es apropiado para crecimiento compuesto pero no es el patrimonio esperado. En cíclico la media aritmética del patrimonio está dominada por pocas series (auditoría); aquí se reporta log-retorno.
5. **Recompensa reconstruida**, no guardada: coincide con la del entorno porque la réplica es exacta, pero no se pudo contrastar con una recompensa de entrenamiento almacenada.
6. **Convergencia:** curvas con 10 series, una por entrenamiento; criterio dependiente del umbral y de la meseta; no se evalúa convergencia matemática de Q.
7. **Las 5 políticas por celda no son una muestra aleatoria perfecta de la población de políticas**, y los efectos de semilla de Q-Learning y SARSA se tratan como independientes aunque comparten las secuencias de entrenamiento.
8. **No se reanalizó** la brecha validación/test (H11) ni el efecto de α (H4/H6): están fuera del alcance de esta fase.

---

## 9. Implicaciones para la presentación al profesor

* Sustituir "± entre semillas" por IC por series (y, si hay espacio, de población). La tabla de §4.1 sirve como tabla principal.
* Incluir el baseline de efectivo en la tabla y en las figuras. En browniano y regímenes cambia la lectura: el RL no aporta valor medible frente a no operar.
* Formular las conclusiones por mercado así: **reversión** (robusta, la mejor demostración); **cíclico** (robusta, pero atribuible al generador y al horizonte); **browniano** (sin ventaja; es una política de efectivo/Buy & Hold); **regímenes** (evita pérdidas, no demuestra rentabilidad).
* No afirmar superioridad de ningún algoritmo ni "converge antes". Se puede decir: "no se detectan diferencias en rendimiento; Q-Learning opera algo más en los cuatro mercados (no significativo); en convergencia hay una tendencia a favor de SARSA con hiperparámetros idénticos que requiere un experimento específico".
* Explicar en una frase la discrepancia recompensa/log-retorno, con el browniano como ejemplo (Σ R > 0 y log-retorno < 0 para Buy & Hold). Es una aportación original y verificable.
* Para la "evolución": esta fase aporta rigor estadístico, pero **no** demuestra la evolución desde la configuración inicial (ver §10).

---

## 10. Experimentos pendientes para la siguiente fase

**Prioridad 1 — Evolución configuración inicial → final.** Reentrenar con `Config()` por defecto (α = 0,05, γ = 0,95, 2 000 episodios) y con la configuración final, en los 4 generadores y ambos algoritmos, mismas semillas y mismo test. Hipótesis: la configuración inicial no opera en los mercados de señal débil y la final recupera la señal; el rendimiento satura con el número de episodios. Variables controladas: discretización, semillas, series de test. Métricas: log-retorno, operaciones, IC pareados por serie. Falsaría: que la configuración por defecto iguale a la optimizada. Coste: 40 entrenamientos × 2 000 episodios; con los 17,5 µs/paso medidos por Marco son ≈ 17 s por entrenamiento en un núcleo (unos 12 min en serie). Requiere, o bien instalar `gymnasium` (necesita autorización), o bien reimplementar el agente en `analisis_adicional/` sobre la réplica verificada del entorno. Barrido de episodios {500, 1 000, 2 000, 5 000, 10 000}: ε depende de N, no basta truncar curvas.

**Prioridad 2 — Test fresco con las políticas guardadas (sin reentrenar).** Evaluar las 40 políticas finales sobre una semilla de test nueva y mayor (p. ej., 1 000 series). Sin coste de entrenamiento; resuelve la baja potencia y la reutilización del test.

**Prioridad 3 — Convergencia en condiciones justas.** Misma configuración en ambos algoritmos (rejilla α × ε), curvas sobre ≥ 50 series de validación, ≥ 20 semillas y registro de instantáneas de Q (para medir la variación de Q, no solo de la política). Sin esto no se puede responder "quién converge antes".

**Otros.** Robustez a σ, κ y `n_steps` por inferencia (piloto en la auditoría anterior; falta hacerlo con IC), efecto de α sobre la varianza de Q entre semillas (explicación de H4 sin contrastar), brecha validación/test en regímenes con las series de validación regeneradas.

---

## Anexo: archivos

* `comun.py`: carga restringida de pickles, réplica del entorno y generadores, reconstrucción de políticas, utilidades estadísticas.
* `reanalisar_resultados.py`: verificación, métricas, comparaciones, recompensa frente a rendimiento, selección por criterio, figuras 1, 2, 3 y 5.
* `analizar_convergencia.py`: convergencia, sensibilidad al umbral, hiperparámetros idénticos, figuras 4 y 4b.
* `analizar_politicas.py`: estabilidad de las políticas, figura 6.
* `resultados/`: `metricas_por_agente.csv`, `comparaciones_pareadas.csv`, `observaciones_test.csv` (6 800 filas), `recompensa_vs_financiero*.csv`, `seleccion_criterio.csv`, `convergencia_*.csv/json`, `estabilidad_politicas*.csv`, `verificacion_*.csv`.
* `figuras/`: `fig1` a `fig6` (las originales de Marco no se tocan).
