# Contribuciones de Dani al Taller RL

Documento de trazabilidad para Marco, Cristian y el profesor. Su objetivo es que cualquiera pueda identificar **qué se hizo, cómo, por qué y qué se aprendió**, y de quién es cada pieza.

Fecha de redacción: 28/10/2026. Rama: `feature/analisis-estadistico-rl`, que parte del commit `5a6e1ee` y todavía no tiene commits propios.

---

## 0. Cómo leer este documento: etiquetas de procedencia

Cada pieza lleva una de estas cuatro etiquetas. Si algo no la lleva, es una descripción, no un resultado.

| Etiqueta | Significado | Quién |
|---|---|---|
| **[ORIGINAL]** | Código, decisiones o texto escritos por Marco y presentes en el commit `5a6e1ee` (27/10/2026, 18:49 +01:00) | Marco Corpa |
| **[REUTILIZADO]** | Resultados que Marco generó y guardó (`resultados/*.pkl`), analizados de nuevo **sin reentrenar y sin modificarlos** | Marco (datos); Dani (análisis) |
| **[REPRODUCIDO]** | Entrenamientos de Marco repetidos con su código original y comparados con sus artefactos | Dani |
| **[NUEVO]** | Análisis, entrenamientos, scripts o documentos que no existían en el repositorio | Dani |

**Autoría del código del taller.** El entorno `SimpleTradingEnv` y los cuatro generadores de precios proceden del enunciado del profesor (`taller_trader.ipynb`), que Marco copió en `trader_rl.py`. El agente tabular, Q-Learning, SARSA, la discretización por cuantiles, la máscara de acciones, el protocolo de ajuste y la bitácora son de Marco. **Nada de eso se ha reescrito ni modificado.**

**Asistencia de IA.** Los análisis y documentos de esta rama se elaboraron con ayuda de Claude Code (herramienta de IA de Anthropic), bajo la dirección de Dani. Conviene que el equipo decida cómo declararlo en la entrega (ver `RESUMEN_PARA_EQUIPO.md`, sección de decisiones).

---

## 1. Estado inicial del repositorio de Marco

**Fuente:** `git log`, `git ls-files` y lectura directa. Remoto: `origin` = `github.com/marcocorpacriado-pixel/taller_RL`.

| Elemento | Estado en `5a6e1ee` |
|---|---|
| Historial | **Un único commit** de Marco Corpa (27/10/2026 18:49). Existe otra rama remota, `origin/datos-reales-btc` (commit `db4a963`, 27/10/2026 19:29, experimento con BTC-USD diario), que no está en `main` |
| `trader_rl.py` | Entorno, generadores, discretizador, agente tabular (Q-Learning y SARSA con un único `AgenteTabular`), entrenamiento, evaluación, baselines |
| `experimentos.py` | Ajuste secuencial α → γ → ε, experimento de discretización, señal por bin. Selección por validación (50 series, semilla 2024); test (100 series, semilla 7777) solo para reportar |
| `graficas.py` | 9 figuras y `figuras/tabla_resultados.md` |
| `BITACORA.md` | Requisitos, hallazgos H1–H11, decisiones D1–D16 y registro de cambios |
| `resultados/` | `alpha.pkl`, `gamma.pkl`, `epsilon.pkl`, `discretizacion.pkl` (600 entrenamientos), `baselines.pkl`, `senal.pkl`, `log_ajuste.txt` |
| `taller_trader.ipynb` | **Solo el enunciado del profesor**, sin código del alumno |
| Lo que no existía | PDF de entrega, comparación cuantificada "configuración inicial frente a final", barrido del número de episodios, baseline de efectivo, intervalos de confianza por serie |

Los 600 entrenamientos de Marco son: α (5 variantes × 8 combinaciones generador-algoritmo × 5 semillas = 200), γ (120), ε (120) y discretización (160). Según la bitácora, el ajuste tardó unos 24 min, la repetición con el grid ampliado de α unos 26 min y la discretización unos 15 min, con 32 núcleos.

**Limitación estructural.** Los `.pkl` solo contienen la pasada final: la "primera pasada" con la configuración por defecto y el grid inicial de α (4 valores) fueron sobrescritos. Esa parte de la historia existe únicamente como texto en la bitácora (H4 y la "primera pasada" de H7).

---

## 2. Objetivo de las aportaciones y relación con los requisitos del profesor

**Requisitos** (extraídos del enunciado por Marco en `BITACORA.md` §0 y comprobados contra el notebook en la auditoría): Q-Learning y SARSA en los 4 generadores; muchos episodios; test con series nuevas; comparar con aleatorio y Buy & Hold; razonar si la estrategia es la esperada; ajustar α, ε, γ **y número de episodios**; tabla con rendimiento y **tiempo de convergencia**. Evaluación: 50 % mínimos, 10 % ir más allá, 20 % presentación, 10 % justificación de decisiones, 10 % análisis crítico. Entrega: PDF de máximo 2 páginas el 29/10/2026 a las 23:59.

| Requisito o criterio | Hueco detectado en el repositorio inicial | Aportación |
|---|---|---|
| Comparar con aleatorio y Buy & Hold; razonar si es lo esperado | Sin baseline de "no operar"; la incertidumbre ("± entre semillas") no refleja la de las series de test | Fase 1: efectivo, IC por series y de población, comparaciones pareadas |
| Ajustar el **número de episodios** | Fijo en 5 000; nunca se barrió | Fase 2: barrido de N con calendario de ε recalibrado |
| **Tiempo de convergencia** | Criterio frágil; "Tiempo (s)" es el entrenamiento completo con 32 procesos en paralelo | Fase 1: criterio con umbral variable; Fase 2: criterios de "N suficiente" |
| Justificación de decisiones | La evolución "inicial → final" existía solo como texto | Fase 2: reconstrucción cuantificada con el mismo N |
| Análisis crítico | Algunas afirmaciones sin contrastar | Auditoría, Fases 1 y 2: registro de matices (§7) |

---

## 3. Auditoría inicial [NUEVO]

**Cuándo y cómo.** Mañana del 28/10/2026, en **solo lectura**: no se modificó ni generó ningún archivo del repositorio. El informe de la auditoría **se entregó en la conversación y no se guardó como archivo**; este apartado lo resume. Las cifras marcadas con ★ en aquel informe se calcularon en memoria con los artefactos guardados y **no están en el repositorio**.

**Qué se revisó.** `trader_rl.py`, `experimentos.py`, `graficas.py`, `BITACORA.md`, el notebook, el log, la tabla, tres figuras y los dos commits, incluido el diff de `origin/datos-reales-btc`. **No se auditó** `datos_reales.py` ni el CSV de BTC, ni se recalculó la tabla H10, ni se ejecutó ningún entrenamiento. Los `.pkl` se inspeccionaron sin ejecutarlos (solo referencian `numpy.dtype` y `numpy._core.numeric._frombuffer`) y se cargaron con una lista blanca.

**Resultado principal.** Motor correcto y protocolo razonable (sin fuga de test en la selección, comparación pareada de hecho), pero con debilidades de entrega y estadística:

| Hallazgo | Gravedad en la auditoría |
|---|---|
| Los "±" de la tabla son desviaciones entre 5 semillas sobre las mismas 100 series; ignoran la incertidumbre por series | Principal |
| Falta el baseline de efectivo | Principal |
| Tiempo y criterio de convergencia frágiles; comparación Q-Learning/SARSA confundida por hiperparámetros distintos | Principal |
| El historial Git no demuestra la evolución (un solo commit) | Principal |
| El notebook committed es solo el enunciado (la bitácora dice que "importa este módulo") | Baja, pero visible |
| `accion()` consume números aleatorios incluso con ε = 0; la evaluación greedy durante el entrenamiento comparte el RNG del agente | Media (C1) |
| `ganadores()` elige por la media de validación sin tolerancia; en browniano gana una variante por 0,01 pp | Media (C3) |
| Curva de convergencia sobre 10 series, no 50 | Media (C4) |
| La grid de α de H4 se diseñó mirando el test | Baja-media (C10) |
| H9 (1 400 episodios) no coincide con la tabla (800) | Baja (C9) |

**Hechos que la auditoría dio por buenos:** la réplica del entorno reproduce los `logret` de test guardados a precisión de máquina; Q-Learning y SARSA están bien implementados; los 533 empates exactos en Q están todos en la variante "10 uniformes" (con 10 cuantiles no hay empates).

**Lo que la auditoría afirmó pero no está respaldado por archivos del repositorio** (no usar como evidencia sin repetirlo):

* El piloto de robustez por inferencia (κ, volatilidad, `n_steps`; una semilla, 60 series, cortes sin recalibrar) y la dependencia lineal del +610 % del cíclico con el horizonte (+316 / +624 / +1 272 % para 250 / 500 / 1 000 pasos).
* La explicación alternativa a la brecha validación/test en regímenes (composición de las muestras en lugar de *winner's curse*).
* Los errores estándar de la auditoría para Q-Learning − SARSA con bootstrap simple; **los sustituyen los de la Fase 1**, que separan series y semillas.

---

## 4. Fase 1 — Reanálisis estadístico sin reentrenar [REUTILIZADO + NUEVO]

**Fecha:** 28/10/2026 (archivos modificados entre las 08:37 y las 08:44, hora local). **Entrenamientos nuevos: ninguno.**

**Qué es de Marco y qué es nuevo.** Los datos son los 600 entrenamientos y los baselines de Marco [REUTILIZADO]. Todo el análisis estadístico, los scripts y las figuras son [NUEVO]. El entorno y los generadores se **re-implementaron** en `comun.py` (réplica del enunciado) y se verificó que coincide con `SimpleTradingEnv` del notebook (diferencia 0 en 40 episodios aleatorios).

**Metodología.**

* Matriz `L[semilla, serie]` de log-retornos en test (5 × 100). Las 500 observaciones **no** se tratan como independientes.
* Dos intervalos distintos: **IC por series** (bootstrap percentil, condicional a estas 5 políticas) e **IC de población** (series + semillas; efectos aleatorios cruzados por el método de momentos; t con 4 g.l.).
* Comparaciones **pareadas por serie**. Familia confirmatoria declarada a priori: 28 contrastes con corrección de Holm. "Robusta" exige a la vez Holm < 0,05, ambos IC excluyendo 0 y ≥ 80 % de pares de semillas en el mismo sentido. Un IC que incluye 0 **no** se interpreta como equivalencia.
* Baseline de efectivo implementado fuera del entorno (acción "mantener" siempre).
* Recompensa del entorno frente a log-retorno; selección con otro criterio; convergencia con umbral variable (80/90/95 %) y estabilización permanente; estabilidad de políticas.

**Resultados principales** (test, log-retorno medio en pp; IC95 de población entre corchetes):

| Mercado | Efectivo | Buy & Hold | Q-Learning | SARSA |
|---|---|---|---|---|
| Cíclico | 0 | +337,5 [+277,1, +397,9] | +610,4 [+566,2, +654,5] | +611,5 [+568,0, +655,0] |
| Browniano | 0 | −6,2 [−18,2, +5,8] | −2,4 [−7,9, +3,1] | −0,9 [−3,5, +1,7] |
| Reversión | 0 | −1,1 [−3,2, +1,0] | +21,6 [+16,8, +26,5] | +21,4 [+16,5, +26,4] |
| Regímenes | 0 | −21,9 [−35,2, −8,6] | +0,8 [−6,2, +7,8] | +2,6 [−1,4, +6,5] |

* De las 28 comparaciones, **17 son robustas** (todo frente a aleatorio, cíclico y reversión frente a B&H y efectivo, y regímenes SARSA − B&H en el límite: Holm 0,047), 1 es indicio y **10 no son concluyentes** (entre ellas las 4 de Q-Learning − SARSA, y browniano y regímenes frente a efectivo).
* "+21,6 ± 0,0" en reversión escondía una incertidumbre real de unos ±5 pp: la desviación entre semillas (0,04 pp) mide cuántas políticas distintas hay (2 de 5), no el error de estimación.
* En browniano, la política "no opera" es en realidad una mezcla: 3 de 10 políticas compran y mantienen.
* La recompensa del entorno sobreestima el log-retorno y **cambia de signo en el browniano** para Buy & Hold (Σ R = +3,6; log-retorno = −6,2). Cambiar el criterio de selección alteraría 4 de 24 selecciones; no hay evidencia de que explique el comportamiento del browniano.
* **Convergencia:** se replicó el criterio original (8 de 8 coinciden con la tabla). El orden Q-Learning/SARSA cambia con el umbral. Con hiperparámetros idénticos, SARSA estabiliza antes en 19 de 33 configuraciones y Q-Learning en 4, pero solo 3 de 33 tienen p < 0,05 sin ajustar, y no son independientes. **No se puede concluir quién converge antes.**
* **Estabilidad de políticas:** acuerdo entre semillas 0,92–0,98 en cíclico y reversión; 0,63–0,71 en browniano; 0,88–0,91 en regímenes. Política estable no implica rendimiento estable.

**Conclusiones de la Fase 1.** Reversión es el resultado más sólido; cíclico es robusto pero atribuible al generador y al horizonte; browniano no muestra ventaja; en regímenes el agente evita parte de las pérdidas de Buy & Hold, pero no se demuestra rentabilidad absoluta. No se detecta diferencia entre Q-Learning y SARSA en rendimiento.

**Limitaciones declaradas:** 5 semillas; un solo test ya usado; 28 contrastes con t de 4 g.l. aproximada; no se reanalizó H11 ni el efecto de α.

**Evidencia:** `analisis_adicional/INFORME_FASE1.md`, 17 CSV/JSON en `analisis_adicional/resultados/`, 7 figuras en `analisis_adicional/figuras/`.

---

## 5. Fase 2 — Reconstrucción experimental de la evolución [REPRODUCIDO + NUEVO]

**Fecha:** 28/10/2026. Entrenamientos entre las 16:28 y las 17:19 (hora del registro); análisis, tablas e informe entre las 17:25 y las 17:28. Registro completo en `fase2_evolucion/registro_ejecucion.log`.

**Pregunta.** ¿Cómo evolucionó el agente desde la configuración por defecto hasta la final, y qué evidencia justifica cada decisión?

**Entorno.** Un entorno virtual **aislado fuera del repositorio** (`C:\Users\dgall\.venvs\taller_rl_fase2`, ruta del equipo del autor; para recrearlo en otro equipo, `requirements_entrenamiento.txt` y `README_REPRODUCCION.md`): Python 3.12.10, `gymnasium 1.4.0`, `numpy 2.5.4`, más las dependencias de gymnasium (`cloudpickle 3.1.2`, `Farama-Notifications 0.0.6`, `typing_extensions 4.16.0`). Los entrenamientos **importan `trader_rl.py` sin modificarlo**; solo se añade una subclase que cuenta actualizaciones por (bin, posición, acción) sin consumir números aleatorios. El análisis usa el Python 3.14.3 del sistema (NumPy 2.4.2, Matplotlib 3.10.8).

**Puerta de reproducibilidad [REPRODUCIDO].** Antes de entrenar nada nuevo se reentrenaron **40 corridas de Marco** (grupo `REP`: reversión y regímenes, α = 0,002 y 0,005, N = 5 000). Las **40 son idénticas bit a bit** (Q, curvas, recompensas, balances, val y test). Esto valida el entorno nuevo y el contador de visitas.

**Entrenamientos [NUEVO].** Son **540 en total: 500 genuinamente nuevos y 40 réplicas de Marco** (`REP`). Ninguno usa el test para seleccionar nada.

| Grupo | Qué es | Generadores | Entrenamientos |
|---|---|---|---|
| `E0_N500/1000/2000/5000` | **E0** = `Config()` por defecto (α = 0,05 constante, γ = 0,95, ε 1 → 0,01 con fracción 0,6, 10 cuantiles, 500 pasos) con distinto N | 4 | 4 × 40 = 160 |
| `E0_N10000` | E0 con N = 10 000 | reversión, regímenes | 20 |
| `F_N500/1000/2000` | Configuración final de Marco reentrenada con otro N | 4 | 3 × 40 = 120 |
| `F_N10000` | Final con N = 10 000 | reversión, regímenes | 20 |
| `alfa{0,002; 0,005}_N{500; 1000; 2000; 10000}` | α × N sobre la base de Marco | reversión, regímenes | 8 × 20 = 160 |
| `E0eps0p1_N5000` | E0 con ε = 0,1 constante (intervención sobre el bloqueo) | reversión, regímenes | 20 |
| `REP` | Réplica de Marco (verificación) | reversión, regímenes | 40 |
| **Total** | | | **540** (500 sin las réplicas) |

**Reutilizado de Marco [REUTILIZADO]:** las etapas α, γ, ε y discretización (600 corridas, sin reentrenar). La cadena E1/E2/E3 son los ganadores por validación de cada etapa, reconstruidos con la misma regla que `experimentos.ganadores`. `F@5000` es el artefacto de la etapa ε de Marco.

**Importante: lo nuevo no es histórico.** Todo lo etiquetado "nuevo" se entrenó el 28/10/2026. E0 es una **reconstrucción experimental actual** de la "primera pasada" de Marco, no una recuperación de ella.

**Resultados.**

1. **E0 reproduce el punto de partida descrito en H4.** Reversión y regímenes: 10 de 10 corridas por mercado no operan (rendimiento 0,0 = efectivo). Browniano: 8 de 10 (las otras 2 operan ≈ 85 veces por serie y pierden −8,5 pp). Cíclico: opera y gana, pero es inestable (SD entre semillas 74–129 pp frente a 1,6–4,5 pp de la configuración final).
2. **La duración no explica el fallo de E0.** Con N = 10 000, 20 de 20 corridas de E0 siguen sin operar.
3. **El salto lo da α.** Con el mismo N = 5 000, los hiperparámetros dan +23,7 [+16,2, +31,2] (Q-Learning) y +21,4 [+16,5, +26,4] (SARSA) en reversión en test; robustos con Holm. La duración, con los mismos hiperparámetros, no aporta de forma detectable.
4. **En regímenes solo operan α ≤ 0,002**, y con α = 0,002 Q-Learning necesita N = 10 000 para que ninguna semilla se quede sin operar (+7,8 ± 3,3 en validación). Una α pequeña sí retrasa el aprendizaje.
5. **Mecanismo de α: parcialmente respaldado, no demostrado.** El ruido de Q entre semillas crece con α, pero menos que √α (pendientes 0,09–0,45). "Falta de exploración de comprar" queda rechazada (centenares de actualizaciones por estado). Con α = 0,05 la acción "mantener en posición" recibe una mediana de 3–6 actualizaciones tardías y Q negativo: es la firma de la bitácora, **pero es circular**. Forzar ε = 0,1 constante no recupera la señal.
6. **γ y ε aportan poco.** Solo SARSA en regímenes muestra un efecto de ε detectable (val +2,2 [+0,2, +4,3]; test +3,4 [+1,9, +4,9]). En regímenes el paso de γ empeora el test (−2,0 Q-Learning; −2,1 SARSA).
7. **N = 5 000 es razonable.** Con la configuración final no hay cambio detectable entre 5 000 y 10 000 en reversión y regímenes; 2 000 basta en cíclico (Q-Learning); regímenes no opera con N ≤ 1 000 (Q-Learning). Es convergencia empírica, no matemática.
8. **Discretización (E4, experimento posterior de sensibilidad):** los bins uniformes pierden en los tres mercados con señal; 5 cuantiles pierden en regímenes y cíclico Q-Learning; 20 ≈ 10 salvo cíclico Q-Learning.
9. **Q-Learning y SARSA:** ningún IC de población de la diferencia excluye 0 en ninguna etapa. Q-Learning opera algo más (p. ej. 62 frente a 53 en cíclico final).
10. **Un agente que no hace nada ya supera a Buy & Hold en regímenes** (+21,9 [+8,6, +35,2]). Superar a B&H en un mercado bajista no demuestra aprendizaje.

**Recortes de alcance declarados.** N = 10 000, α × N y la intervención con ε solo en reversión y regímenes; α × N solo con α = 0,002 y 0,005 (no se probó α = 0,001 con N = 10 000); no se reejecutó la cadena α → γ → ε ni se re-ablacionaron γ y ε por separado; no se reconstruyó la primera pasada histórica.

**Limitaciones.** El test ya lo había inspeccionado Marco: la fase es **exploratoria**, no confirmación independiente. Solo 5 semillas. El mecanismo causal de α no está demostrado. El entorno de Marco (versiones de Python y NumPy) es desconocido; la réplica 40/40 muestra que no afecta a los resultados.

**Evidencia:** `analisis_adicional/fase2_evolucion/INFORME_FASE2.md` (13 secciones y 5 anexos), 19 archivos de resultados (CSV, JSON y tablas) en `fase2_evolucion/resultados/`, 6 figuras en `fase2_evolucion/figuras/`, 19 artefactos `.npz` en `fase2_evolucion/artefactos/`.

---

## 6. Resultados originales que se confirmaron

| Afirmación de Marco | Cómo se confirmó | Fuente |
|---|---|---|
| Los resultados guardados son coherentes con el código | La política reconstruida desde `Q` y `cortes` reproduce val y test de **560 entrenamientos × 150 series** con diferencia 0 (los otros 40, de "10 uniformes", tienen empates en Q y no pueden reproducirse de forma determinista) | Fase 1 |
| El entrenamiento es determinista | 120 repeticiones (configuración, semilla) entre etapas dan Q y curvas idénticas; 40 de 40 réplicas bit a bit en Fase 2 | Fases 1 y 2 |
| Tabla final (H7), baselines y configuraciones ganadoras | 16 filas de la tabla, 8 pares ganadores frente a `log_ajuste.txt`, Buy & Hold y aleatorio recalculados: todos coinciden | Fase 1 |
| Reversión: el agente aprende la reversión a la media y supera a Buy & Hold | +22,7 [+17,4, +28,1] frente a B&H; política de 2 de 5 variantes con acuerdo 0,97 | Fase 1 |
| Cíclico: +610 / +612 %, muy por encima de B&H | +273 [+230, +315] frente a B&H; 96–98 % de las series | Fase 1 |
| Q-Learning y SARSA rinden igual | Ningún IC de población de la diferencia excluye 0 (Fases 1 y 2). **Es "no se detecta diferencia", no equivalencia** | Fases 1 y 2 |
| H4: con α = 0,05 el agente no opera en reversión y regímenes | 10/10 corridas por mercado | Fase 2 |
| H6: hace falta α ≤ 0,002 en mercados de señal débil | Reversión y regímenes (con α ≤ 0,002 operan; en regímenes α ≥ 0,005 no opera en 10/10) | Fase 2 |
| H8: las políticas coinciden con la señal de referencia | Acuerdo 0,92–0,98 entre semillas en cíclico y reversión | Fase 1 |
| H10: los bins uniformes pierden; 10 cuantiles es la meseta | Reanalizado con IC pareados; uniformes pierden en los tres mercados con señal | Fase 2 |
| H2, H3, H5 (B&H por generador; señal por bin) | Coinciden con los artefactos (comprobado en la auditoría; `senal.pkl` y `baselines.pkl`) | Auditoría y Fase 1 |
| El criterio original de convergencia produce la tabla | 8 de 8 coinciden | Fase 1 |
| Seleccionar por validación y reportar en test | No se encontró fuga de test en la selección | Auditoría |

---

## 7. Afirmaciones de la bitácora que debemos matizar o corregir

No se ha editado `BITACORA.md`. Cada fila indica qué propongo; la decisión de cambiar la bitácora corresponde a Marco, como autor.

| # | Afirmación (bitácora) | Hallazgo | Propuesta | Evidencia |
|---|---|---|---|---|
| 1 | Tabla de resultados "media ± desviación entre semillas" (p. ej. +21,6 ± 0,0) | La desviación entre semillas no mide la incertidumbre sobre el rendimiento | **Rechazar** como medida de incertidumbre; usar IC por series (y, si cabe, de población) | Fase 1 §4.1, §6 |
| 2 | H9: "SARSA converge antes en reversión (1 400 frente a 2 500)" | El texto (1 400) y la tabla (800) no coinciden; la comparación mezcla hiperparámetros distintos; el orden depende del umbral | **Rechazar**; no afirmar qué algoritmo converge antes | Fase 1 §5 |
| 3 | "Tiempo (s)" como tiempo de convergencia | Es el entrenamiento completo con 32 procesos en paralelo | **Rechazar** como tiempo de convergencia; reportar episodios hasta estabilización | Fase 1 §6 |
| 4 | H4/H3: en browniano "no operan nunca" y "lo óptimo es no operar" | Con E0, 8/10 no operan. Con la configuración final, es una mezcla de efectivo y Buy & Hold (3/10 compran y mantienen); no se distingue de efectivo ni de B&H. El óptimo depende del criterio (Σ R frente a log-retorno) | **Matizar** | Fases 1 y 2 |
| 5 | H4: tabla de α en reversión (test, N = 5 000): α = 0,01 → +7,4 % | Con los datos de Marco, α = 0,01 da +11,6 (Q-Learning) y +11,9 (SARSA). Coinciden α = 0,005, α decreciente y α = 0,05 | **Pendiente**: no verificable (primera pasada sobrescrita) | Fase 2, Anexo A3 |
| 6 | H4: ruido de Q ≈ σ·√(α/2) ≈ 0,003 | Medido entre semillas con α = 0,05: ≈ 0,002; el ruido escala con exponente 0,09–0,45, no 0,5 | **Matizar**: parcialmente respaldado | Fase 2 §5.3 |
| 7 | H4/H6: α = 0,001 pierde "porque aprende demasiado despacio" | Falta α = 0,001 con N = 10 000; con α = 0,002 el rendimiento en reversión sigue subiendo con N | **Pendiente**: ni demostrado ni refutado | Fase 2 §5.2 |
| 8 | H6: γ "coincide con la escala temporal de cada mercado" y ε "gana casi siempre el decaimiento con f = 0,6" | En 3 de 8 pares la γ ganadora coincide con la de E1; en los otros 5, ningún efecto de γ excluye 0 en validación y en regímenes el test va en sentido contrario. El efecto de ε es detectable solo en SARSA-regímenes | **Matizar**: γ y ε no están justificados estadísticamente; solo α lo está | Fase 2 §7.1–7.2 |
| 9 | H7: "primera pasada ... regímenes +2,9 / +1,1" | Con α = 0,002 y N = 5 000 (artefactos de Marco): +1,0 y +1,3 | **Pendiente**: no verificable | Fase 2, Anexo A7 |
| 10 | H7/H11: en regímenes "el valor del agente es evitar pérdidas" | Correcto como matiz, pero un agente que **no opera** también supera a Buy & Hold (+21,9). Frente a efectivo no se detecta ventaja (+0,8 y +2,6, IC incluyen 0) | **Mantener el matiz y añadir** que superar a B&H aquí no es aprendizaje | Fases 1 y 2 |
| 11 | H11: la brecha val/test en regímenes se debe al *winner's curse* | No se reanalizó. La auditoría apuntó otra explicación (composición de las muestras) que **no está respaldada por archivos del repositorio** | **Pendiente** | Fase 1 §8 |
| 12 | D13/D12: la selección α → γ → ε elige lo mejor | En variantes casi empatadas (browniano: márgenes de 0,01 a 0,24 pp) la elección es ruido; el error estándar de validación (≈ 3 pp en reversión) supera a las diferencias de γ y ε | **Matizar** | Fase 1 §4.4; Fase 2 |
| 13 | D11: log-retorno como métrica principal | Se mantiene, pero el agente optimiza una recompensa distinta (suma de retornos simples) que sobreestima y cambia de signo en el browniano | **Mantener con salvedad** explícita | Fase 1 §4.4 |
| 14 | D7: "el notebook importa este módulo" | El notebook del repositorio es solo el enunciado | **Corregir** o aportar el notebook real | Auditoría C8 |
| 15 | H4 compara α "en test" | El diseño del grid se informó con el test; la selección posterior fue por validación | **Declarar** | Auditoría C10; Fase 2 A6 |

---

## 8. Experimentos que se ejecutaron realmente, cuándo y con qué configuración

| ID | Experimento | Autor | Cuándo | Configuración | Entrenamientos | Evidencia | Estado |
|---|---|---|---|---|---|---|---|
| M-1 | Señal por bin (H1–H3) | Marco | 27/10/2026 | 400 series, k = 10 pasos | — | `resultados/senal.pkl`, `figuras/fig_senales.png` | [ORIGINAL] |
| M-2 | Baselines aleatorio y Buy & Hold | Marco | 27/10/2026 | 100 series de test, semilla 7777 | — | `resultados/baselines.pkl` | [ORIGINAL] |
| M-3 | Primera pasada con la configuración por defecto (H4) | Marco | 27/10/2026 | α = 0,05, γ = 0,95, 2 000 episodios | Sin dato | **Solo texto** en `BITACORA.md`; sin artefacto | [ORIGINAL], no verificable |
| M-4 | Ajuste secuencial α → γ → ε | Marco | 27/10/2026 | N = 5 000; 5 semillas; α ∈ {0,001, 0,002, 0,005, 0,01, 0,05→0,001}; γ ∈ {0,9, 0,95, 0,99}; ε ∈ {decay f = 0,6, decay f = 0,3, 0,1 constante}; selección en validación | 200 + 120 + 120 | `resultados/alpha.pkl`, `gamma.pkl`, `epsilon.pkl`, `log_ajuste.txt` | [ORIGINAL] / [REUTILIZADO] |
| M-5 | Discretización | Marco | 27/10/2026 | 5, 10, 20 cuantiles y 10 uniformes sobre la configuración final | 160 | `resultados/discretizacion.pkl` | [ORIGINAL] / [REUTILIZADO] |
| A-0 | Auditoría en solo lectura | Dani | 28/10/2026 (mañana) | Sin entrenamientos | 0 | Informe en la conversación (no guardado como archivo) | [NUEVO] |
| F1-1 | Reanálisis estadístico, convergencia, estabilidad de políticas | Dani | 28/10/2026, 08:37–08:44 | Sobre los 600 entrenamientos de Marco | **0** | `INFORME_FASE1.md`, `resultados/`, `figuras/` | [NUEVO] sobre datos [REUTILIZADO] |
| F2-0 | Réplica de 40 corridas de Marco (`REP`) | Dani | 28/10/2026, 16:28–16:32 | Reversión y regímenes, α = 0,002 y 0,005, N = 5 000 | 40 | `fase2_evolucion/artefactos/REP.npz`, `resultados/verificacion_reproduccion_marco.csv` | [REPRODUCIDO] |
| F2-1 | E0 con N ∈ {500, 1000, 2000, 5000} (4 generadores) y N = 10 000 (reversión, regímenes) | Dani | 28/10/2026, 16:32–17:00 | `Config()` por defecto con N variable | 180 | `artefactos/E0_N*.npz` | [NUEVO] |
| F2-2 | Configuración final de Marco con N ∈ {500, 1000, 2000} y N = 10 000 (reversión, regímenes) | Dani | 28/10/2026, 16:32–17:00 | Ganadores de la etapa ε de Marco con N distinto; ε recalibrada por N | 140 | `artefactos/F_N*.npz` | [NUEVO] |
| F2-3 | α × N (α = 0,002 y 0,005 con N ∈ {500, 1000, 2000, 10000}) | Dani | 28/10/2026, 16:32–17:01 | Base de Marco, reversión y regímenes | 160 | `artefactos/alfa*.npz` | [NUEVO] |
| F2-4 | E0 con ε = 0,1 constante, N = 5 000 | Dani | 28/10/2026, 17:16–17:19 | α = 0,05, reversión y regímenes | 20 | `artefactos/E0eps0p1_N5000.npz` | [NUEVO] |
| F2-5 | Análisis, tablas, figuras e informe de Fase 2 | Dani | 28/10/2026, 17:25–17:28 | Sobre artefactos de Marco y de Fase 2 | 0 | `INFORME_FASE2.md` y `resultados/`, `figuras/` | [NUEVO] |

Las horas de la Fase 2 son las del registro (`registro_ejecucion.log`, hora local). Marco registra sus experimentos con fecha, no con hora, y no se conserva la hora exacta de cada uno; **no se inventan**.

---

## 9. Archivos y figuras que contienen las evidencias

Rutas relativas a la raíz del repositorio. Todo lo que cuelga de `analisis_adicional/` es [NUEVO].

**Documentos**

| Archivo | Contenido |
|---|---|
| `analisis_adicional/INFORME_FASE1.md` | Informe de la Fase 1 |
| `analisis_adicional/fase2_evolucion/INFORME_FASE2.md` | Informe de la Fase 2 (13 secciones + anexos A–E; Anexo A = discrepancias con la bitácora) |
| `analisis_adicional/CONTRIBUCIONES_DANI.md` | Este documento |
| `analisis_adicional/MATRIZ_EVIDENCIAS.md` | Matriz pregunta-hipótesis-experimento-resultado |
| `analisis_adicional/RESUMEN_PARA_EQUIPO.md` | Resumen breve para Marco y Cristian |
| `analisis_adicional/PROPUESTA_ENTREGA.md` | Propuesta del PDF de 2 páginas y de la defensa |
| `analisis_adicional/AUDITORIA_REPOSITORIO.md` | Estado del repositorio y propuesta de qué versionar |
| `analisis_adicional/README_REPRODUCCION.md` | Fase 4: entornos, orden de ejecución, verificación sin reentrenar y reentrenamiento opcional |
| `analisis_adicional/requirements_analisis.txt`, `requirements_entrenamiento.txt` | Fase 4: versiones realmente verificadas |

**Código**

| Archivo | Qué hace |
|---|---|
| `analisis_adicional/comun.py` | Carga restringida de pickles, réplica del entorno, reconstrucción de políticas, utilidades estadísticas |
| `analisis_adicional/reanalisar_resultados.py`, `analizar_convergencia.py`, `analizar_politicas.py` | Fase 1 |
| `analisis_adicional/fase2_evolucion/entrenar.py` | Entrenamiento con el código original de Marco, en el entorno aislado |
| `fase2_evolucion/utilidades.py`, `estadistica.py`, `analisis_fase2.py` | Plan de entrenamientos, comparaciones (reutilizan la Fase 1) y análisis |
| `fase2_evolucion/generar_tablas_informe.py`, `ensamblar_informe.py`, `informe_plantilla.md` | Generan `INFORME_FASE2.md` sin cifras transcritas a mano |
| `fase2_evolucion/manifiesto.py`, `verificar_fase2.py` | Huellas sha256, procedencia y verificaciones (en la Fase 4 se reescribió la comprobación de integridad por categorías con listas explícitas; ver `README_REPRODUCCION.md`) |

**Figuras** (las originales de Marco no se tocan)

| Figura | Qué muestra |
|---|---|
| `analisis_adicional/figuras/fig1_rendimientos_ic.png` | Rendimiento en test con IC y baselines |
| `figuras/fig2_qlearning_vs_sarsa.png`, `fig3_vs_efectivo_y_buyhold.png` | Comparaciones pareadas |
| `figuras/fig4_convergencia_sensibilidad.png`, `fig4b_convergencia_hiperparametros_identicos.png` | Convergencia con umbral variable |
| `figuras/fig5_recompensa_vs_rendimiento.png`, `fig6_politicas_por_semilla.png` | Recompensa frente a log-retorno; políticas por semilla |
| `fase2_evolucion/figuras/fig1_inicial_vs_optimizada.png` | E0 frente a E3 por mercado, con IC |
| `fig2_sensibilidad_alpha.png`, `fig2b_ruido_Q_y_alpha_por_N.png` | Sensibilidad a α, ruido de Q y α × N |
| `fig3_numero_de_episodios.png` | Efecto del número de episodios |
| `fig4_cadena_E0_E3.png` | Cadena E0 → E1 → E2 → E3 con el mismo N |
| `fig5_politica_reversion.png` | Política aprendida en reversión |

**Tablas y registros:** `fase2_evolucion/resultados/*.csv` (bloques 1–6), `tablas_informe.md`, `manifiesto_corridas.csv` (1 140 corridas: 600 de Marco y 540 de Fase 2, con origen, fecha y configuración), `manifiesto_procedencia.json`, `manifiesto_huellas_inicio.json` (sha256 de 51 archivos protegidos tomado **antes** de analizar), `manifiesto_integridad.json` (Fase 4: sha256 de artefactos, código y documentos añadidos, con historial), `verificacion_fase2.csv` (se regenera en cada verificación), `verificacion_fase2_ejecucion_28oct.csv` (la de la Fase 2, conservada), `verificacion_reproduccion_marco.csv`.

---

## 10. Qué aporta nuestro trabajo que no existía en la solución original

1. **Una medida correcta de la incertidumbre** que separa series de test y semillas de entrenamiento, y comparaciones pareadas por serie con corrección por comparaciones múltiples.
2. **El baseline de efectivo**, sin el cual no se pueden interpretar browniano y regímenes, y la advertencia de que superar a Buy & Hold en un mercado bajista no es aprendizaje.
3. **La evolución cuantificada** de la configuración inicial a la final con el mismo N. Separa lo que debe la mejora a α de lo que debe a la duración, γ y ε.
4. **El barrido del número de episodios** que pide el enunciado, con calendario de ε recalibrado por N.
5. **Un contraste del mecanismo de α**, que deja claro qué está demostrado (reducir α recupera la señal; no es cuestión de duración; "comprar" sí se explora) y qué no (la causa exacta).
6. **Reproducibilidad verificada:** 40/40 réplicas bit a bit, 560 entrenamientos reconstruidos desde Q, y entrenamiento reproducible en un entorno aislado sin tocar el código de Marco.
7. **Un registro explícito de discrepancias** entre bitácora y datos, sin corregirlas en silencio.

**Lo que este trabajo no aporta** (para no sobrevender): no hay nuevos algoritmos, generadores ni datos reales; no se ha hecho una confirmación independiente con test nuevo; no se hizo robustez a σ, κ ni `n_steps` con evidencia versionada; la rama BTC de Marco no se ha auditado a fondo.

**Pendientes con valor, por orden:** (1) evaluar las políticas guardadas sobre un test fresco y mayor, sin reentrenar; (2) α = 0,001 con N = 10 000 en reversión; (3) robustez a σ, κ y `n_steps` con IC; (4) ε constante en Q-Learning frente a SARSA (donde debería aparecer la diferencia on/off-policy); (5) convergencia en condiciones justas (mismos hiperparámetros, ≥ 50 series, ≥ 20 semillas).
