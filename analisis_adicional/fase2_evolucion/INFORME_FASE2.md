# Informe Fase 2 — Reconstrucción experimental de la evolución del agente RL

Registro independiente de investigación (rama `feature/analisis-estadistico-rl`, partiendo de `5a6e1ee`). Continúa el [informe de la Fase 1](../INFORME_FASE1.md).

**Estatus de la evidencia.**

* Todo lo marcado como **"Marco"** procede de los artefactos que él guardó el 27/10/2026 (`resultados/*.pkl`) y se reutiliza sin modificarlos.
* Todo lo marcado como **"Fase 2 (nuevo)"** se entrenó **el 28/10/2026** con el código original `trader_rl.py` en un entorno virtual aislado. **No son ejecuciones históricas** y no se presentan como tales.
* El test (100 series, `seed 7777`) ya fue inspeccionado por Marco. Esta fase es **exploratoria**: no es una confirmación independiente. La validación (50 series, `seed 2024`) es la única base para describir y para fundamentar decisiones; el test solo evalúa configuraciones fijadas de antemano.

---

## 1. Preguntas de investigación

| # | Pregunta |
|---|---|
| Q1 | ¿La configuración inicial (`Config()` por defecto) aprende una política razonable en los cuatro mercados? ¿Deja de operar en browniano, reversión y regímenes, como dice la bitácora (H4)? |
| Q2 | ¿Reducir α recupera la capacidad de explotar señales débiles? ¿Una α demasiado pequeña retrasa el aprendizaje? ¿Es "menos ruido en Q" la explicación, o hay otras? |
| Q3 | ¿Cuántos episodios necesita cada algoritmo para alcanzar una política suficientemente estable, con la configuración inicial y con la final? |
| Q4 | ¿Cada etapa de la cadena E0 → E1 (α) → E2 (γ) → E3 (ε) mejora el resultado de forma demostrable, una vez separado el efecto de la duración del entrenamiento (2 000 frente a 5 000 episodios)? |
| Q5 | ¿Q-Learning y SARSA difieren en alguna de estas etapas? |

## 2. Hipótesis (formuladas antes de ejecutar)

| # | Hipótesis | Procede de | Qué la refutaría |
|---|---|---|---|
| HA | Con α = 0,05 el agente no explota señales débiles (reversión, regímenes) | Bitácora H4 | E0 opera y gana en esos mercados |
| HB | El motivo es el ruido de Q ∝ √α | Bitácora H4 | La dispersión de Q entre semillas no crece con α o la relación señal/ruido no cae |
| HC | Alternativa 1: aprendizaje lento con α alto o pocos episodios | Auditoría | E0 mejora al aumentar N |
| HD | Alternativa 2: bloqueo — acciones que dejan de explorarse y el error ya no se corrige | Bitácora H4 | Todas las acciones se siguen actualizando y, aun así, el agente sigue sin operar; o forzar más exploración lo arregla |
| HE | Una α demasiado pequeña retrasa el aprendizaje (el óptimo es interior) | Bitácora H6 | α pequeña no mejora con N |
| HF | La mejora E0 → E3 se debe sobre todo a α, no a la duración del entrenamiento ni a γ o ε | Auditoría | La duración, o γ/ε, explican una parte detectable |
| HG | N = 5 000 basta en los mercados con señal | Decisión de Marco | El rendimiento o la política siguen cambiando entre 5 000 y 10 000 |

## 3. Configuraciones experimentales

**Entorno de ejecución.** Entorno virtual aislado fuera del repositorio (`C:\Users\dgall\.venvs\taller_rl_fase2`): Python 3.12.10, `gymnasium 1.4.0`, `numpy 2.5.4`. Solo se instaló `gymnasium` y sus dependencias. *(La ruta `C:\Users\dgall\...` es específica del equipo del autor y se conserva porque forma parte del registro de la ejecución; no hace falta para reproducir. En otro equipo basta cualquier entorno con las versiones de `requirements_entrenamiento.txt`; ver `../README_REPRODUCCION.md`.)* El análisis y las figuras usan el Python 3.14.3 del sistema (NumPy 2.4.2, Matplotlib 3.10.8). Los entrenamientos importan `trader_rl.py` **sin modificarlo**; la única adición es una subclase de `AgenteTabular` que cuenta las actualizaciones por (bin, posición, acción) sin consumir números aleatorios ni cambiar el aprendizaje.

**Puerta de reproducibilidad.** Antes de entrenar nada nuevo se reentrenaron **40 corridas de Marco** (grupo `REP`: reversión y regímenes, α = 0,002 y 0,005, N = 5 000, ambos algoritmos, 5 semillas) y se compararon con sus artefactos. **40 de 40 son idénticas bit a bit** (Q, curvas de validación, recompensas y balances por episodio, val y test): el entorno nuevo y el contador de visitas no alteran nada.

**Evaluación (sin cambios).** Validación: 50 series, `seed 2024`. Test: 100 series, `seed 7777`. 500 pasos por serie. Curva de validación: 10 primeras series cada 100 episodios. Semillas de entrenamiento 0–4, las de Marco.

**Qué es E0.** `Config()` del código: α = 0,05 constante, γ = 0,95, ε 1 → 0,01 con fracción 0,6, 2 000 episodios, 10 bins por cuantiles, 500 pasos. El script comprueba que coincide con los valores del enunciado de la fase. **La bitácora solo cita α = 0,05, γ = 0,95 y 2 000 episodios** (H4); el resto procede del código (ver anexo A).

**Entrenamientos nuevos** (todos con configuración fijada de antemano; no se selecciona nada con el test):

| Grupo | Qué es | Generadores | Entrenamientos |
|---|---|---|---|
| `E0_N500/1000/2000/5000` | E0 con distinto N (`N=5000` equivale a "α = 0,05 constante, 5 000 episodios") | 4 | 4 × 40 |
| `E0_N10000` | E0 con N = 10 000 | reversión, regímenes | 20 |
| `F_N500/1000/2000` | Configuración final de Marco (E3) reentrenada con otro N | 4 | 3 × 40 |
| `F_N10000` | E3 con N = 10 000 | reversión, regímenes | 20 |
| `alfa{0,002; 0,005}_N{500; 1000; 2000; 10000}` | α × N sobre la base de Marco | reversión, regímenes | 8 × 20 |
| `E0eps0p1_N5000` | E0 con ε = 0,1 constante (intervención sobre el bloqueo) | reversión, regímenes | 20 |
| `REP` | Réplica de corridas de Marco (verificación) | reversión, regímenes | 40 |
| **Total** | | | **540** (500 sin contar las réplicas) |

**Reutilizado de Marco (600 corridas, sin reentrenar):** etapas `alpha`, `gamma`, `epsilon` y `discretizacion`. La cadena E1/E2/E3 son los ganadores por validación de cada etapa, reconstruidos con la misma regla que `experimentos.ganadores`; `F@5000` es el artefacto de la etapa `epsilon`.

**Reducciones de alcance declaradas.**

* N = 10 000, α × N y la intervención con ε constante solo en reversión y regímenes (mercados de señal débil). Cíclico y browniano llegan a N = 5 000.
* α × N solo con α = 0,002 y 0,005 (α = 0,05 es E0). No se probó α = 0,001 con N = 10 000.
* No se reejecutó la cadena α → γ → ε ni el grid de Marco; γ y ε no se vuelven a ablacionar por separado (dependen del camino secuencial de Marco).
* No se reconstruyó la "primera pasada" histórica de Marco: sus artefactos fueron sobrescritos y no existen. Lo que se hace aquí es una **reconstrucción experimental actual** de ese punto de partida.

**Manifiesto.** `resultados/manifiesto_corridas.csv` lista las 1 140 corridas con su origen, fecha, fuente, configuración y semilla; `manifiesto_procedencia.json` resume la procedencia.

---

## 4. Resultados de la configuración inicial (Q1)

**Tabla T1 — E0 (configuración por defecto, N=2000). Validación = media ± SD entre 5 semillas; test con IC95 por series**

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

* **Reversión y regímenes: los dos algoritmos aprenden a no operar nunca** (10 de 10 corridas por mercado, 20 en total, con 0 operaciones en validación y test; rendimiento 0,0 pp = efectivo), tal como describe la bitácora (H4).
* **Browniano: 8 de 10 no operan.** Las dos que operan (semilla 4, ambos algoritmos) hacen ≈ 85 operaciones por serie y pierden −8,5 pp en test. La afirmación de la bitácora de que "no operan nunca" es válida en 8 de 10 corridas; no es una regla general.
* **Cíclico: la configuración inicial sí opera y gana** (+429 ± 129 pp en validación Q-Learning y +513 ± 74 SARSA), pero es **inestable**: 5 políticas distintas en 5 semillas, acuerdo medio entre semillas 0,73–0,79, y SD entre semillas de 74–129 pp frente a 1,6–4,5 pp de la configuración final.
* **Respuesta a Q1:** en los mercados con señal fuerte (cíclico) E0 aprende una política razonable pero ruidosa; en los de señal débil el resultado es "no hacer nada", que coincide con efectivo y no es una política aprendida.

---

## 5. Análisis de alpha (Q2)

### 5.1 Rendimiento frente a α (N = 5 000, base de Marco)

**Tabla T4 — Alpha (N=5000, base de Marco). Validación (pp): media ± SD entre semillas (operaciones por serie; semillas que nunca operan en val ni test)**

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

Diferencias emparejadas por serie en **validación** frente a α = 0,05 constante (IC95 por series; Q-Learning / SARSA):

* **Reversión:** α = 0,002 → +24,0 [+18,3, +29,5] / +20,6 [+15,0, +26,2]; α = 0,001 → +20,9 / +21,6; α = 0,005 → +19,5 / +16,2; α = 0,01 → +13,6 / +12,8. Todos excluyen 0.
* **Cíclico:** todas las α de la rejilla mejoran a 0,05 en +112 a +146 pp con IC que excluyen 0.
* **Regímenes:** solo α = 0,001 y 0,002 operan (QL +9,6 y +6,2; SARSA +3,8 y +3,4; todos con IC por series que excluyen 0). **α ≥ 0,005 no opera en ninguna de las 10 corridas** (el "+1,6 [+1,0, +2,2]" de QL frente a E0 es solo que E0 tiene una semilla que opera y pierde).
* **Browniano:** ninguna diferencia (todas las variantes ≈ efectivo).

**Respuesta a Q2 (primera parte): sí.** Reducir α recupera la capacidad de explotar señales débiles en reversión y, con α ≤ 0,002, en regímenes. La relación no es monótona: en reversión, el óptimo de Q-Learning es interior (α = 0,002), y el efecto es de varios puntos entre vecinos.

### 5.2 ¿Una α demasiado pequeña retrasa el aprendizaje?

**Tabla T7 — α × N (validación, pp, media ± SD; entre paréntesis semillas que nunca operan)**

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

* **En regímenes sí hay un retraso medible:** con α = 0,002 y Q-Learning, 5/5 semillas no operan con N = 500, 4/5 con N = 1 000, 2/5 con N = 2 000 y 0/5 solo con N = 10 000 (+7,8 ± 3,3 pp). α = 0,005 no opera a ningún N hasta 10 000.
* **En reversión el rendimiento mejora despacio con N** (Q-Learning α = 0,002: +18,9 con N = 500 → +22,3 con 5 000 → +23,9 con 10 000; SARSA no es monótono). No hay mínimo claro: α = 0,001 con N = 10 000 no se probó, así que la afirmación de la bitácora de que α = 0,001 pierde "porque con 5 000 episodios aprende demasiado despacio" **no queda demostrada ni refutada** aquí.
* **α = 0,05 no se arregla con más episodios:** 20 de 20 corridas no operan con N = 10 000; con N = 5 000 solo hay 1 semilla de Q-Learning por mercado que opera (y pierde), y ninguna de SARSA. Por tanto, para α = 0,05 el problema **no es la duración** (se rechaza HC para E0).

### 5.3 ¿Es "menos ruido en Q" la explicación? Contraste con otras hipótesis

**Tabla T5 — Ruido de Q entre semillas: pendiente log-log de SD(ΔQ) frente a α (teoría 1/2), SD(ΔQ) y cociente señal/ruido en α=0.002 y α=0.05**

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

**Tabla T6 — Visitas (N=5000, grupos instrumentados): estados sin compra en la política final, mediana de actualizaciones de 'comprar' tras ε≈0.01 en esos estados, fracción con < 50 y índice de aprendizaje α × actualizaciones totales de 'comprar'**

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

**Tabla T6b — Intervención sobre el bloqueo (N=5000, α=0.05): E0 con ε decreciente frente a ε=0.1 constante (validación)**

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

* **Ruido de Q (HB): parcialmente respaldado.** La dispersión entre semillas de Q(operar) − Q(mantener) crece con α pero **menos de lo que predice ∝ √α**: pendientes log-log de 0,09 a 0,45 (teoría 0,5). De α = 0,002 a 0,05 (×25, es decir ×5 en √α) la SD sube ×1,7 a ×3,6 en los tres mercados de señal débil y ×2,9 en cíclico SARSA; solo en cíclico Q-Learning sube más (×6,3). La relación señal/ruido cae de 2,0–5,8 a 0,9–1,8. Es coherente con ruido, pero no reproduce la ley √α.
* **Falta de exploración (HD, versión "comprar"): rechazada.** Con α = 0,05 la acción "comprar" recibe una mediana de 660–690 actualizaciones tardías por estado (ε ≈ 0,01); ningún estado tiene menos de 50.
* **Bloqueo en el lado "mantener dentro": patrón observado, causalidad no demostrada.** Con α = 0,05, la política final vende en el 98–100 % de los estados con posición, la acción "mantener" dentro recibe solo una **mediana de 3 a 6 actualizaciones tardías** y su Q es negativo (−0,003 a −0,004) por debajo de "vender" (−0,001); con α = 0,002 en reversión la política mantiene la posición en el 66–74 % de los estados y "mantener dentro" tiene Q positivo (+0,007 a +0,009). Es la firma que describe la bitácora, **pero es circular**: si la política vende siempre, "mantener" se visita poco por construcción. Lo que no se ha mostrado es que el error inicial fuera ruido.
* **Intervención (T6b):** forzar más exploración con ε = 0,1 constante **no recupera** la señal: la validación empeora (reversión −6,9 [−8,5, −5,2] QL y −8,7 [−10,2, −7,3] SARSA; regímenes −1,8 [−2,5, −1,2] QL y −3,7 [−4,5, −2,8] SARSA frente a E0), con 2–3 de 5 semillas sin operar y las que operan perdiendo. Es coherente con que a α = 0,05 la Q está dominada por ruido y las políticas que operan son, en la práctica, aleatorias. Refuta que "revisitar más" sea suficiente, aunque la intervención altera a la vez más cosas (ε constante).
* **Interacción con N (HC/HE):** descrita en 5.2.
* **Qué queda sin contrastar:** el mecanismo causal (ruido inicial que sesga Q hacia "vender" y se auto-perpetúa) requeriría registrar Q durante el entrenamiento o inicializar de forma optimista; no se hizo.
* **Q-Learning frente a SARSA en α:** reversión y cíclico se comportan igual. En regímenes, Q-Learning opera con más semillas (α = 0,001: 0/5 sin operar frente a 3/5 de SARSA), pero con 5 semillas no se puede afirmar una diferencia.

Figuras: [`fig2_sensibilidad_alpha.png`](figuras/fig2_sensibilidad_alpha.png), [`fig2b_ruido_Q_y_alpha_por_N.png`](figuras/fig2b_ruido_Q_y_alpha_por_N.png).

---

## 6. Sensibilidad al número de episodios (Q3)

Todas las configuraciones se reentrenaron con su propio calendario de ε (que depende de N); no se truncaron curvas.

**Tabla T8 — Número de episodios (validación, pp, media ± SD entre semillas) y N 'suficiente' empírico (cumple C1-C4 en N y en todos los mayores; ver informe)**

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

**Criterios de "política suficientemente estable"** (todos en validación; `resultados/bloque3_n_suficiente.csv` da cada uno por separado). N es suficiente si cumple el criterio en N y en todos los N mayores de la rejilla:

* **C1** rendimiento no peor, de forma detectable, que con el N máximo (IC95 por series de la diferencia emparejada).
* **C2** acuerdo ≥ 0,9 con la política del N máximo, ponderado por visitas.
* **C3** acuerdo ≥ 0,9 con la política del siguiente N (la política deja de cambiar al alargar).
* **C4** acuerdo ≥ 0,9 entre semillas.

**Resultados (configuración final E3; N máximo = 10 000 en reversión/regímenes y 5 000 en cíclico/browniano):**

* **Cíclico:** el rendimiento a N = 2 000 ya está en −3,4 [−17,5, +10,7] pp de N = 5 000 (Q-Learning); SARSA oscila (−16,0 a N = 2 000). N suficiente 2 000 (QL) y 5 000 (SARSA) con los cuatro criterios.
* **Reversión:** la **política** se parece a la de 10 000 desde N = 500–1 000 (acuerdo ponderado 0,90–0,95), pero el **rendimiento** de Q-Learning sigue subiendo (−5,0 pp a N = 500, −4,1 a N = 2 000, −1,6 [−3,4, +0,2] a N = 5 000). N suficiente = 5 000 en ambos algoritmos.
* **Regímenes:** **no opera con N ≤ 1 000** (Q-Learning, 0 operaciones). Con N = 2 000 ya hay política (+6,3 / +4,1), a N = 5 000 el rendimiento está dentro del ruido de N = 10 000 (−0,4 [−2,4, +1,6] QL; −0,2 [−3,7, +3,2] SARSA). Criterio conjunto: 10 000 (QL) y > 10 000 (SARSA, por falta de coherencia entre semillas, C4).
* **Browniano:** no hay rendimiento que estabilizar (≈ efectivo); la etiqueta ">5 000" refleja ruido entre semillas, no aprendizaje pendiente.
* **E0:** las "N suficiente" de E0 (500 en SARSA reversión/regímenes) son **estabilidad sin calidad**: la política "no operar" es perfectamente estable.

**Respuesta a Q3 / HG.** Con la configuración final, **N = 5 000 es razonable en reversión y regímenes** (no hay cambio detectable respecto a 10 000) y **2 000 basta en cíclico**; en regímenes N ≥ 2 000 es imprescindible. Por debajo, los resultados degradan o no hay política. La política "se asienta" (acuerdo ≥ 0,9) mucho antes que el rendimiento.

**Convergencia empírica frente a matemática.** Todo lo anterior es convergencia empírica de la política greedy y de su rendimiento sobre 50 series. No es convergencia matemática: con α constante, la tabla Q de Q-Learning o SARSA no converge a un punto; fluctúa en un entorno del óptimo, de ancho creciente con α (ver 5.3). Y las pruebas de "estable" tienen poca potencia con 5 semillas.

Figura: [`fig3_numero_de_episodios.png`](figuras/fig3_numero_de_episodios.png).

---

## 7. Reconstrucción de la evolución completa (Q4)

### 7.1 Tabla etapa → decisión

| Etapa | Problema observado | Hipótesis | Experimento | Resultado | Decisión |
|---|---|---|---|---|---|
| **E0** Config. por defecto (α = 0,05, γ = 0,95, ε 1 → 0,01, 2 000 ep.) *(reconstruido ahora)* | No opera en 3 de 4 mercados | α = 0,05 es demasiado grande para señales débiles | 40 entrenamientos nuevos de E0 | Reversión y regímenes: 10/10 por mercado no operan (= efectivo). Browniano: 8/10. Cíclico opera pero inestable (val. +429 ± 129 QL, +513 ± 74 SARSA) | Investigar α |
| **E0 + más episodios** *(nuevo)* | ¿Es cuestión de duración? | Más episodios lo arreglan | E0 con N = 500 a 10 000 | Con N = 10 000, 20/20 no operan; con N = 5 000, 1 semilla de Q-Learning por mercado opera y pierde | **No es la duración** |
| **E1** α *(Marco, histórico; contraste nuevo)* | α alta apaga el aprendizaje | α menor recupera la señal | 200 entrenamientos de Marco (reutilizados) + α = 0,05 constante (nuevo) | Val. frente a E0 (N = 5 000): reversión +24,0 [+18,4, +29,7] QL y +21,6 [+16,4, +27,0] SARSA; cíclico +145 / +146; regímenes +9,6 [+4,2, +15,2] / +3,8 [+1,6, +6,1]; browniano 0 | **Mantener.** Mejora demostrada en reversión y cíclico; en regímenes solo con α ≤ 0,002 y pocas semillas |
| **Mecanismo de E1** *(nuevo)* | ¿Por qué α alta bloquea? | Ruido de Q; falta de exploración; aprendizaje lento | Dispersión de Q, visitas, α × N, intervención ε = 0,1 | Ruido: crece menos que √α; "comprar" sí se explora; "mantener dentro" casi no se revisita (3–6 act.); ε = 0,1 no lo arregla; más N tampoco | Hipótesis de ruido **parcialmente** respaldada; mecanismo causal **no demostrado** |
| **E2** γ *(Marco)* | ¿γ acorde a la escala temporal de cada mercado? | γ alto en regímenes (horizonte ≈ 100), medio en reversión | Etapa `gamma` de Marco | Misma γ que E1 en 3 de 8 pares. Resto, val.: +0,3 / +0,5 / +0,9 (browniano, regímenes) y +5,9 [−2,5, +14,6] (cíclico SARSA): ninguno excluye 0. En regímenes el test va en sentido contrario (−2,0 [−3,3, −0,7] QL; −2,1 [−4,0, −0,2] SARSA) | **No justificada estadísticamente** |
| **E3** ε *(Marco)* | ¿Decaimiento (f = 0,3 o 0,6) frente a constante? | El decaimiento evita pagar comisiones aleatorias | Etapa `epsilon` de Marco | Solo SARSA en regímenes: val. +2,2 [+0,2, +4,3] y test +3,4 [+1,9, +4,9]. El resto, sin cambio o dentro del ruido | Mantener; efecto **pequeño y localizado** |
| **Episodios** *(nuevo)* | Marco fijó 5 000 | 5 000 basta | E0 y E3 con N = 500 a 10 000 | E3: reversión y regímenes sin cambio detectable entre 5 000 y 10 000; regímenes no opera con N ≤ 1 000; cíclico OK con 2 000 | **N = 5 000 razonable** |
| **E4** discretización *(Marco; experimento posterior de sensibilidad, no etapa secuencial)* | ¿10 bins por cuantiles? | Uniformes y 5 bins pierden información | Etapa `discretizacion` de Marco | Uniformes pierden en los 3 mercados con señal (−13 a −15 pp en reversión; −43 a −52 en cíclico; −7 a −9 en regímenes); 5 cuantiles pierden en regímenes y cíclico QL; 20 ≈ 10, salvo cíclico QL (−10,2 [−19,2, −1,4]) | **Mantener 10 cuantiles** |

La bitácora cita estas decisiones (D1, D10–D16), pero la comparación "inicial frente a final" no estaba cuantificada.

### 7.2 Cadena controlada con el mismo N = 5 000

La comparación histórica (E0 con 2 000 frente a E1–E3 con 5 000) **mezcla** duración e hiperparámetros. Aquí todas las etapas tienen N = 5 000 (E0 es `E0_N5000`, nuevo). Etapas sin cambio de configuración respecto a la anterior se marcan "(=)".

**Tabla T9 — Cadena en VALIDACIÓN (pp, media ± SD semillas). '(=)' = la etapa no cambió la configuración respecto a la anterior (corridas idénticas)**

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

**Tabla T9b — Cadena en TEST (pp, IC95 población; configuraciones ya fijadas)**

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

**Tabla T9c — Incremento de cada etapa respecto a la anterior (pp; IC95 por series). La validación fundamenta la decisión; el test sólo la evalúa**

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

**Lectura.** Casi todo el salto está en E1 (α). Lo que añaden γ y ε es pequeño y, en la mayoría de los pares, no excluye 0. La figura 4 muestra puntos por semilla y media; no es una mejora monótona (p. ej., en regímenes el test baja en E2 y sube en E3, y en browniano el test es ruido).

Figura: [`fig4_cadena_E0_E3.png`](figuras/fig4_cadena_E0_E3.png).

### 7.3 Duración frente a hiperparámetros

**Tabla T3 — Descomposición en TEST (pp): diferencia [IC95 de población] (% de series donde mejora). Primeras 2 columnas = efecto de los hiperparámetros; siguientes 2 = efecto de la duración; última = lo que mezcla la comparación histórica**

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

* **Hiperparámetros a igual N:** reversión +23,7 [+16,2, +31,2] (QL) y +21,4 [+16,5, +26,4] (SARSA) a N = 5 000, y +20,4 y +18,0 a N = 2 000. **Robustas** tras Holm.
* **Duración con los mismos hiperparámetros:** ningún IC de población excluye 0, ni con E0 ni con E3 (p. ej. reversión E3: +1,3 [−0,6, +3,1] QL y +3,5 [−0,6, +7,5] SARSA).
* **Cíclico:** la mejora es grande (+147 y +153 a N = 5 000) pero con IC amplios porque E0 es muy variable; es un indicio (no pasa Holm).
* **Regímenes y browniano:** diferencias no concluyentes.
* **TOTAL histórico (E3(5 000) − E0(2 000)):** en reversión (+21,6 QL y +21,4 SARSA) es casi idéntico al efecto de los hiperparámetros a N = 5 000, de modo que **la duración no explica la mejora**; en cíclico es mayor (+214 y +124) por la variabilidad de E0 con N = 2 000.

**Tabla T3b — Familia declarada de 24 contrastes en test (Holm). Exploratoria: el test ya fue inspeccionado por Marco**

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

### 7.4 E4 — discretización

**Tabla T10 — E4: discretización (experimento de sensibilidad posterior, en validación; Δ = diferencia con 10 cuantiles, IC95 por series)**

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

### 7.5 Qué es histórico, qué reconstruido, qué nuevo

| | Contenido | Evidencia |
|---|---|---|
| Histórico (Marco, documentado) | Etapas α, γ, ε y discretización; decisiones D1–D16; texto de H4–H11 | Artefactos `alpha/gamma/epsilon/discretizacion.pkl` y bitácora. **No hay historial de commits** que muestre la secuencia (un solo commit) |
| Texto sin artefacto | Primera pasada con la configuración por defecto (H4, "primera pasada" de H7) | **Solo texto**: sus resultados fueron sobrescritos |
| Reconstrucción experimental (esta fase) | E0 y su contraste con la solución final | 540 entrenamientos del 28/10/2026 |
| Nuevos experimentos de sensibilidad | N, α × N, visitas, ε = 0,1 constante, QL frente a SARSA por etapa | Ídem |

Figura: [`fig1_inicial_vs_optimizada.png`](figuras/fig1_inicial_vs_optimizada.png) y [`fig5_politica_reversion.png`](figuras/fig5_politica_reversion.png).

---

## 8. Comparación Q-Learning y SARSA (Q5)

**Tabla T11 — Q-Learning − SARSA en TEST (pp; IC95 de población) y operaciones (Q/S) por configuración**

| Generador | E0 (N=2000) | E0 (N=5000) | E1 | E3 final |
|---|---|---|---|---|
| Cíclico | −91,4 [−277,6, +94,8]; ops 117/86 | +4,2 [−187,2, +195,7]; ops 87/96 | +9,3 [−2,6, +21,1]; ops 62/63 | −1,1 [−13,3, +11,0]; ops 62/53 |
| Browniano | +0,0 [−6,6, +6,6]; ops 17/17 | +0,0 [+0,0, +0,0]; ops 0/0 | −1,1 [−4,3, +2,0]; ops 0/0 | −1,4 [−5,3, +2,4]; ops 0/0 |
| Reversión | +0,0 [+0,0, +0,0]; ops 0/0 | −2,0 [−7,7, +3,6]; ops 17/0 | +0,3 [−0,6, +1,1]; ops 52/50 | +0,2 [−1,8, +2,2]; ops 52/45 |
| Regímenes | +0,0 [+0,0, +0,0]; ops 0/0 | −1,4 [−5,3, +2,5]; ops 16/0 | +1,7 [−2,0, +5,3]; ops 39/16 | −1,7 [−6,7, +3,3]; ops 33/28 |

* **No hay una diferencia estable.** En test, el IC de población de Q-Learning − SARSA incluye 0 en las cuatro configuraciones y los cuatro mercados.
* **Signo cambiante en E0 cíclico:** a N = 2 000 SARSA gana en las series (−91,4 pp; IC series [−100,5, −82,0], pero IC de población [−277,6, +94,8]); a N = 5 000 el signo se invierte (+4,2). Con 5 semillas y E0 tan variable es un artefacto del muestreo, no una propiedad del algoritmo.
* **Operaciones:** Q-Learning opera algo más que SARSA en las configuraciones con política (cíclico 117 frente a 86 con E0 a N = 2 000; 62 frente a 53 con E3; reversión 52 frente a 45; regímenes 33 frente a 28). Consistente con la Fase 1, pero con IC amplios.
* **Convergencia por N:** las dos curvas son similares (T8). No se detecta que uno necesite menos episodios. SARSA es menos monótono en reversión.
* **Conclusión:** no declarar superior a ninguno. Q-Learning y SARSA se comportan igual que en la Fase 1.

---

## 9. Interpretación financiera

**Tabla T2 — Inicial frente a optimizada: log-retorno medio en test (pp) con IC95 de población (series + semillas, t 4 g.l.). Efectivo = 0**

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

**Tabla T12 — Frente a Buy & Hold en TEST (pp; IC95 población; % de series donde el agente gana)**

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

* **E0 = efectivo en reversión y regímenes.** Y **ese "agente que no hace nada" ya supera a Buy & Hold en regímenes** (+21,9 [+8,6, +35,2] pp, 68 % de las series). Marco obtiene +22,7 [+11,5, +34,0] (QL) y +24,5 [+13,1, +35,8] (SARSA) con E3: la diferencia entre E0 y E3 en esa comparación es de 0,8 a 2,6 pp, no significativa. **Superar a Buy & Hold en un mercado bajista no es evidencia de aprendizaje.** En reversión, E0 no supera a B&H (+1,1 [−1,0, +3,1]) y E3 sí (+22,7 [+17,3, +28,1]).
* **Reversión:** de 0 % (efectivo) a +21,6 % de log-retorno medio; es el único mercado donde la evolución produce una ganancia absoluta robusta.
* **Cíclico:** incluso E0 supera a B&H (SARSA a N = 2 000 +150,0 [+46,6, +253,4]; QL a N = 5 000 +125,5 [−5,9, +256,8]), por ser un generador con señal muy fuerte. E3 lo supera de forma robusta (+272,9 [+230,3, +315,4]).
* **Browniano:** ninguna configuración supera a efectivo o a B&H de forma distinguible.
* Se mantiene la salvedad de la Fase 1 sobre el cíclico: su +610 % es una característica del generador y del horizonte de 500 pasos.

---

## 10. Incertidumbre estadística

* Todas las comparaciones usan **las mismas series** (val y test), emparejadas por serie, con las funciones de la Fase 1 (IC por series: bootstrap percentil; IC de población: efectos cruzados, t de 4 g.l.).
* Con 5 semillas el IC de población es aproximado. Donde las políticas son casi deterministas (reversión), el IC por series y el de población coinciden; en E0 cíclico, el de población es mucho más amplio (±150–190 pp) porque las semillas difieren.
* **Familia declarada (3 contrastes × 8 pares = 24 en test)** con Holm: reversión es robusta en los 6 contrastes (los 3 de Q-Learning y los 3 de SARSA); cíclico, regímenes y browniano no superan Holm. Ver T3b.
* Todo lo demás es **exploratorio** (cientos de comparaciones en las tablas). No se interpreta un IC que incluye 0 como equivalencia.

---

## 11. Limitaciones

1. **El test ya fue inspeccionado** por Marco; esta fase no es confirmación independiente. Mismas 100 series en todas las comparaciones.
2. **Solo 5 semillas** (las de Marco). Pocas para estimar la variación entre semillas y para detectar bloqueos intermitentes.
3. **Recorte de alcance**: N = 10 000, α × N e intervención ε solo en reversión y regímenes; α × N con dos valores de α; sin α = 0,001 con N = 10 000.
4. **Cadena secuencial de Marco**: E2 y E3 dependen del camino (α → γ → ε) y no se reablacionan por separado; no se exploran interacciones.
5. **Mecanismo causal de α no demostrado** (ver 5.3); los recuentos de visitas son consecuencia de la política (circulares).
6. **E0 es una reconstrucción**: no hay artefactos de la primera pasada real ni se pueden verificar sus cifras (H4, H7).
7. **Tiempos** medidos con 16 procesos en paralelo: no son tiempos de convergencia ni son comparables con los de Marco (32 procesos).
8. **Entorno distinto al de Marco** (NumPy 2.5.4 y Python 3.12): irrelevante para los resultados (40/40 réplicas idénticas), pero no se pudo comprobar su entorno original.
9. **Criterios de "N suficiente"** con umbrales arbitrarios (0,9) y baja potencia; deben leerse junto con las tablas, no como veredicto.

---

## 12. Conclusiones

1. **El punto de partida existe y es reproducible:** con `Config()` por defecto, reversión y regímenes no operan (10/10 por mercado) y el resultado equivale a efectivo; browniano no opera en 8/10; cíclico opera pero es inestable.
2. **La mejora de E0 a la solución final se debe principalmente a α, no a la duración ni a γ/ε.** Reversión +23,7 [+16,2, +31,2] (QL) y +21,4 [+16,5, +26,4] (SARSA) con el mismo N, robustas; la duración no aporta de forma detectable.
3. **γ y ε aportan poco y no de forma demostrable en general**; solo SARSA en regímenes muestra un efecto de ε detectable. La bitácora presenta γ y ε como decisiones justificadas; los datos solo respaldan α.
4. **La explicación "ruido de Q ∝ √α" está parcialmente respaldada** (el ruido crece, pero menos que √α) y el bloqueo en la acción "mantener dentro" es un patrón observado, no un mecanismo demostrado; "falta de exploración" y "falta de episodios" quedan rechazadas como explicaciones únicas.
5. **N = 5 000 es razonable** para la configuración final; 2 000 basta en cíclico y N ≥ 2 000 es imprescindible en regímenes. La política se asienta antes que el rendimiento.
6. **No hay diferencia estable entre Q-Learning y SARSA** en ninguna etapa.
7. **Un agente que no hace nada ya "supera" a Buy & Hold en regímenes**; esa comparación no demuestra aprendizaje.

---

## 13. Qué aporta esta fase respecto al trabajo de Marco

* Convierte la historia de la bitácora (hasta ahora texto) en una **cadena cuantificada con el mismo N**, que separa el efecto de la duración del de los hiperparámetros.
* **Reproduce el punto de partida** (E0) y contrasta H4: confirma "no opera" en reversión/regímenes (10/10 por mercado) y lo **matiza** en browniano (8/10).
* Aporta un **análisis de N** que el enunciado exige y Marco no hizo, con calendario de ε recalibrado por N.
* **Contrasta el mecanismo de α** con medidas que la bitácora no tenía (ruido de Q, visitas, α × N, intervención) y deja claro qué queda demostrado y qué no.
* **Verifica la reproducibilidad** de los artefactos de Marco (40/40 idénticos) y deja el entrenamiento del código original reproducible en un entorno aislado.
* **Cuestiona** la justificación de γ y ε, y la lectura de "supera a Buy & Hold" en regímenes.

---

## Anexo A — Discrepancias con la bitácora (registradas, no corregidas)

| # | Bitácora | Hallazgo | Estado |
|---|---|---|---|
| A1 | H4: configuración inicial "α = 0,05, γ = 0,95, 2 000 episodios" | El resto (ε 1 → 0,01 con fracción 0,6, 10 cuantiles, 500 pasos) no se menciona en la bitácora; procede de `Config()` | Documentado |
| A2 | H4: "En browniano, reversión y regímenes los dos algoritmos aprenden a no operar nunca" | Cierto en reversión y regímenes (10/10). En browniano, 8/10; las 2 restantes operan ≈ 85 veces y pierden −8,5 pp | Matizado |
| A3 | H4, tabla de α en reversión (test, 5 000 episodios): α = 0,05 → 0 %; 0,01 → +7,4 %; 0,005 → +16,9 %; 0,05 → 0,001 → +16,5 % | Datos actuales de Marco (test): α = 0,01 → +11,6 (QL) / +11,9 (SARSA); α = 0,005 → +17,2 / +16,4; decreciente → +16,5 / +13,9; α = 0,05 constante (nuevo) → −2,0 / 0,0. Coinciden α = 0,005, decreciente (QL) y α = 0,05; **no coincide α = 0,01** | No verificable (primera pasada sobrescrita) |
| A4 | H4: "ruido de Q ≈ σ·√(α/2) ≈ 0,003" | Dispersión medida entre semillas de ΔQ en α = 0,05: ≈ 0,002 en reversión y regímenes; escala con exponente 0,09–0,45, no 0,5 | Matizado |
| A5 | H4/H6: α = 0,001 pierde "porque aprende demasiado despacio" | No contrastado: falta α = 0,001 con N = 10 000. Con α = 0,002 el rendimiento en reversión sigue subiendo con N | Pendiente |
| A6 | H4 usa **test** para comparar los valores de α | La selección posterior fue por validación, pero el diseño del grid se informó con test (Fase 1) | Ya registrado |
| A7 | H7: "primera pasada del ajuste (α ≥ 0,002): browniano 0/0, regímenes +2,9/+1,1" | Con α = 0,002 y N = 5 000 (artefactos de Marco): regímenes +1,0 (QL) y +1,3 (SARSA). No coincide con +2,9/+1,1 | No verificable |
| A8 | D7: "el notebook importa este módulo" | El notebook committed es el enunciado sin código del alumno (Fase 1) | Ya registrado |
| A9 | H9: "SARSA converge antes en reversión (1 400 frente a 2 500)" | La tabla dice 800; la comparación no es justa (Fase 1) | Ya registrado |

## Anexo B — Procedencia y registro de ejecución

* `registro_ejecucion.log`: inicio y fin de cada grupo, versiones y duración. El entrenamiento nuevo duró ≈ 35 min de pared con 16 procesos.
* `manifiesto_huellas_inicio.json`: sha256 de 51 archivos protegidos (código y resultados de Marco, bitácora, notebook, `analisis_adicional/` de la Fase 1), tomado **antes** de analizar.
* `manifiesto_corridas.csv` y `manifiesto_procedencia.json`: una fila por entrenamiento (Marco 600, Fase 2 540).

## Anexo C — Verificaciones

> **Nota de la Fase 4.** Si esta tabla no incluye las filas V4a–V4e, es la de la ejecución original de la Fase 2 (28/10/2026), conservada tal cual en `resultados/verificacion_fase2_ejecucion_28oct.csv`. En la Fase 4 se reescribieron V4–V6 (ver `README_REPRODUCCION.md`); `resultados/verificacion_fase2.csv` se regenera en cada ejecución de `verificar_fase2.py`.

| Comprobación | Elementos | Detalle | Resultado |
|---|---|---|---|
| V1 política reconstruida de Q == val/test guardados (artefactos nuevos) | 540 | max|dif|=0; 0 corridas con empates exactos en Q excluidas | OK |
| V2 réplica de 40 corridas de Marco (Q, curvas, recompensas, balances, val, test) idéntica bit a bit | 40 | 40/40 idénticas | OK |
| V3 series de test regeneradas (seed 7777, 100) == baselines.pkl de Marco | 400 | max|dif|=0 | OK |
| V4 archivos protegidos sin cambios (sha256 frente a la instantánea de inicio) | 51 | modificados=0, añadidos=0 | OK |
| V5 reejecutar entrenar.py sobre un grupo existente no sobrescribe el artefacto | 1 | salida: [saltado] REP: ya existe artefacto (no se sobrescribe sin --forzar) | OK |
| V6 git: sin archivos rastreados modificados; rama y HEAD | 1 | rama=feature/analisis-estadistico-rl, HEAD=5a6e1ee (inicio 5a6e1ee), rastreados modificados=ninguno | OK |

## Anexo D — Cómo reproducir

> **Nota de la Fase 4.** Los comandos de abajo son los **ejecutados el 28/10/2026 en el equipo del autor**, con rutas de su carpeta de usuario (`C:\Users\dgall\...`). Son registro histórico, no instrucciones portables. Para reproducir en otro equipo, con rutas relativas, entornos y orden de ejecución, usa `analisis_adicional/README_REPRODUCCION.md`.

```
# Entrenamiento (entorno aislado con gymnasium; no sobrescribe artefactos existentes)
C:\Users\dgall\.venvs\taller_rl_fase2\Scripts\python.exe -B -X utf8 analisis_adicional\fase2_evolucion\entrenar.py --listar
C:\Users\dgall\.venvs\taller_rl_fase2\Scripts\python.exe -B -X utf8 analisis_adicional\fase2_evolucion\entrenar.py --grupos REP      # puerta de reproducibilidad
C:\Users\dgall\.venvs\taller_rl_fase2\Scripts\python.exe -B -X utf8 analisis_adicional\fase2_evolucion\entrenar.py --grupos todos

# Análisis, figuras, tablas e informe (Python del sistema: numpy + matplotlib)
py -3.14 -B -X utf8 analisis_adicional\fase2_evolucion\manifiesto.py --huellas        # una sola vez, antes de analizar
py -3.14 -B -X utf8 analisis_adicional\fase2_evolucion\analisis_fase2.py
py -3.14 -B -X utf8 analisis_adicional\fase2_evolucion\generar_tablas_informe.py
py -3.14 -B -X utf8 analisis_adicional\fase2_evolucion\ensamblar_informe.py
py -3.14 -B -X utf8 analisis_adicional\fase2_evolucion\manifiesto.py --procedencia
py -3.14 -B -X utf8 analisis_adicional\fase2_evolucion\verificar_fase2.py
```

## Anexo E — Archivos

* `utilidades.py` (plan de entrenamientos, carga/guardado), `entrenar.py` (entrenamiento con código original), `estadistica.py` (comparaciones, reutiliza la Fase 1), `analisis_fase2.py` (bloques 1–6 y figuras), `generar_tablas_informe.py`, `ensamblar_informe.py`, `manifiesto.py`, `verificar_fase2.py`.
* `artefactos/*.npz` (19 grupos nuevos, sin pickle), `resultados/*.csv`, `figuras/fig1…fig5`, `INFORME_FASE2.md` (generado desde `informe_plantilla.md` y `resultados/tablas_informe.md`).
