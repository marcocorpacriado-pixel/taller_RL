# BITÁCORA — Taller RL: Agente Trader

## 0. Requisitos reales (verificados contra `taller_trader.ipynb`, 27/10/2026)

**Entrega:** PDF de **máx. 2 páginas** (pruebas, resultados, conclusiones). Fecha tope 29/10/2026 23:59.
Primera versión por e-mail a luis.lago@uam.es antes de la clase del jueves 29/10 (presentación de 15 min con el profesor).
Un representante del grupo debe enviar un e-mail al profesor **antes del martes 27/10** para organizar el orden.

**Entorno `SimpleTradingEnv` (dado, no se modifica):**
- Acciones: 0 = vender, 1 = mantener, 2 = comprar. Comprar con posición o vender sin ella = mantener.
- Observación: `(tendencia, posición)`. Tendencia = `(P_t − media(P_{t-5..t})) / media(...)` → media de **6** precios incluyendo el actual, recortada a [-1, 1].
- Capital 1000 $, todo dentro / todo fuera, comisión 0.1 % por operación.
- Recompensa: `−0.001` al operar + **retorno porcentual** `(P_{t+1} − P_t)/P_t` si se está en posición (no ΔP absoluto).
- La operación se ejecuta al precio `P_t` (el que ve la observación) → no hay look-ahead.

**Generadores (son 4, no 3):**
1. Tendencia suave + componente cíclica (ruido σ = 10 %/paso; el periodo del ciclo depende de `n_steps`).
2. Browniano geométrico (μ = 0.0001, σ = 0.02).
3. Reversión a la media (OU discreto, media 100, κ = 0.05).
4. Regímenes alcista/bajista (p cambio = 0.02).

**Mínimos exigidos:**
- Q-Learning y SARSA en los 4 generadores.
- Muchos episodios, cada uno con una serie nueva **del mismo generador** (un agente por generador).
- Test con series nuevas del mismo generador, resultados promediados sobre varias series.
- Comparar con agente aleatorio y Buy & Hold. Razonar si la estrategia es la esperada.
- Ajustar α, ε (probablemente con decaimiento), γ y nº de episodios.
- Tabla comparativa (algoritmos + baselines × generador) con rendimiento **y tiempo de convergencia**.

**Evaluación:** 50 % mínimos · 10 % ir más allá · 20 % presentación · 10 % justificación de decisiones · 10 % análisis crítico.

## 1. Hallazgos empíricos

- **H1 — Rango de la tendencia** (200 series × 500 pasos, percentiles 1 % / 99 %):
  cíclico ±0.31–0.38 · browniano ±0.06 · reversión ±0.06 · regímenes ±0.06.
  ⇒ 10 bins uniformes en [-1, 1] meterían casi todos los estados en 2 bins en 3 de los 4 generadores.
- **H2 — Buy & Hold por generador** (media / mediana del ratio final/inicial): cíclico 570× / 27× · browniano 1.09 / 1.02 · reversión 1.00 / 0.99 · regímenes 0.97 / 0.88.

- **H3 — Señal disponible por bin.** Log-retorno futuro medio condicionado al bin de tendencia (400 series, ×1000; coste de ida y vuelta = 2.0):

  | Generador | k = 10 pasos, bin 0 → bin 9 | Lectura |
  |---|---|---|
  | Cíclico | −199 … +324 | Momentum enorme: comprar en bins altos, vender en bajos |
  | Browniano | ≈ −1 ± 0.45 en todos | **Sin señal**: lo óptimo es no operar |
  | Reversión | +18.0 … −10.0 | Reversión: comprar tras caídas fuertes (bin 0, +1.8 % ≫ 0.2 % de coste), vender tras subidas |
  | Regímenes | −14.3, ≈ 0 en el centro, −11.9 | Ambos extremos son negativos: un movimiento fuerte delata volatilidad alta, es decir, régimen bajista. Ventaja pequeña |

  ⇒ Esta tabla es la "verdad de referencia" con la que juzgar si la política aprendida tiene sentido.
- **H4 — Primera pasada** (config. por defecto: α = 0.05, γ = 0.95, 2000 episodios). En browniano, reversión y regímenes los dos algoritmos aprenden a **no operar nunca**. En el browniano es correcto (H3); en reversión no lo es.
  Causa: con α = 0.05 el ruido de Q es ≈ σ·√(α/2) ≈ 0.003, mayor que la ventaja por paso. Cuando el ruido hace que una acción parezca mala, la política greedy deja de visitarla y ese error **ya no se corrige**; cuando la hace parecer buena, sí se corrige. El resultado es un sesgo sistemático hacia "no operar".
  Comprobado en reversión con 5000 episodios (log-retorno medio en test):

  | α | Log-retorno | Política aprendida |
  |---|---|---|
  | 0.05 | 0 % | no opera |
  | 0.01 | +7.4 % | |
  | 0.005 | +16.9 % | compra en el bin 0 y vende en bins altos: reversión a la media de manual |
  | 0.05 → 0.001 (decay 0.999) | +16.5 % | |
- **H5 — Cíclico.** Las medias están dominadas por unas pocas series que se disparan (B&H: media 281k, mediana 33k). La media aritmética no sirve para comparar.

- **H6 — Ajuste secuencial** (5000 episodios, 5 semillas, selección en validación; `resultados/log_ajuste.txt`, `figuras/fig_ajuste.png`):
  - **α:** primera pasada con {0.002, 0.005, 0.01, 0.05→0.001}: ganaba 0.002, el borde inferior. Se amplió el grid con 0.001 (D12) y ahora:
    - Cíclico: gana 0.005, un óptimo **interior**. Es el mercado de señal fuerte y tolera más ruido.
    - Reversión con Q-Learning: gana 0.002, también **interior**. Con 0.001 baja a 19 %, porque con 5000 episodios aprende demasiado despacio.
    - Regímenes y reversión con SARSA: 0.001 y 0.002 se solapan dentro de la desviación entre semillas, así que la diferencia **no es significativa**. Con α ≥ 0.005 en regímenes el agente deja de operar (H4).
    - Browniano: todas las variantes están en ≈ 0 ± 0.03 %; la elección es ruido.

    Conclusión: hace falta **α ≤ 0.002 en mercados de señal débil**. Bajar más no aporta con 5000 episodios.
  - **γ:** coincide con la escala temporal de cada mercado. En regímenes gana 0.99 (horizonte ≈ 100 pasos; cada régimen dura ≈ 50). En reversión ganan 0.9–0.95 (horizonte 10–20; vida media de la desviación ≈ 13.5). En el browniano da igual.
  - **ε:** gana casi siempre el decaimiento con f = 0.6. Con ε = 0.1 constante la curva de validación oscila y en el cíclico arranca más despacio: el agente nunca deja de pagar comisiones aleatorias y su política sigue cambiando.
- **H7 — Resultados en test** tras ampliar α (`figuras/tabla_resultados.md`), log-retorno medio ± desv. entre semillas:

  | Generador | Aleatorio | Buy & Hold | Q-Learning | SARSA |
  |---|---|---|---|---|
  | Cíclico | +145 % | +338 % | +610 ± 6 % | +612 ± 6 % |
  | Browniano | −21 % | −6 % | −2.4 ± 2.9 % (≈ no opera) | −0.9 ± 1.9 % (≈ no opera) |
  | Reversión | −16 % | −1 % | **+21.6 ± 0.0 %** | **+21.4 ± 0.5 %** |
  | Regímenes | −28 % | −22 % | +0.8 ± 3.2 % | +2.6 ± 1.3 % |

  Primera pasada del ajuste (con α ≥ 0.002), para comparar: browniano 0 / 0, regímenes +2.9 / +1.1. Las diferencias caen dentro del ruido entre semillas.
- **H8 — La política aprendida coincide con la señal (H3)** en los 4 mercados (`fig_senal_politica.png`):
  - Cíclico: momentum. Compra en los bins ≥ 4–5 y vende en los ≤ 2.
  - Browniano: no compra nunca.
  - Reversión: compra en los bins 0–1 (caída fuerte) y vende en los 7–9 (subida). Es justo "comprar bajo la media y vender al cruzarla".
  - Regímenes: vende en **ambos** extremos (volatilidad = régimen bajista) y sólo compra en el bin central 5.
- **H9 — Q-Learning frente a SARSA:** rendimiento final prácticamente igual; con ε → 0.01, el objetivo de SARSA tiende al de Q-Learning. SARSA converge antes en reversión (1400 frente a 2500 episodios) y su PnL de entrenamiento sube más deprisa en regímenes, porque al aprender el valor de la política *con* exploración evita antes las posiciones que la exploración convierte en pérdidas.

- **H10 — Discretización** (`fig_discretizacion.png`, validación, sobre la configuración final):

  | | 5 cuantiles | **10 cuantiles** | 20 cuantiles | 10 uniformes |
  |---|---|---|---|---|
  | Cíclico | 614 / 627 | **635 / 637** | 625 / 637 | 583 / 593 |
  | Reversión | 22 / 20 | **22 / 22** | 21 / 22 | 9 / 7 |
  | Regímenes | 0.3 / 0.9 | **8.5 / 6.9** | 8.6 / 9.2 | 0 / 0 |

  (Q-Learning / SARSA, log-retorno en %.)
  - Los **uniformes pierden en todos los mercados con señal**. En reversión caen a un tercio; en regímenes, a 0. Confirma D1 empíricamente: si sólo se distingue el signo de la tendencia, se pierde la información de *cuánto* se ha movido el precio, que es donde está la ventaja.
  - **5 bins** no bastan en regímenes (0.3 frente a 8.5): la ventaja está en separar los movimientos extremos de los moderados.
  - **20 bins** no mejoran de forma significativa sobre 10, porque cada celda recibe la mitad de visitas.

  ⇒ **10 es el menor número de bins que alcanza la meseta.** Es la elección parsimoniosa.
- **H11 — Brecha entre validación y test en regímenes.** Q-Learning: +8.5 % en validación frente a +0.8 % en test. Al elegir la mejor variante entre muchas con resultados que se solapan, se elige en parte el ruido de validación (*winner's curse*), y en regímenes la varianza entre series es enorme. Esto justifica la separación entre validación y test de D13: el test da la estimación honesta. **La ventaja en regímenes es pequeña y frágil.** El valor del agente aquí es **evitar pérdidas** (+1–3 % frente a −22 % de Buy & Hold), no ganar.

## 2. Decisiones

| # | Decisión | Estado | Motivo |
|---|----------|--------|--------|
| D1 | **10 bins** por **cuantiles** de la tendencia, calculados por generador con series de calibración | decidida | H1 y H10: los uniformes colapsan; 10 es el menor número de bins que alcanza la meseta |
| D2 | `n_steps = 500` fijo en train y test | decidida | el periodo del ciclo del generador 1 depende de `n_steps` |
| D3 | Evaluación greedy (ε = 0) sobre un conjunto fijo de series de test (mismas semillas para todos los agentes) | decidida | comparación justa, menos varianza |
| D4 | 5 semillas de entrenamiento por configuración, media ± desviación | decidida | robustez de las conclusiones |
| D5 | **Máscara de acciones válidas**: sin activo {mantener, comprar}; con activo {vender, mantener}. El entorno no se modifica | decidida | Las acciones sin efecto son duplicados de "mantener"; aprender su Q gasta muestras, que son escasas (σ ruido ≈ 0.02 frente a señal ≈ 0.001) |
| D6 | **Desempate aleatorio** en el argmax | decidida | Q empieza a 0; `argmax` elegiría siempre la primera acción y el agente greedy nunca entraría al mercado |
| D7 | Lógica en `trader_rl.py`; el notebook sólo hace experimentos y gráficas | decidida | Notebook legible como informe; módulo comprobable con `py trader_rl.py` |
| D8 | Curva de convergencia = balance medio **greedy** sobre series de validación fijas, cada 100 episodios | decidida | La recompensa de entrenamiento incluye exploración (ε) y es muy ruidosa |
| D9 | Un único `AgenteTabular` para ambos algoritmos; sólo cambia el objetivo del update. En SARSA a' se elige antes del update; en Q-Learning después | decidida | Comparación justa: todo lo demás es idéntico |
| D11 | **Métrica principal: log-retorno medio** `mean(log(B_T/B_0))`, junto con la mediana del balance y el nº de operaciones | decidida | H5: la media aritmética la dominan los outliers; el log-retorno corresponde al crecimiento compuesto |
| D12 | α elegido entre {0.001, 0.002, 0.005, 0.01, 0.05→0.001} por generador; 5000 episodios | decidida | H4 y H6: relación señal/ruido. Se amplió el grid con 0.001 porque 0.002 era el borde |
| D15 | La discretización es un experimento **aparte** sobre la configuración final, no una etapa más del ajuste | decidida | La política tiene que poder compararse con la señal en los mismos 10 bins, y α se ajustó para 10 bins |
| D16 | Grupo de control: 10 bins **uniformes** en [-1, 1] | decidida | Es la implementación ingenua; demuestra empíricamente D1 |
| D13 | **Protocolo de ajuste**: secuencial α → γ → ε por generador y algoritmo; 5 semillas; selección en validación (50 series, semilla 2024); el test (100 series, semilla 7777) sólo se reporta | decidida | Un grid completo sería ~5× más caro; elegir con test sería *data leakage* |
| D14 | En `fig_senales` se muestra la **semilla mediana** del algoritmo ganador en validación, no la mejor | decidida | Mostrar el mejor caso sería presentar como típico un resultado favorable |
| D10 | ε con decaimiento exponencial calibrado: `d = (ε_final/ε_inicial)^(1/(f·N))`; ε constante = `eps_inicial == eps_final` | decidida | El decaimiento queda definido por cuándo termina la exploración, no por un número mágico |

## 3. Registro de cambios

- 27/10/2026 — Creada la bitácora con los requisitos verificados y los hallazgos H1 y H2.
- 27/10/2026 — Creado `trader_rl.py` (entorno, generadores, discretizador, agente, entrenamiento, evaluación y baselines) con auto-comprobación. Velocidad ≈ 17.5 µs/paso (2000 episodios ≈ 17 s).
  Primera prueba (regímenes, 300 episodios): Q-Learning y SARSA terminan con balance greedy de 1000 en validación, es decir, aprenden a no operar. Pendiente de analizar con más episodios.
- 27/10/2026 — Primera pasada completa y análisis de la señal (H3, H4, H5).
- 27/10/2026 — Creados `experimentos.py` (ajuste α → γ → ε, baselines y señal; ≈ 24 min con 32 núcleos) y `graficas.py` (8 figuras y tabla en `figuras/`). Resultados H6–H9.
- 27/10/2026 — Ampliado el grid de α con 0.001 y repetida la cadena α → γ → ε (≈ 26 min). Experimento de discretización (≈ 15 min). Nueva figura `fig_discretizacion.png`. Resultados H10 y H11.

## 4. Datos reales: BTC-USD (rama `datos-reales-btc`)

**Objetivo:** comprobar si los métodos que funcionan en mercados sintéticos funcionan en un activo real. Script: `datos_reales.py`.

| # | Decisión | Motivo |
|---|----------|--------|
| B1 | Velas **diarias** de BTC-USD (yfinance), congeladas en `datos/btc_usd.csv` (4423 días, 2014-09-17 a 2026-10-26) | Reproducibilidad: yfinance puede limitar peticiones o revisar datos antiguos. Las velas horarias sólo cubren 730 días |
| B2 | División **cronológica**: entrenamiento 2014-09 → 2020 (2298 días), validación 2021–2022 (730), test 2023 → 2026-10 (1395) | Barajar fechas filtraría información del futuro. Validación incluye un mercado alcista (2021) y uno bajista (2022) |
| B3 | Cada episodio = **ventana aleatoria de 500 días** del periodo de entrenamiento, empaquetada como "generador" con la misma firma que los sintéticos | Cumple "una serie distinta en cada episodio" con una sola serie real, y reutiliza `entrenar()` y `Discretizador` sin cambios. Los cuantiles salen **sólo** de entrenamiento |
| B4 | Validación y test = **backtest continuo** del periodo, con los 5 días previos antepuestos | Es la evaluación estándar en finanzas; los 5 días permiten decidir desde el primer día del periodo |
| B5 | Grid α ∈ {0.001, 0.002, 0.005, 0.01} × γ ∈ {0.9, 0.95, 0.99}; ε con decaimiento (f = 0.6); 10 bins por cuantiles; 5000 episodios; 5 semillas; selección en validación | Mismo protocolo que en los sintéticos (D13), con grid conjunto α × γ porque es barato |
| B6 | Métricas: rentabilidad total, CAGR, volatilidad anual, Sharpe (rf = 0), máximo drawdown, operaciones, % del tiempo invertido. Anualización con **365 días** | Métricas estándar de backtest; cripto cotiza todos los días |
| B7 | `fig_backtest`: semilla **mediana en validación** | Igual que D14: no se elige con test |

**Limitaciones conocidas** (para el análisis crítico):
- El entorno ejecuta al cierre del día *t* con una señal que ya incluye ese cierre. En la realidad se operaría a la apertura de *t+1*, así que el backtest es ligeramente optimista.
- Las ventanas de entrenamiento se solapan mucho: el agente ve los mismos 2300 días repetidamente, con riesgo de sobreajuste.
- **No estacionariedad**: BTC en 2014–2020 tenía otra volatilidad y otra estructura de mercado que en 2023–2026.
- Comisión del 0.1 %: realista para spot en un exchange grande, pero sin *slippage*.

### Resultados BTC (`figuras/btc_*.png`, `figuras/btc_tabla.md`)

Configuración elegida en validación: Q-Learning α = 0.005, γ = 0.9; SARSA α = 0.01, γ = 0.95.

| Test 2023 → 2026-10 | Rent. total | Sharpe | Máx. DD | Volatilidad | Operaciones | Tiempo invertido |
|---|---|---|---|---|---|---|
| Buy & Hold | **+381 %** | **1.12** | 53 % | 46 % | 1 | 100 % |
| Aleatorio | +57 ± 61 % | 0.45 | 46 % | 33 % | 464 | 50 % |
| Q-Learning | +82 ± 27 % | 0.69 | **40 %** | **27 %** | 288 | 33 % |
| SARSA | +42 ± 40 % | 0.42 | 41 % | 26 % | 290 | 29 % |

Validación 2021–2022, con el mercado bajista de 2022: Buy & Hold −44 % (DD 77 %), Q-Learning y SARSA −11 % (DD 47–51 %).

- **BH1 — En test no supera a Buy & Hold**, ni en rentabilidad ni en Sharpe. Su menor volatilidad y drawdown se deben a estar **fuera del mercado el 67 % del tiempo**, no a acertar el momento: Buy & Hold tiene mejor Sharpe (1.12 frente a 0.69), así que cada unidad de riesgo que asume el agente rinde menos.
- **BH2 — Sí protege en mercado bajista:** en 2022 pierde 11 % frente a 44 %. Es el mismo comportamiento que en el generador de regímenes (H11): el valor está en evitar pérdidas.
- **BH3 — La política es coherente con la señal de entrenamiento:** momentum. Compra en los bins 6, 8 y 9 y vende en los 0–3 y 5. Pero la señal **no es estable**: en 2014–2020 los bins 6–8 daban +2–3 % a 10 días; en 2023–2026 dan ≈ 0. Sólo el bin 9 (subida fuerte) mantiene la señal. Además, **todos los bins tienen retorno medio positivo** (deriva alcista), así que estar fuera tiene un coste de oportunidad, y las barras de ±2 e.s. se solapan casi todas. No hay una ventaja de *timing* estadísticamente clara.
- **BH4 — Las comisiones pesan mucho:** 288 operaciones al 0.1 % suponen 0.999²⁸⁸ ≈ 0.75, es decir, ≈ 25 % del capital. La tendencia de 6 días cambia de bin muy a menudo y el agente entra y sale demasiado.
- **BH5 — Gran sensibilidad a los hiperparámetros** (`btc_ajuste.png`): en validación, γ = 0.99 o α = 0.001 dan entre −68 % y −94 %, frente a −12 % del mejor. Además, la validación es un único camino de 730 días, así que la selección es ruidosa. Q-Learning y SARSA difieren mucho en test (+82 % frente a +42 %) con configuraciones que en validación empataban (−12 %).

**Conclusión:** el método funciona como algoritmo (aprende una política coherente con los datos de entrenamiento y reduce el riesgo), pero con este estado (una sola variable de tendencia de 6 días) **no encuentra una ventaja explotable en BTC** después de comisiones. Coincide con lo esperado para un activo casi eficiente: es el caso "browniano con deriva", no el de "reversión a la media".

**Mejoras posibles** (no implementadas):
- Variable de tendencia larga (precio frente a media de 50/200 días).
- Penalizar el cambio de posición o añadir histéresis para reducir operaciones.
- Velas semanales.
- Reentrenamiento *walk-forward* para adaptarse a la no estacionariedad.
