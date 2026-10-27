| Generador | Agente | Log-ret. test (%) | Mediana balance ($) | Operaciones | Episodios hasta convergencia | Tiempo (s) | α · γ · ε |
|---|---|---|---|---|---|---|---|
| Tendencia + ciclo | Aleatorio | +145.2 ± 10.5 | 4,043 | 165 | — | — | — |
| Tendencia + ciclo | Buy & Hold | +337.5 ± 0.0 | 32,600 | 1 | — | — | — |
| Tendencia + ciclo | **Q-Learning** | +610.4 ± 6.3 | 432,288 | 62 | 500 | 97 | 0.005 · 0.95 · 1→0.01 (f=0.6) |
| Tendencia + ciclo | **SARSA** | +611.5 ± 6.4 | 429,966 | 53 | 500 | 91 | 0.005 · 0.9 · 1→0.01 (f=0.6) |
| Browniano geométrico | Aleatorio | -20.8 ± 1.6 | 815 | 165 | — | — | — |
| Browniano geométrico | Buy & Hold | -6.2 ± 0.0 | 958 | 1 | — | — | — |
| Browniano geométrico | **Q-Learning** | -2.4 ± 2.9 | 1,000 | 0 | 200 | 99 | 0.001 · 0.99 · 1→0.01 (f=0.3) |
| Browniano geométrico | **SARSA** | -0.9 ± 1.9 | 1,000 | 0 | 200 | 88 | 0.001 · 0.99 · 1→0.01 (f=0.3) |
| Reversión a la media | Aleatorio | -15.7 ± 0.7 | 845 | 165 | — | — | — |
| Reversión a la media | Buy & Hold | -1.1 ± 0.0 | 991 | 1 | — | — | — |
| Reversión a la media | **Q-Learning** | +21.6 ± 0.0 | 1,246 | 52 | 2500 | 103 | 0.002 · 0.95 · 1→0.01 (f=0.6) |
| Reversión a la media | **SARSA** | +21.4 ± 0.5 | 1,242 | 45 | 800 | 97 | 0.001 · 0.95 · 0.1 |
| Regímenes alcista/bajista | Aleatorio | -28.3 ± 1.5 | 765 | 165 | — | — | — |
| Regímenes alcista/bajista | Buy & Hold | -21.9 ± 0.0 | 826 | 1 | — | — | — |
| Regímenes alcista/bajista | **Q-Learning** | +0.8 ± 3.2 | 1,008 | 33 | 1000 | 102 | 0.001 · 0.99 · 1→0.01 (f=0.6) |
| Regímenes alcista/bajista | **SARSA** | +2.6 ± 1.3 | 1,000 | 28 | 1300 | 101 | 0.001 · 0.99 · 1→0.01 (f=0.3) |

Log-ret.: media ± desv. entre semillas de la media sobre las series de test.
