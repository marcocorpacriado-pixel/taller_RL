# Propuesta de entrega: PDF de 2 páginas y defensa de 15 minutos

Es una propuesta para consensuar entre Marco, Cristian y Dani, no un texto definitivo. Todas las cifras proceden de tablas ya generadas (se indica la fuente); no se ha inventado ningún resultado.

**Criterio de selección.** Priorizo el **proceso de investigación y la evidencia** sobre la cantidad de técnicas: pregunta → fallo observado → hipótesis → experimento → decisión. Cada afirmación del PDF debe tener un número y un archivo detrás.

**Restricciones del enunciado** (`BITACORA.md` §0): PDF de máximo 2 páginas con pruebas, resultados y conclusiones; fecha límite 29/10/2026 a las 23:59; primera versión por e-mail a luis.lago@uam.es antes de la clase del jueves 29/10; presentación de 15 minutos. Evaluación: 50 % mínimos, 10 % ir más allá, 20 % presentación, 10 % justificación de decisiones, 10 % análisis crítico.

---

## 1. Mensaje central del PDF (una frase)

> Un agente tabular aprende una política útil cuando el mercado ofrece una señal que su estado puede capturar (reversión a la media, momentum), y su éxito depende sobre todo de la tasa de aprendizaje α: con la configuración por defecto el agente aprende a no operar, y reducir α, no entrenar más, es lo que recupera la señal. En los mercados sin señal clara (browniano, regímenes) no se detecta ventaja sobre no operar.

Todo lo que entre en el PDF debe servir a esta frase o a sus límites.

---

## 2. Contenido exacto del PDF

Formato propuesto: A4, márgenes de 1,5 cm, tipografía de 9 pt para el texto y 8 pt para las tablas, dos páginas. Presupuesto de texto: ≈ 800–900 palabras más las tablas.

### Página 1 — Qué se hizo y qué se esperaba

**Cabecera (2 líneas).** Título: "Agente trader con Q-Learning y SARSA tabulares: del punto de partida a la solución". Autores. Una línea con el mensaje central.

**1. Diseño (≈ 130 palabras).** Contenido:

* Cuatro generadores del enunciado y entorno sin modificar. Estado = (tendencia discretizada en 10 cuantiles por generador, posición); acciones con máscara; desempate aleatorio.
* Protocolo: 5 semillas de entrenamiento; selección de hiperparámetros en **validación** (50 series, semilla 2024); reporte en **test** (100 series, semilla 7777); métrica principal = log-retorno medio; baselines = efectivo, aleatorio y Buy & Hold.
* Una frase de honestidad: "El test ya se había inspeccionado durante el desarrollo; los análisis posteriores son exploratorios".

**Figura 1 (ancho completo, ≈ 7 cm de alto).** `figuras/fig_senal_politica.png` (Marco). Pie: "La señal disponible por bin de tendencia (arriba) y la política aprendida (abajo) coinciden en cíclico y reversión. En browniano no hay señal y el agente casi no compra; en regímenes vende en ambos extremos".

**2. Resultados finales (≈ 150 palabras y Tabla 1).**

**Tabla 1. Test, log-retorno medio en pp (IC95 de población: series + semillas).** Fuente: `analisis_adicional/INFORME_FASE1.md` §4.1 y §5A.

| Mercado | Efectivo | Aleatorio | Buy & Hold | Q-Learning | SARSA | Episodios hasta estabilizar (Q / S)* |
|---|---|---|---|---|---|---|
| Cíclico | 0 | +145,2 | +337,5 | +610,4 [+566,2, +654,5] | +611,5 [+568,0, +655,0] | 200 / 200 |
| Browniano | 0 | −20,8 | −6,2 | −2,4 [−7,9, +3,1] | −0,9 [−3,5, +1,7] | no aplica (no hay rendimiento que estabilizar) |
| Reversión | 0 | −15,7 | −1,1 | +21,6 [+16,8, +26,5] | +21,4 [+16,5, +26,4] | 2 000 / 2 000 |
| Regímenes | 0 | −28,3 | −21,9 | +0,8 [−6,2, +7,8] | +2,6 [−1,4, +6,5] | 1 700 / 1 000 |

\* Mediana por semilla; criterio de la Fase 1 (umbral q = 90 % entre aleatorio y meseta, estabilización permanente). **Ninguna diferencia entre algoritmos es significativa** y el orden cambia con el umbral. Es convergencia empírica, no matemática.

**Lectura de la tabla (frases sugeridas):**

* "En reversión el agente supera de forma robusta a no operar y a Buy & Hold: +21,6 y +21,4 pp (IC95 que excluyen 0)."
* "En cíclico los +610 pp son una propiedad del generador y del horizonte de 500 pasos, no una ganancia de mercado."
* "En browniano y regímenes los IC incluyen 0: **no se detecta ventaja sobre no operar**. En regímenes el agente evita parte de las pérdidas de Buy & Hold (−21,9 pp), pero un agente que no hace nada también lo supera (+21,9 pp), así que eso no demuestra aprendizaje."

### Página 2 — Cómo se llegó y qué se aprendió

**3. De la configuración por defecto a la final (≈ 350 palabras, Figura 2 y Tabla 2).**

**Figura 2 (ancho completo, ≈ 5,8 cm).** `analisis_adicional/fase2_evolucion/figuras/fig1_inicial_vs_optimizada.png`. Pie: "Configuración inicial (E0) frente a la final por mercado; la línea horizontal en 0 es el efectivo. Con E0, reversión y regímenes quedan exactamente en 0 (no operan)".

**Tabla 2. Cada decisión con su evidencia.** Fuente: `INFORME_FASE2.md` §7.1, T3b y T9c.

| Etapa | Problema observado | Experimento | Resultado | Decisión |
|---|---|---|---|---|
| Configuración por defecto (α = 0,05, γ = 0,95, 2 000 ep.) | No opera en reversión ni en regímenes | 40 entrenamientos de la configuración por defecto | 10/10 corridas por mercado no operan; browniano 8/10; cíclico opera con mucha varianza | Investigar α |
| Más episodios | ¿Falta entrenar? | Config. por defecto con 500 a 10 000 episodios | Con 10 000, 20/20 siguen sin operar | **No es la duración** |
| α | α alta apaga el aprendizaje | α ∈ {0,001…0,05}; mismo N = 5 000 | Reversión: +23,7 [+16,2, +31,2] QL y +21,4 [+16,5, +26,4] SARSA, robustos (Holm). En regímenes solo opera α ≤ 0,002 | **Mantener: demostrado** |
| γ | ¿Debe coincidir con el horizonte del mercado? | γ ∈ {0,9, 0,95, 0,99} | Sin efecto detectable; en regímenes el test empeora (−2,0 y −2,1 pp) | No justificada estadísticamente |
| ε | ¿Decaimiento o constante? | Tres variantes | Solo SARSA en regímenes: +3,4 [+1,9, +4,9] en test | Efecto pequeño y localizado |
| Episodios | ¿Basta con 5 000? | Config. final con 500 a 10 000 | Sin cambio detectable entre 5 000 y 10 000 en reversión y regímenes; regímenes no opera con ≤ 1 000 | **N = 5 000** |
| Discretización | ¿10 cuantiles? | 5, 10, 20 cuantiles y 10 uniformes | Uniformes pierden en los 3 mercados con señal; 20 ≈ 10 | **10 cuantiles** |

**Texto sugerido:** "Nuestra evidencia respalda la decisión sobre α, la discretización y el número de episodios; **no respalda de forma estadística las decisiones sobre γ y ε**, cuyos efectos están dentro del ruido. La explicación de por qué α alta bloquea el aprendizaje está solo parcialmente contrastada: el ruido de Q crece con α (menos que √α) y forzar más exploración no lo resuelve; el mecanismo causal queda sin demostrar."

**Figura 3 (opcional, solo si cabe, ≈ 6,5 cm).** `analisis_adicional/fase2_evolucion/figuras/fig2_sensibilidad_alpha.png`, solo la fila superior de los paneles de reversión y regímenes. Requiere recortar una copia, sin tocar el original. **Es lo primero que se elimina si falta espacio.**

**4. Q-Learning frente a SARSA (≈ 60 palabras).** "No se detecta diferencia de rendimiento: los cuatro IC de población de la diferencia incluyen 0 (p. ej., reversión +0,2 [−1,8, +2,2]). Q-Learning opera algo más (52 frente a 45 operaciones en reversión), sin significación. No podemos afirmar cuál converge antes."

**5. Conclusiones y limitaciones (≈ 200 palabras).** Cinco conclusiones, que son las del apartado 4 de este documento, y una frase de limitaciones: 5 semillas; test ya inspeccionado; el efecto de α en regímenes con pocas semillas; mecanismo de α no demostrado; cíclico dependiente del generador.

---

## 3. Distribución visual

```
PÁGINA 1                                   PÁGINA 2
+---------------------------------+        +---------------------------------+
| Título · autores · mensaje      |  8 %   | 3. Evolución (texto breve)      |  12 %
+---------------------------------+        +---------------------------------+
| 1. Diseño y protocolo (texto)   | 17 %   | Fig. 2  E0 frente a final (IC)  |  22 %
+---------------------------------+        +---------------------------------+
| Fig. 1  señal y política        | 29 %   | Tabla 2  etapa → decisión       |  25 %
| (ancho completo, 4 paneles)     |        +---------------------------------+
+---------------------------------+        | [Fig. 3 α, opcional]            |  (22 %)
| 2. Tabla 1  resultados + IC     | 24 %   +---------------------------------+
+---------------------------------+        | 4. Q-Learning frente a SARSA    |  8 %
| Lectura de la tabla (3 frases)  | 22 %   +---------------------------------+
+---------------------------------+        | 5. Conclusiones y limitaciones  | 19 %
                                           +---------------------------------+
```

Si no cabe, el orden de recorte es: Figura 3 → texto de la sección 4 (pasa a una línea en la Tabla 1) → pie de la Figura 1.

**Reglas de estilo:** colores de las figuras de las Fases 1 y 2 (azul = Q-Learning, naranja = SARSA, línea negra en 0 = efectivo); cada figura y tabla con su pie, y citada en el texto; todas las cifras con IC95 de población o con "IC95 por series" indicado; la palabra "significativo" solo con el contraste que lo respalda.

---

## 4. Las cinco conclusiones propuestas

Todas respaldadas; ninguna declara superioridad entre algoritmos.

1. **El punto de partida es reproducible:** con la configuración por defecto, el agente no opera en reversión y regímenes (10/10 corridas por mercado). Es equivalente a no hacer nada, no una política aprendida.
2. **La mejora se debe sobre todo a α, no a la duración:** con el mismo número de episodios, reducir α da +23,7 y +21,4 pp en reversión (robusto). Entrenar 10 000 episodios con α = 0,05 sigue sin operar.
3. **γ y ε aportan poco y no podemos demostrar su efecto en general.** Solo ε en SARSA-regímenes es detectable.
4. **N = 5 000 es razonable;** regímenes necesita al menos 2 000. La política se estabiliza antes que el rendimiento. Es convergencia empírica.
5. **Superar a Buy & Hold no es aprender:** solo en reversión hay una ganancia absoluta robusta frente a no operar. No se detecta diferencia entre Q-Learning y SARSA.

---

## 5. Qué no afirmar

* "SARSA converge antes" / "Q-Learning es mejor" / cualquier superioridad entre algoritmos.
* "El agente es rentable en regímenes" o "en browniano no opera porque aprendió que es lo óptimo".
* "+610 % de rentabilidad" sin explicar que es una propiedad del generador y del horizonte.
* "γ y ε están justificados por los experimentos".
* "El ruido de Q ∝ √α explica el bloqueo": el mecanismo no está demostrado.
* "Los resultados de test confirman de forma independiente": el test ya se había inspeccionado.
* Citar como evidencia los pilotos de robustez de la auditoría (no hay archivos que los respalden).

---

## 6. Estructura de la defensa de 15 minutos

Los presentadores son una sugerencia para que el equipo los reparta.

| Minutos | Bloque | Mensaje | Apoyo visual | Presentador (a consensuar) |
|---|---|---|---|---|
| 0:00–2:00 | Problema y diseño | 4 mercados, estado, protocolo val/test, baselines, 5 semillas | Esquema del entorno + `fig_generadores.png` | |
| 2:00–4:30 | Qué se puede aprender | Señal por bin; la política aprendida la recupera donde existe | `fig_senal_politica.png` | |
| 4:30–7:00 | Punto de partida y primer fallo | Con la configuración por defecto, el agente aprende a no operar (10/10 en reversión y regímenes); ni más episodios lo arreglan | Fig. 2 (`fig1_inicial_vs_optimizada.png`) | |
| 7:00–10:30 | Evolución y evidencia | α demostrado; γ y ε débiles; N = 5 000; 10 cuantiles. Qué está demostrado y qué no (mecanismo de α) | Tabla 2 y fig. de α o de la cadena | |
| 10:30–12:30 | Resultados finales y algoritmos | Tabla con IC y baselines; no hay diferencia entre Q-Learning y SARSA; browniano y regímenes sin ventaja detectable | Tabla 1; `fig1_rendimientos_ic.png` | |
| 12:30–14:00 | Limitaciones y qué aprendimos | 5 semillas, test ya visto, cíclico dependiente del generador; lo que no se pudo demostrar | Una diapositiva de límites | Todos |
| 14:00–15:00 | Margen y preguntas | | | |

**Hilo narrativo para decirlo en voz alta:** "Empezamos con la configuración por defecto y falló. Estudiamos por qué, probamos si era cuestión de duración y no lo era, y vimos que el factor decisivo era α. Para cada decisión posterior hemos medido cuánta evidencia hay, y en algunas (γ, ε) hay poca".

### Preguntas previstas y cómo responder

| Pregunta | Respuesta respaldada | Evidencia |
|---|---|---|
| ¿Por qué con α = 0,05 no opera? | El ruido de Q crece con α y la relación señal/ruido cae (2,0–5,8 → 0,9–1,8), pero menos que √α; no es falta de exploración de "comprar"; forzar ε = 0,1 no lo arregla. El mecanismo exacto no está demostrado | `INFORME_FASE2.md` §5.3 |
| ¿No bastaba con más episodios? | Con 10 000 episodios y α = 0,05, 20/20 corridas siguen sin operar | `INFORME_FASE2.md` §5.2 |
| ¿Qué algoritmo es mejor? | No se detecta diferencia; Q-Learning opera algo más, sin significación. No sabemos cuál converge antes | `INFORME_FASE1.md` §4.2 y §5 |
| ¿El +610 % es real? | Es real en el generador, pero depende del horizonte y de la forma del generador; no es un resultado de mercado | `INFORME_FASE1.md` §6 |
| ¿Elegisteis hiperparámetros con el test? | No: se eligió en validación. El test ya se había visto, así que los análisis posteriores son exploratorios. El diseño de la grid de α se informó con test | `BITACORA.md` D13; `INFORME_FASE2.md` Anexo A6 |
| ¿Por qué regímenes no es rentable? | Frente a efectivo no hay ventaja detectable (+0,8 y +2,6, IC incluyen 0). Ganar a Buy & Hold (−21,9) es esperable en un mercado bajista, incluso sin operar | `INFORME_FASE1.md` §4.3 |
| ¿Es fiable el entrenamiento? | 40 entrenamientos repetidos con el código original son idénticos bit a bit; 560 entrenamientos reconstruidos desde Q reproducen los resultados guardados | `analisis_adicional/fase2_evolucion/resultados/verificacion_fase2.csv` y `analisis_adicional/resultados/verificacion_reproduccion.csv` |
| ¿Convergen Q-Learning y SARSA? | Con α constante la tabla Q no converge a un punto: fluctúa en un entorno del óptimo. Hablamos de estabilización empírica | `INFORME_FASE2.md` §6 |

---

## 7. Lista de comprobación antes de enviar

* [ ] Cada cifra del PDF aparece en una tabla de `INFORME_FASE1.md` o `INFORME_FASE2.md` y se ha copiado sin redondear de forma distinta.
* [ ] Las figuras recortadas son copias nuevas en una carpeta aparte; los originales no se tocan.
* [ ] El PDF dice que el test ya había sido inspeccionado.
* [ ] Se distingue lo de Marco, lo reproducido y lo nuevo (una línea de créditos al pie).
* [ ] Se ha decidido y declarado el uso de herramientas de IA.
* [ ] Se ha verificado que ocupa dos páginas en el PDF final (no solo en el editor).
* [ ] Primera versión enviada por e-mail antes de la clase del jueves 29/10; versión final antes de las 23:59 del 29/10.
