# Chuleta de la presentación: Taller RL

**Duración:** 15 minutos con el profesor. Unos **10–11 minutos hablando** (Marco ≈ 3, Dani ≈ 3, Cristian ≈ 4) y el resto para preguntas.
**Hilo de la historia:** *qué aprende el agente* (Marco) → *cómo llegamos ahí* (Dani) → *cómo opera sobre el precio, qué rinde y dónde falla* (Cristian).

---

## Diapositiva 1. Portada (Marco abre, 20 segundos)

> "Somos Cristian, Dani y Marco. Hemos entrenado un agente de *trading* con Q-Learning y SARSA tabulares en los cuatro mercados sintéticos del enunciado, y además lo hemos probado con bitcoin real. En total hay 1 140 entrenamientos guardados y reproducibles."

---

## Diapositiva 2. Qué aprende el agente (Marco, ≈ 3 minutos)

**Mensaje:** el agente aprende la estrategia que cada mercado permite, ni más ni menos.

1. **Estado = (bin de tendencia, posición):** 10 bins × 2 posiciones = 20 estados, con 3 acciones.
   - Los bins son **cuantiles de cada generador**. En 3 de los 4 mercados la tendencia apenas sale de ±0,06; con bins uniformes en [−1, 1] todo caería en 2 bins (lo medimos: se pierden entre 13 y 15 pp en reversión).
2. **Cómo leer la figura:**
   - **Arriba:** la *señal real*, es decir, cuánto sube o baja el precio en los 10 pasos siguientes según el bin. Las líneas discontinuas marcan el coste de comprar y vender.
   - **Abajo:** lo que *aprende* el agente (C = compra, V = vende; cuanto más oscuro, más semillas coinciden).
3. **Recorrido por los cuatro mercados**, señalando cada uno:
   - **Cíclico:** *momentum*. Compra cuando sube y vende cuando cae.
   - **Browniano:** no hay señal por encima del coste, así que **casi nunca compra**. Es lo correcto.
   - **Reversión:** compra tras caídas fuertes (bins 0 y 1) y vende tras subidas (bins 7 a 9). Es la estrategia de manual.
   - **Regímenes:** los dos extremos son negativos, así que **vende en los extremos**.
4. **Frase de cierre:** "No le dijimos qué estrategia usar: la descubrió sola, y coincide con la señal que medimos aparte."

**Métrica:** usamos el log-retorno `ln(B_final / B_inicial)` y no la media del balance, porque en el cíclico unas pocas series se disparan (Buy & Hold: 570× de media frente a 27× de mediana).

---

## Diapositiva 3. La evolución: la clave fue α (Dani, ≈ 3 minutos)

**Mensaje:** el problema era de señal frente a ruido, no de entrenar poco.

1. **Qué falló (tarjeta 1):** con la configuración por defecto (α = 0,05, γ = 0,95, 2 000 episodios), en reversión y regímenes **los 20 agentes no operan nunca**. Se quedaban en efectivo.
2. **Hipótesis (nota inferior):** con α = 0,05, el ruido en Q (≈ 0,003) es mayor que la ventaja por paso. Si por mala suerte "comprar" parece malo, el agente deja de probarlo y el error ya no se corrige. Eso crea un sesgo hacia no operar.
3. **Qué probamos primero (tarjeta 2):** ¿y si es que entrena poco? Con **10 000 episodios sigue sin operar**. Descartado.
4. **Qué funcionó (gráfico + tarjeta 3):** bajar α. Con α = 0,05 el resultado es 0; con 0,01, +11,6; con 0,005, +17,2; con **0,002, +21,6 %**. Con el mismo número de episodios, el cambio de hiperparámetros da **+23,7 pp** (IC95 [16,2; 31,2]).
5. **Lo que aprendimos:**
   - γ y ε, ajustados después, **casi no aportan**. Solo ε en SARSA-regímenes tiene un efecto medible (+3,4 pp).
   - Con **N = 5 000 episodios** basta; regímenes necesita al menos 2 000.
   - Honestidad: la hipótesis está **parcialmente** confirmada. El ruido crece con α, pero menos de lo que predice la teoría, y forzar más exploración (ε = 0,1) no lo arregla.

**Cómo lo comprobamos:** reconstruimos la evolución con 500 entrenamientos nuevos más 40 réplicas de los originales, que salen **idénticas bit a bit**.

---

## Diapositiva 4. Así opera sobre el precio (Cristian, ≈ 1 minuto)

**Mensaje:** lo mismo que vimos en la diapositiva 2, ahora sobre el precio real de cada momento.

**Cómo se lee:** triángulo **azul** = compra, triángulo **rojo** = venta, franja **azul clara** = está dentro del mercado. Es Q-Learning con la semilla mediana (la representativa, no la mejor) sobre una serie de test, mostrando 200 pasos para que se vea bien.

- **Cíclico (arriba a la izquierda):** compra cuando el precio ya sube y se queda dentro durante toda la subida (por ejemplo, entra en ≈ 200 hacia el paso 197 y sale en ≈ 1 000 en el 229). Vende en cuanto empieza a caer. Es *momentum*.
- **Reversión (arriba a la derecha):** la mayoría de las compras están **en los valles** (≈ 93–97) y las ventas **tras los rebotes** (≈ 100–103). Entra y sale rápido: compra barato y vende caro.
- **Regímenes (abajo a la izquierda):** solo 9 operaciones. Se queda dentro durante la tendencia alcista larga (pasos 180–280) y sale tras los movimientos bruscos.
- **BTC 2025 (abajo a la derecha):** 66 operaciones en un año. Entra y sale constantemente y **no se ve un patrón claro**: sigue la tendencia de 6 días, pero con mucho ruido. Esto adelanta por qué en BTC no supera a Buy & Hold.

**Frase puente:** "Visualmente tiene sentido. Ahora veamos si además gana dinero frente a no hacer nada."

*(El browniano no aparece porque no opera: sería una línea sin ningún triángulo.)*

---

## Diapositiva 5. Resultados y análisis crítico (Cristian, ≈ 3 minutos)

**Mensaje:** opera donde hay señal y se queda fuera donde no la hay. La comparación honesta es contra **no operar**, no solo contra Buy & Hold.

1. **Las 4 tarjetas** (log-retorno en test frente a quedarse en efectivo):
   - **Cíclico, +610 pp:** casi el doble que Buy & Hold (+338). *Ojo:* es una propiedad del generador (ruido del 10 % por paso), no una rentabilidad realista.
   - **Reversión, +21,6 pp:** IC95 [18,3; 25,2]. Es el resultado más sólido.
   - **Browniano, ≈ 0:** no hay señal, así que no opera. **Es el comportamiento correcto.**
   - **Regímenes, ≈ 0 frente al efectivo**, pero +22 pp frente a Buy & Hold (−21,9). Su valor es **evitar pérdidas**.
2. **Análisis crítico (recuadro blanco):**
   - **Q-Learning ≈ SARSA:** ninguna diferencia significativa (en reversión, Q − S = +0,2 [−1,8; +2,2]). Q-Learning opera un poco más, lo esperable de un método *off-policy*.
   - **Batir a Buy & Hold no prueba que haya aprendido:** en regímenes, el agente por defecto, **que no opera**, ya bate a Buy & Hold por +21,9 pp.
3. **Datos reales: BTC (recuadro oscuro).** Entrenado con 2014–2020, seleccionado en 2021–22 y probado en 2023 a octubre de 2026:
   - **+82 %** frente a **+381 %** de Buy & Hold. **No lo supera.**
   - Pero en el año bajista **2022 pierde un 11 %** frente al 44 % de Buy & Hold. Protege, como en regímenes.
   - Sus 288 operaciones cuestan ≈ 25 % del capital en comisiones. Con una sola variable de tendencia, BTC se comporta como un "browniano con deriva".
4. **Cierre del grupo:**
   > "Tres conclusiones: el RL tabular encuentra la estrategia óptima **cuando la señal supera los costes**; el hiperparámetro decisivo es **α**, no el número de episodios; y Q-Learning y SARSA rinden igual."

---

## Preguntas probables y respuesta corta

| Pregunta | Quién | Respuesta |
|---|---|---|
| ¿Por qué cuantiles y no bins uniformes? | Marco | La tendencia vive en ±0,06 en 3 de 4 mercados; con bins uniformes se pierden 13–15 pp en reversión y 43–52 en el cíclico. |
| ¿Cuál converge antes? | Dani | No lo afirmamos. El orden cambia según el umbral elegido; con hiperparámetros idénticos SARSA va antes en 19 de 33 casos, pero solo 3 son significativos. En la práctica, unos 500 episodios en el cíclico y entre 1 000 y 2 500 en reversión y regímenes. |
| ¿Por qué Q-Learning y SARSA dan casi lo mismo? | Dani | Con ε → 0,01, el objetivo de SARSA tiende al de Q-Learning. La única diferencia visible es que Q-Learning opera un poco más. |
| ¿Hubo fuga del test? | Marco | Los hiperparámetros se eligieron en validación (otra semilla, 50 series). Limitación declarada: las series de test se habían mirado durante el desarrollo, así que la reconstrucción es exploratoria. |
| ¿Por qué el log-retorno y no la recompensa? | Cristian | La recompensa suma retornos simples. En el browniano, Buy & Hold tiene Σr = +3,6 pero log-retorno = −6,2. Medimos lo que importa financieramente. |
| ¿Por qué solo 5 semillas? | Cristian | Por coste de cómputo. Por eso damos intervalos de confianza sobre las 100 series de test y no solo el "±" entre semillas. |
| ¿Qué mejoraríais? | Cristian | Una tendencia larga (media de 50/200 días), penalizar los cambios de posición para pagar menos comisiones y reentrenar periódicamente (*walk-forward*) en BTC. |
| ¿Usasteis IA? | Todos | Sí, Claude Code como apoyo en el código y el análisis; todo está verificado y es reproducible. *(Ajustad esta respuesta a lo que pida la normativa del máster.)* |
