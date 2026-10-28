# Resumen para Marco y Cristian

Lectura de unos 10 minutos. El detalle y las pruebas están en `CONTRIBUCIONES_DANI.md`, `MATRIZ_EVIDENCIAS.md` y los informes de las Fases 1 y 2. Para comprobar que todo es íntegro o repetirlo en otro equipo: `README_REPRODUCCION.md` (la verificación no reentrena y tarda unos 12 s). **No he modificado nada de lo de Marco** (`trader_rl.py`, `experimentos.py`, `graficas.py`, `BITACORA.md`, notebook, `resultados/`, `figuras/`) y **no he hecho commits ni pushes**. Todo está en la carpeta `analisis_adicional/` de la rama `feature/analisis-estadistico-rl`.

---

## Qué he hecho

1. **Auditoría de solo lectura** del repositorio de Marco (código, bitácora, protocolo, estadística).
2. **Fase 1:** reanálisis estadístico de los 600 entrenamientos que Marco ya había guardado, **sin reentrenar nada**.
3. **Fase 2:** reconstrucción de cómo evolucionó el agente desde la configuración por defecto hasta la final, con **500 entrenamientos nuevos** más 40 réplicas de entrenamientos de Marco. Los hice con **su código original, sin tocarlo**, en un entorno virtual aislado fuera del repositorio.

Las 40 réplicas salen **idénticas bit a bit** a las de Marco, así que sus resultados son reproducibles.

## Por qué era necesario

* El repositorio tenía **un solo commit**: la historia "configuración inicial → solución final" solo existía como texto en la bitácora. Los artefactos de la primera pasada fueron sobrescritos.
* El enunciado pide **ajustar el número de episodios** y reportar **tiempo de convergencia**. Ninguno de los dos estaba resuelto de forma sólida.
* Los "± entre semillas" de la tabla no reflejan la incertidumbre real, y faltaba el baseline de **no operar**, imprescindible para interpretar browniano y regímenes.
* El profesor valora la justificación de decisiones (10 %) y el análisis crítico (10 %): conviene que cada decisión tenga evidencia.

## Resultados nuevos

1. **Los resultados de Marco se reproducen.** Reversión (+21,6 / +21,4 pp frente a efectivo) y cíclico (+610 / +612 pp) son robustos. Q-Learning y SARSA son indistinguibles en rendimiento.
2. **La configuración por defecto no opera en reversión ni en regímenes** (10 de 10 corridas por mercado). En browniano, 8 de 10. En cíclico opera, pero es muy inestable.
3. **La mejora se debe sobre todo a α**, no a entrenar más. Con el mismo número de episodios, los hiperparámetros dan +23,7 y +21,4 pp en reversión. Con la configuración por defecto, 10 000 episodios siguen sin operar (20 de 20 corridas).
4. **γ y ε aportan poco** y casi nada se puede demostrar. Solo ε en SARSA-regímenes muestra un efecto detectable.
5. **N = 5 000 episodios es razonable.** Regímenes necesita al menos 2 000 (con 1 000 o menos, Q-Learning no opera).
6. **El motivo de que α alta bloquee el aprendizaje está solo parcialmente explicado.** El ruido de Q crece con α, pero menos de lo que dice la bitácora, y forzar más exploración no lo arregla.
7. **Un agente que no hace nada ya supera a Buy & Hold en regímenes** (+21,9 pp). Superar a Buy & Hold en ese mercado no demuestra aprendizaje.

**Importante:** el test (100 series, semilla 7777) ya lo había visto Marco, y solo hay 5 semillas. La Fase 2 es **exploratoria**, no una confirmación independiente. Las ejecuciones del 28/10 **no** son históricas.

## Afirmaciones que cambian

| Afirmación actual | Cómo cambia |
|---|---|
| "± desviación entre semillas" en la tabla | Usar IC por series (o de población). El "+21,6 ± 0,0" esconde ≈ ±5 pp |
| "SARSA converge antes" (H9) | **Retirar.** El texto (1 400) y la tabla (800) no coinciden, mezcla hiperparámetros distintos y el orden depende del umbral |
| "Tiempo (s)" como tiempo de convergencia | Es el entrenamiento completo con 32 procesos. Reportar **episodios hasta estabilización** |
| Browniano: "no opera nunca" / "lo óptimo es no operar" | Es una mezcla de efectivo y Buy & Hold; el óptimo depende de la métrica |
| Regímenes: "evita pérdidas" | Mantenerlo, pero añadir que un agente que no opera ya supera a Buy & Hold, y que frente a efectivo no hay ventaja detectable |
| γ y ε "justificados" (H6) | Solo α está justificada estadísticamente |
| Tabla de α de H4 (α = 0,01 → +7,4 %) | No coincide con los datos actuales (+11,6 / +11,9). No verificable: la primera pasada se sobrescribió |
| D7: "el notebook importa este módulo" | El notebook del repositorio solo contiene el enunciado |

La lista completa (15 puntos) está en `CONTRIBUCIONES_DANI.md` §7. **La bitácora es de Marco: él decide qué se cambia.**

## Figuras que recomiendo incorporar

1. `figuras/fig_senal_politica.png` (Marco): señal disponible frente a política aprendida. Es la mejor figura del trabajo y no hay que rehacerla.
2. `analisis_adicional/fase2_evolucion/figuras/fig1_inicial_vs_optimizada.png`: E0 frente a final con IC y la línea de efectivo.
3. `analisis_adicional/fase2_evolucion/figuras/fig2_sensibilidad_alpha.png` (fila superior) o `fig4_cadena_E0_E3.png`: evidencia de que la mejora está en α.
4. Opcionales: `analisis_adicional/figuras/fig1_rendimientos_ic.png` (IC y baselines) y `fase2_evolucion/figuras/fig3_numero_de_episodios.png` (episodios).

La propuesta completa de distribución del PDF está en `PROPUESTA_ENTREGA.md`.

## Qué falta para entregar

* **El PDF de 2 páginas no existe.** Fecha límite: 29/10/2026 a las 23:59, con una primera versión por e-mail a luis.lago@uam.es antes de la clase del jueves 29/10.
* Decidir qué cifras y figuras entran (ver abajo) y recortar una versión de la figura de evolución para el espacio disponible.
* Confirmar si se envió el e-mail para organizar el orden de presentación (la bitácora indica que debía enviarse antes del martes 27/10; no puedo verificarlo).
* Preparar el guion de 15 minutos (propuesta en `PROPUESTA_ENTREGA.md`).
* Resolver cómo y dónde publicar la rama (ver `AUDITORIA_REPOSITORIO.md`).

**Pendientes científicos que no están hechos** (opcionales, si queda tiempo): test fresco y mayor con las políticas guardadas; α = 0,001 con N = 10 000; robustez a σ, κ y `n_steps` con IC; ε constante en Q-Learning frente a SARSA.

## Preguntas y decisiones para consensuar

1. **Bitácora:** ¿Marco corrige sus afirmaciones (H9, "Tiempo (s)", H4, D7) o las citamos como "matices" en el PDF sin editar la bitácora?
2. **Métrica de incertidumbre:** ¿mostramos IC por series, IC de población (series + semillas) o ambos? Propongo IC por series en la tabla y mencionar el de población en el texto. Los de población son más anchos y más honestos.
3. **Tiempo de convergencia:** ¿qué criterio presentamos? Opciones: el original de Marco (frágil), el de la Fase 1 (episodios hasta estabilizar con q = 90 %) o el "N suficiente" de la Fase 2. No declaramos qué algoritmo converge antes.
4. **Test:** como el test ya lo vio Marco, ¿lo declaramos explícitamente en el PDF? Propongo que sí, en una frase.
5. **Cíclico:** ¿cómo presentamos el +610 %? Es una propiedad del generador y del horizonte de 500 pasos, no una "ganancia financiera". Propongo mostrarlo, pero sin presentarlo como éxito de mercado.
6. **Regímenes:** ¿"evita pérdidas" o "no hay evidencia de rentabilidad"? Propongo ambos, con la comparación frente a efectivo.
7. **Rama BTC de Marco (`origin/datos-reales-btc`):** ¿la incluimos? Yo no la he auditado a fondo.
8. **Declaración de IA:** los análisis y documentos de esta rama se elaboraron con Claude Code. ¿Cómo lo declaramos según las normas del máster?
9. **Autoría en el PDF:** ¿cómo atribuimos el trabajo de cada uno? Propongo usar las etiquetas de `CONTRIBUCIONES_DANI.md` (original, reutilizado, reproducido, nuevo).
10. **Dónde publicar:** el remoto `origin` es el repositorio de Marco. ¿Tengo permiso de escritura, o hago un fork y un pull request? ¿Incluimos los 17 MB de artefactos `.npz` en Git o los dejamos fuera?
11. **Alcance de lo nuevo:** ¿hacemos algún pendiente científico antes de la entrega o lo dejamos como limitación declarada?
