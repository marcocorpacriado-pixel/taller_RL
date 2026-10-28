# Auditoría final del repositorio (Fase 3, actualizada en la Fase 4)

> **Actualización de la Fase 4 (28/10/2026).** Los problemas 2, 3 y 4 de la tabla del apartado 5 (ruta personal, dependencias no declaradas, comprobación V4) **están resueltos**; ver el apartado 7 y `README_REPRODUCCION.md`. Los apartados 1–4 conservan el estado de la Fase 3; el apartado 6 contiene los comandos de commit **actualizados**.

Fecha: 28/10/2026. Comprobaciones en solo lectura: **no se ha modificado `.gitignore`, no se ha eliminado nada y no se ha hecho ningún commit, push ni pull request.** No se ha entrenado nada.

---

## 1. Estado de Git

| Elemento | Valor |
|---|---|
| Rama activa | `feature/analisis-estadistico-rl` |
| HEAD | `5a6e1ee004f11c20f2ad0fd617f0b93f5415df37` (igual a `main` y a `origin/main`) |
| Commits propios en la rama | 0 |
| Archivos rastreados modificados | 0 (`git diff` y `git diff --cached` vacíos) |
| Archivos sin seguimiento | 95 (85 de las Fases 1 y 2, 5 documentos de la Fase 3 y 5 archivos de la Fase 4), ≈ 20 MB, todos dentro de `analisis_adicional/` |
| Rama de seguimiento (upstream) | No configurada |
| Remoto | `origin` = `https://github.com/marcocorpacriado-pixel/taller_RL.git` (repositorio de Marco) |
| Otras ramas | `origin/datos-reales-btc` (`db4a963`, 27/10/2026, BTC-USD con yfinance). Modifica `BITACORA.md` y `experimentos.py` y añade `datos_reales.py`, el CSV de BTC y figuras y resultados de BTC. **No contiene `analisis_adicional/`**, así que no habría conflictos de ruta con nuestros archivos |

## 2. Integridad

| Comprobación | Resultado | Cómo |
|---|---|---|
| Archivos originales de Marco (código, bitácora, notebook, `resultados/`, `figuras/`) | **Sin cambios** | `git diff` vacío y sha256 frente a la instantánea tomada antes de la Fase 2 (51 archivos protegidos: 0 modificados) |
| Fase 1 (`analisis_adicional/` sin `fase2_evolucion/`) | **Sin cambios** | Incluida en esos 51 archivos protegidos |
| Fase 2 | **Íntegra** | `verificar_fase2.py` de la Fase 2 (17:28 del 28/10): V1–V6 OK, conservado en `resultados/verificacion_fase2_ejecucion_28oct.csv`. En la Fase 4 se reescribió y se volvió a ejecutar: ver el apartado 7 |
| Rutas citadas en los informes | **Todas existen** | 72 referencias a archivos de `INFORME_FASE1.md` e `INFORME_FASE2.md` comprobadas; las 7 que no estaban dentro de `analisis_adicional/` son los `resultados/*.pkl` de Marco, que existen en la raíz |
| `__pycache__`, `*.pyc` | **Ninguno** en el repositorio | `find` |
| Archivos temporales o copias de seguridad | **Ninguno.** El único `.log` es `registro_ejecucion.log`, que es evidencia | `find` |
| Entorno virtual | **Fuera del repositorio** (`C:\Users\dgall\.venvs\taller_rl_fase2`) | — |

**Aviso sobre las huellas (resuelto en la Fase 4).** Al terminar la Fase 3, `manifiesto.py --comprobar` y el paso V4 informaban de "añadidos: 5" porque la regla recorría `analisis_adicional/` y trataba como protegido todo lo que hubiera allí. La causa era el diseño (comparar contra una carpeta, no contra una lista), no una alteración de lo original. Ahora cada categoría se compara con una lista explícita de archivos y huellas (apartado 7).

## 3. Tamaño de lo nuevo

| Carpeta | Tamaño | Archivos |
|---|---|---|
| `analisis_adicional/fase2_evolucion/artefactos/` | 16,2 MB (17 MB según `du`) | 19 `.npz` (el mayor, 1,86 MB) |
| `analisis_adicional/fase2_evolucion/figuras/` | 1,0 MB | 6 PNG |
| `analisis_adicional/fase2_evolucion/resultados/` | 0,35 MB | 19 CSV, JSON y tablas |
| `analisis_adicional/figuras/` | 0,8 MB | 7 PNG |
| `analisis_adicional/resultados/` | 0,55 MB | 17 CSV/JSON |
| Código y documentos (`*.py`, `*.md`, `*.json`, `*.log`) | 0,37 MB | — |
| **Total** | **≈ 20 MB** | 85 archivos antes de la Fase 3 |

Ningún archivo supera los 2 MB, muy por debajo de los límites de GitHub (aviso a 50 MB, bloqueo a 100 MB).

## 4. Qué versionar y qué excluir

### Versionar (todo lo de `analisis_adicional/`)

| Grupo | Archivos | Motivo |
|---|---|---|
| Documentos | `INFORME_FASE1.md`, `fase2_evolucion/INFORME_FASE2.md`, `CONTRIBUCIONES_DANI.md`, `MATRIZ_EVIDENCIAS.md`, `RESUMEN_PARA_EQUIPO.md`, `PROPUESTA_ENTREGA.md`, `AUDITORIA_REPOSITORIO.md`, `README_REPRODUCCION.md` | Explican qué se hizo y cómo reproducirlo |
| Dependencias | `requirements_analisis.txt`, `requirements_entrenamiento.txt` | Versiones verificadas |
| Código | `comun.py`, `reanalisar_resultados.py`, `analizar_convergencia.py`, `analizar_politicas.py`; en `fase2_evolucion/`: `utilidades.py`, `entrenar.py`, `estadistica.py`, `analisis_fase2.py`, `generar_tablas_informe.py`, `ensamblar_informe.py`, `manifiesto.py`, `verificar_fase2.py`, `informe_plantilla.md` | Reproducir cualquier cifra |
| Resultados | `resultados/` (17 archivos) y `fase2_evolucion/resultados/` (19) | Las tablas del informe salen de ellos |
| Figuras | `figuras/` (7) y `fase2_evolucion/figuras/` (6) | Evidencias visuales |
| Procedencia | `manifiesto_huellas_inicio.json`, `manifiesto_integridad.json`, `registro_ejecucion.log`, `manifiesto_corridas.csv`, `manifiesto_procedencia.json`, `verificacion_*.csv` | Demuestran qué es histórico y qué es nuevo |
| **Artefactos** | `fase2_evolucion/artefactos/*.npz` (17 MB) | Ver decisión abajo |

**Decisión sobre los `.npz`.** Recomiendo **versionarlos**: son los datos primarios de los 540 entrenamientos nuevos, el análisis (`analisis_fase2.py`) y la verificación V1 dependen de ellos, y regenerarlos cuesta unos 35 minutos con 16 procesos y un entorno con gymnasium. Pesan 17 MB y no se reescriben, así que no inflan el historial. La alternativa es excluirlos (`fase2_evolucion/artefactos/` en `.gitignore`) y publicarlos como adjunto de una *release* o en una carpeta compartida; en ese caso hay que documentarlo, porque sin ellos solo se puede reproducir el análisis reentrenando.

### Excluir

| Qué | Motivo |
|---|---|
| `__pycache__/`, `*.pyc` | Ya ignorados (`.gitignore` de Marco) |
| El entorno virtual `taller_rl_fase2` | Está fuera del repositorio. Si alguien lo crea dentro, añadir `.venv*/` |
| Archivos del scratchpad de la sesión (fuera del repositorio) | No son del proyecto |
| `.claude/` (ajustes locales de Claude Code), si aparece | Configuración personal |

Hoy **no hay nada que excluir** dentro del árbol. No hace falta tocar `.gitignore`.

## 5. Problemas potenciales para publicar la rama

| # | Problema | Gravedad | Propuesta |
|---|---|---|---|
| 1 | **Permisos.** El remoto es el repositorio de Marco; no sé si Dani tiene permiso de escritura | Alta | Pedir a Marco que añada a Dani como colaborador, o hacer un fork y un pull request (ver comandos) |
| 2 | **Ruta personal.** `C:\Users\dgall\.venvs\taller_rl_fase2` aparece en `entrenar.py` (docstring), `informe_plantilla.md` y `INFORME_FASE2.md` | Media (privacidad y portabilidad) | **Resuelto (Fase 4):** ninguna ruta operativa; en los informes se conserva como registro de la ejecución, con una nota que la declara específica del equipo del autor |
| 3 | **Dependencias no declaradas.** No hay archivo de requisitos | Media | **Resuelto (Fase 4):** `requirements_analisis.txt` y `requirements_entrenamiento.txt` |
| 4 | **Comprobación V4.** Ver el aviso del apartado 2: "añadidos: 5" | Baja | **Resuelto (Fase 4):** manifiesto por categorías con adiciones declaradas |
| 5 | **Binarios en el historial.** 17 MB de `.npz` quedan para siempre en Git | Baja | Ver decisión de los artefactos |
| 6 | **Saltos de línea.** `core.autocrlf=true`: Git avisará de conversión LF/CRLF en archivos de texto | Baja | Es solo un aviso; los `.npz` y `.png` son binarios |
| 7 | **Identidad del commit.** Se usará la de `git config` (usuario "Daniel") | Baja | Comprobar `git config user.name` y `user.email` antes del commit |
| 8 | **Evidencia sin respaldo.** Los pilotos de robustez de la auditoría y partes de H4/H7 no tienen archivos | Media (científica) | Están marcados en `MATRIZ_EVIDENCIAS.md` §E y no se citan como resultado |

## 6. Comandos propuestos (no ejecutados)

Se ejecutarán **solo cuando se autorice**. Ejecutar desde la raíz del repositorio, con la rama `feature/analisis-estadistico-rl`. Tres commits: separan la Fase 1, la Fase 2 y la documentación con la reproducibilidad, de modo que se pueden revisar o descartar por separado.

**Sobre la línea `Co-Authored-By`.** Se ha comprobado contra el registro de la sesión (`~/.claude/projects/.../cefcb8f9-….jsonl`): los 344 mensajes del asistente tienen el campo `model` = `claude-sonnet-5-5`, desde el primer mensaje (28/10/2026 07:02 UTC) hasta el último de esta fase. Ningún otro modelo ni persona escribió código o texto de estos tres commits, de modo que `Claude Sonnet 5.5` refleja la coautoría real. La autoría del commit es la de Dani (identidad de `git config`, que dirigió el trabajo). **Ningún archivo de Marco entra en estos commits**, así que no se le atribuye coautoría; su trabajo se cita en los mensajes como material de partida. Si el máster exige declarar el uso de IA de otra forma (decisión 8 de `RESUMEN_PARA_EQUIPO.md`), se puede quitar la línea o cambiarla antes de ejecutar.

**Comprobaciones previas**

```bash
git branch --show-current                                   # feature/analisis-estadistico-rl
git rev-parse HEAD                                          # 5a6e1ee004f11c20f2ad0fd617f0b93f5415df37
git status --short --untracked-files=no                     # vacío: ningún archivo rastreado modificado
git status --short -uall | grep -v ' analisis_adicional/'   # vacío: todo lo nuevo está en analisis_adicional/
git config user.name && git config user.email               # Daniel · d.gallegosan@gmail.com
python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --comprobar          # salida 0
python -B -X utf8 analisis_adicional/fase2_evolucion/verificar_fase2.py                 # V1–V6 OK  (en Windows: py -3)
```

**Commit 1 — Fase 1: análisis estadístico**

```bash
git add analisis_adicional/INFORME_FASE1.md analisis_adicional/comun.py \
        analisis_adicional/reanalisar_resultados.py analisis_adicional/analizar_convergencia.py \
        analisis_adicional/analizar_politicas.py analisis_adicional/resultados analisis_adicional/figuras
git diff --cached --name-only | grep -v '^analisis_adicional/' ; git diff --cached --stat | tail -1   # primera línea vacía
git commit -F - <<'EOF'
Fase 1: reanálisis estadístico de los artefactos de Marco (sin reentrenar)

Reutiliza los 600 entrenamientos ya guardados en resultados/*.pkl, cargados con
deserialización restringida. Reconstruye cada política desde su tabla Q y la
reproduce contra val/test guardados. Añade intervalos por series y de población,
comparaciones pareadas con Holm, baseline de efectivo, recompensa frente a
log-retorno, convergencia con umbral variable y estabilidad de políticas.
No modifica ningún archivo de Marco.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```

**Commit 2 — Fase 2: evolución experimental** (todo `fase2_evolucion/` salvo el manifiesto de integridad de la Fase 4, que va en el commit 3)

```bash
git add analisis_adicional/fase2_evolucion ':!analisis_adicional/fase2_evolucion/manifiesto_integridad.json'
git diff --cached --name-only | grep -v '^analisis_adicional/fase2_evolucion/' ; git diff --cached --stat | tail -1   # primera línea vacía
git commit -F - <<'EOF'
Fase 2: reconstrucción experimental de la evolución del agente

500 entrenamientos nuevos y 40 réplicas de entrenamientos de Marco (40/40
idénticas bit a bit) con su trader_rl.py sin modificar, en un entorno aislado
(Python 3.12.10, numpy 2.5.4, gymnasium 1.4.0). Barrido de episodios, alfa x N,
ruido de Q, intervención sobre epsilon y cadena E0 -> E3 con el mismo N.
Exploratorio: el test de Marco ya había sido inspeccionado. Los entrenamientos
del 28/10/2026 no son histórico.

Incluye las ediciones de portabilidad de la Fase 4 en los scripts de esta
carpeta (sin rutas personales; no cambian resultados ni algoritmos).

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```

**Commit 3 — Fase 3 y 4: documentación y reproducibilidad**

```bash
git add analisis_adicional/CONTRIBUCIONES_DANI.md analisis_adicional/MATRIZ_EVIDENCIAS.md \
        analisis_adicional/RESUMEN_PARA_EQUIPO.md analisis_adicional/PROPUESTA_ENTREGA.md \
        analisis_adicional/AUDITORIA_REPOSITORIO.md analisis_adicional/README_REPRODUCCION.md \
        analisis_adicional/requirements_analisis.txt analisis_adicional/requirements_entrenamiento.txt \
        analisis_adicional/fase2_evolucion/manifiesto_integridad.json
git diff --cached --stat | tail -1
git commit -F - <<'EOF'
Fases 3 y 4: contribuciones, evidencias, propuesta de entrega y reproducibilidad

Documenta con procedencia (original, reutilizado, reproducido, nuevo) qué es de
Marco y qué es nuevo; matriz de evidencias; resumen para el equipo; propuesta del
PDF de 2 páginas y de la defensa; auditoría del repositorio. Añade el manifiesto
de integridad por categorías con listas explícitas, requisitos con las versiones
verificadas y el README de reproducción.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF

git status --short                                            # debe quedar vacío
git log --oneline -4
python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --comprobar   # sigue en salida 0 tras los commits
python -B -X utf8 analisis_adicional/fase2_evolucion/verificar_fase2.py          # V6 comprueba que nada fuera de analisis_adicional/ ha cambiado
```

Entre el commit 2 y el 3, `manifiesto.py --comprobar` falla ("FALTA manifiesto_integridad.json"): es lo esperado, porque ese archivo entra en el commit 3.

Con `core.autocrlf=true` Git mostrará avisos `LF will be replaced by CRLF`; no afectan al contenido (la comprobación de integridad tolera la diferencia de saltos de línea).

**Publicar: opción A (con permiso de escritura en el repositorio de Marco)**

```bash
git push -u origin feature/analisis-estadistico-rl
```

**Publicar: opción B (sin permiso: fork y pull request)**

```bash
gh repo fork marcocorpacriado-pixel/taller_RL --remote --remote-name fork --clone=false
git push -u fork feature/analisis-estadistico-rl
gh pr create --repo marcocorpacriado-pixel/taller_RL --base main \
   --head <usuario-github-de-Dani>:feature/analisis-estadistico-rl \
   --title "Análisis estadístico y reconstrucción de la evolución del agente" \
   --body "Fases 1–4: reanálisis, reconstrucción experimental, documentación de contribuciones y reproducibilidad. No modifica ningún archivo original; todo está en analisis_adicional/. Ver analisis_adicional/CONTRIBUCIONES_DANI.md y README_REPRODUCCION.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

**Después de publicar:** `git log origin/feature/analisis-estadistico-rl --oneline -4` y revisar en la web que `main` no ha cambiado.

## 7. Cierre técnico de la Fase 4

| Problema | Causa | Solución | Comprobación |
|---|---|---|---|
| `manifiesto.py --comprobar` decía "añadidos: 5" y V4 fallaba | La regla recorría `analisis_adicional/` y trataba como protegido cualquier archivo que hubiera allí (comparaba contra una carpeta, no contra una lista) | `manifiesto_integridad.json`: cada categoría (originales de Marco, Fase 1, artefactos de la Fase 2, código e informes de la Fase 2, adiciones de las Fases 3 y 4) se compara con una **lista explícita de archivos y sha256**. Un archivo no declarado hace fallar. No se quitó ningún archivo de la protección ni se desactivó ninguna verificación | `--comprobar` sale con 0; en la copia, modificar `trader_rl.py`, un `.npz` o un documento declarado, o añadir un archivo sin declarar, lo detecta y sale con 1 |
| `verificar_fase2.py` V5 añadía una línea al `registro_ejecucion.log` en cada ejecución | `entrenar.py` registraba "INICIO" aunque no hubiera nada que entrenar | `entrenar.py` solo registra si hay grupos pendientes; V5 comprueba además que el log no cambia. Las líneas `grupos=[]` que ya había (p. ej. 17:07, 17:25 y 17:28) son de ejecuciones de verificación anteriores y se conservan | V5 OK; el log conserva su hash original (`5d9d337f…`) |
| V6 fallaría después de los commits | Exigía `HEAD == HEAD` de inicio | Exige que el HEAD de inicio sea ancestro del actual y que no haya cambiado nada fuera de `analisis_adicional/` desde entonces | V6 OK; en la copia con archivos de Marco alterados, FALLO |
| Ruta personal en `entrenar.py`, `verificar_fase2.py` y los informes | Rutas de ejecución del autor | `entrenar.py`: docstring genérico. `verificar_fase2.py`: intérprete por `TALLER_RL_PYTHON`, `~/.venvs/…` (Windows y POSIX) o el actual; `--sin-v5` explícito. Informes: se conserva la ruta como registro y se aclara que es del equipo del autor | Ninguna ruta operativa (ver más abajo) |
| Sin dependencias ni instrucciones | — | `requirements_analisis.txt`, `requirements_entrenamiento.txt`, `README_REPRODUCCION.md` | Versiones tomadas de `pip list` de los dos entornos |
| Hashes dependientes de CRLF/LF | `core.autocrlf` cambia los saltos de línea al clonar | Hash adicional con saltos de línea normalizados en archivos de texto; la diferencia se acepta y se cuenta aparte | Copia con 59 archivos convertidos a LF: 0 modificados |

**Pruebas sin entrenar (28/10/2026).**

| Prueba | Resultado |
|---|---|
| `verificar_fase2.py` en el repositorio | V1 (540 corridas, max\|dif\|=0), V2 (40/40 idénticas), V3 (400 series), V4a–V4e, V5 y V6: todas OK, salida 0 |
| Misma verificación en una copia, lanzada desde otra carpeta | OK (V5 con `TALLER_RL_PYTHON` y también omitida con `--sin-v5`) |
| Reejecutar los 5 scripts de análisis en la copia | Salida 0 en todos; 50 archivos de `resultados/` y `figuras/` **idénticos byte a byte** a los versionados |
| Copia con los textos convertidos a LF (simula un clon en Linux) | `--comprobar` sale con 0 (los 59 archivos se cuentan como "sólo saltos de línea") |
| Pruebas negativas en la copia | Modificar un original, un `.npz` o un documento declarado, y añadir un archivo no declarado: todos se detectan (salida 1). `--actualizar-adiciones` se niega si un artefacto no coincide |

**Qué no se ha probado:** Linux o macOS reales, otras versiones de Python o NumPy, un venv de análisis creado desde cero (el análisis se verificó con el Python del sistema) y un reentrenamiento completo (la prohibición de entrenar se respetó; solo se comprobó que `entrenar.py` no sobrescribe).

**Rutas.** `grep` de `dgall` en `analisis_adicional/`: aparece en 7 archivos, todos como texto descriptivo (informes, auditoría, README y el propio manifiesto), ninguno como ruta que se use. El código solo usa rutas calculadas desde `__file__` y `Path.home()`.
