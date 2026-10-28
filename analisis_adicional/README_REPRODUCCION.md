# Reproducción y verificación de `analisis_adicional/`

Esta carpeta contiene el análisis estadístico (Fase 1), la reconstrucción experimental (Fase 2) y la documentación de las contribuciones (Fases 3-4). **No modifica ningún archivo de Marco** (`trader_rl.py`, `experimentos.py`, `graficas.py`, `BITACORA.md`, el notebook, `resultados/`, `figuras/`). Para el contenido científico, empieza por `CONTRIBUCIONES_DANI.md`; aquí solo está cómo comprobarlo y repetirlo.

Todos los comandos se lanzan **desde la raíz del repositorio** (también funcionan desde cualquier otra carpeta: los scripts calculan sus rutas a partir de su propia ubicación). En Windows usa `py -3` donde pone `python`; añade `-X utf8` si la consola no es UTF-8 (Windows con cp1252 falla al imprimir `≈` y `±`). `-B` evita crear `__pycache__` dentro del repositorio.

## 1. Versiones verificadas

| Entorno | Para qué | Python | NumPy | Gymnasium | Matplotlib | Archivo |
|---|---|---|---|---|---|---|
| **Análisis** | Fase 1, análisis de la Fase 2, tablas, informe, verificaciones V1–V4 y V6 | 3.14.3 | 2.4.2 | no hace falta | 3.10.8 | `requirements_analisis.txt` |
| **Entrenamiento** | Solo para repetir los entrenamientos de la Fase 2 (y V5) | 3.12.10 | 2.5.4 | 1.4.0 | no se usa | `requirements_entrenamiento.txt` |

Ambos se probaron en Windows 11. Son los dos entornos con los que se obtuvieron los resultados; **no se ha probado un entorno único** que combine NumPy 2.5.4 y Matplotlib, ni otras versiones de Python ni Linux/macOS. El análisis usa solo NumPy y Matplotlib (sin SciPy ni pandas).

Crear los entornos (los archivos de requisitos fijan las versiones exactas; hace falta que existan esas versiones de Python):

```bash
# Linux / macOS
python3.14 -m venv ~/.venvs/taller_rl_analisis
~/.venvs/taller_rl_analisis/bin/python -m pip install -r analisis_adicional/requirements_analisis.txt
python3.12 -m venv ~/.venvs/taller_rl_fase2
~/.venvs/taller_rl_fase2/bin/python -m pip install -r analisis_adicional/requirements_entrenamiento.txt
```

```powershell
# Windows (PowerShell)
py -3.14 -m venv $HOME\.venvs\taller_rl_analisis
& $HOME\.venvs\taller_rl_analisis\Scripts\python.exe -m pip install -r analisis_adicional\requirements_analisis.txt
py -3.12 -m venv $HOME\.venvs\taller_rl_fase2
& $HOME\.venvs\taller_rl_fase2\Scripts\python.exe -m pip install -r analisis_adicional\requirements_entrenamiento.txt
```

Los resultados del análisis se obtuvieron con el Python 3.14.3 del sistema, sin venv; un venv con las mismas versiones debería dar lo mismo, pero no se creó para esta comprobación. Los entornos **no deben crearse dentro del repositorio** (si se hace, hay que ignorarlos en Git).

## 2. Verificar los resultados existentes SIN reentrenar

Es lo que debe hacer cualquiera que reciba la rama. No escribe nada salvo `fase2_evolucion/resultados/verificacion_fase2.csv` (se regenera en cada ejecución).

```bash
# a) Integridad por categorías (originales de Marco, Fase 1, artefactos y código de la Fase 2, documentos añadidos). Salida 0 = todo coincide.
python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --comprobar

# b) Batería completa V1–V6 (reconstruye la política de cada Q guardada y la compara con val/test; réplica de Marco 40/40; integridad; git).
#    V5 necesita un intérprete con gymnasium: define TALLER_RL_PYTHON o crea ~/.venvs/taller_rl_fase2. Sin él: añade --sin-v5 (queda como OMITIDA).
python -B -X utf8 analisis_adicional/fase2_evolucion/verificar_fase2.py
TALLER_RL_PYTHON=/ruta/al/python python -B -X utf8 analisis_adicional/fase2_evolucion/verificar_fase2.py
python -B -X utf8 analisis_adicional/fase2_evolucion/verificar_fase2.py --sin-v5
```

Qué garantiza cada categoría y cómo se declara: ver el docstring de `fase2_evolucion/manifiesto.py`. Resumen:

| Categoría | Qué contiene | Lista de huellas | ¿Editable? |
|---|---|---|---|
| `original_marco` | código, bitácora, notebook, `resultados/`, `figuras/` | `fase2_evolucion/manifiesto_huellas_inicio.json` | No |
| `fase1` | todo `analisis_adicional/` de la Fase 1 | `fase2_evolucion/manifiesto_huellas_inicio.json` | No |
| `fase2_artefactos` | `.npz`, `resultados/`, `figuras/` y registro de ejecución de la Fase 2 | `fase2_evolucion/manifiesto_integridad.json` | No |
| `fase2_codigo` | scripts e informes de la Fase 2, con su huella anterior a la Fase 4 y qué se cambió | `fase2_evolucion/manifiesto_integridad.json` | Sí, con `--actualizar-adiciones` (queda en el historial) |
| `adiciones_fase3` / `adiciones_fase4` | documentos de la Fase 3 y archivos de reproducibilidad de la Fase 4 | `fase2_evolucion/manifiesto_integridad.json` | Sí, con `--actualizar-adiciones` |

Un archivo nuevo en `analisis_adicional/` que no esté en ninguna lista **hace fallar** la comprobación hasta que se declare (`manifiesto.py --actualizar-adiciones`, que lo anota en el historial). Salidas que se reescriben en cada verificación y por eso no se hashean (declaradas en el manifiesto): `fase2_evolucion/resultados/verificacion_fase2.csv`.

**Saltos de línea.** Git con `core.autocrlf` puede convertir CRLF↔LF al clonar en otro sistema. Los archivos de texto tienen registrado también un hash con saltos de línea normalizados; la comprobación acepta esa diferencia y la cuenta aparte ("sólo en saltos de línea"). Los binarios (`.pkl`, `.png`, `.npz`) se comparan byte a byte.

## 3. Regenerar el análisis (sin reentrenar)

Los scripts de análisis **reescriben** sus salidas (`resultados/`, `figuras/`) con semillas fijas (bootstrap `20261028`), sin tocar los `.pkl` de Marco ni los `.npz`. Para no pisar los archivos versionados, **hazlo en una copia**:

```bash
cp -r . /tmp/copia_repo && cd /tmp/copia_repo        # o cualquier carpeta fuera del repositorio
```

Orden de ejecución (en la copia, o en el repositorio si se quiere regenerar a propósito):

| # | Script | Entrada | Salida | Tiempo orientativo |
|---|---|---|---|---|
| 1 | `analisis_adicional/reanalisar_resultados.py` | `resultados/*.pkl` de Marco | `analisis_adicional/resultados/`, `figuras/` | ≈ 30 s |
| 2 | `analisis_adicional/analizar_convergencia.py` | ídem | ídem | ≈ 2 s |
| 3 | `analisis_adicional/analizar_politicas.py` | ídem | ídem | ≈ 2 s |
| 4 | `analisis_adicional/fase2_evolucion/analisis_fase2.py` | `.pkl` de Marco + `artefactos/*.npz` | `fase2_evolucion/resultados/`, `figuras/` | ≈ 2,5 min |
| 5 | `analisis_adicional/fase2_evolucion/generar_tablas_informe.py` | CSV del paso 4 | `resultados/tablas_informe.md` | < 1 s |
| 6 | `analisis_adicional/fase2_evolucion/ensamblar_informe.py` | plantilla + tablas + `verificacion_fase2.csv` | `INFORME_FASE2.md` | segundos |

Los pasos 1–3 son independientes entre sí; 4→5→6 son secuenciales. **Comprobado el 28/10/2026** (Windows 11, Python 3.14.3, NumPy 2.4.2, Matplotlib 3.10.8): se copió el repositorio a una carpeta temporal fuera de él, se ejecutaron los pasos 1–5 lanzándolos desde una carpeta distinta, y los **50 archivos** de `resultados/` y `figuras/` (Fase 1 y Fase 2: CSV, JSON, MD y PNG) salieron **idénticos byte a byte** a los versionados. Los tiempos de la tabla son los medidos. Con otra versión de Matplotlib los PNG pueden diferir aunque los números coincidan; compara entonces los CSV.

Comparar una copia regenerada con el original (ajusta la ruta de la copia):

```python
import hashlib, pathlib
orig, copia = pathlib.Path("."), pathlib.Path("/tmp/copia_repo")
for c in ("analisis_adicional/resultados", "analisis_adicional/figuras",
          "analisis_adicional/fase2_evolucion/resultados", "analisis_adicional/fase2_evolucion/figuras"):
    for p in sorted((orig / c).rglob("*")):
        q = copia / p.relative_to(orig)
        if p.is_file() and (not q.exists() or hashlib.sha256(p.read_bytes()).digest() != hashlib.sha256(q.read_bytes()).digest()):
            print("DISTINTO:", p)
```

(`verificacion_fase2.csv` y `verificacion_fase2_ejecucion_28oct.csv` no se regeneran en los pasos 1–5.)

## 4. Reentrenar (solo si se desea)

**No es necesario para verificar nada**: los 19 `.npz` (540 entrenamientos: 500 nuevos y 40 réplicas de Marco) están en `fase2_evolucion/artefactos/` y `entrenar.py` **no sobrescribe** un artefacto existente (se salta y avisa). Para repetirlos hay que trabajar en una copia o con otra carpeta de artefactos, porque todos los grupos ya existen. Con un intérprete del entorno de entrenamiento:

```bash
PY=~/.venvs/taller_rl_fase2/bin/python        # Windows: $HOME\.venvs\taller_rl_fase2\Scripts\python.exe
$PY -B -X utf8 analisis_adicional/fase2_evolucion/entrenar.py --listar          # plan y tamaño, sin entrenar
$PY -B -X utf8 analisis_adicional/fase2_evolucion/entrenar.py --grupos REP      # puerta de reproducibilidad: 40 corridas de Marco, deben salir idénticas
$PY -B -X utf8 analisis_adicional/fase2_evolucion/entrenar.py --grupos todos    # el resto; ≈ 50 min con 16 procesos (28/10/2026: 16:28–17:19)
python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --procedencia
```

`REP` va primero y, si no reproduce a Marco bit a bit, el script **aborta antes de entrenar el resto**. Los hiperparámetros no se eligen en `entrenar.py`: están fijados de antemano en `utilidades.plan_entrenamientos()`. Los entrenamientos de la Fase 2 **no son históricos**: se hicieron el 28/10/2026 con el código original de Marco.

Con otro NumPy o Python, la reproducción bit a bit **puede no cumplirse**; la puerta `REP` lo detectaría. En el equipo del autor, 40/40 réplicas fueron idénticas con Python 3.12.10 + NumPy 2.5.4 + Gymnasium 1.4.0.

## 5. Rutas de otro equipo

Los scripts no contienen rutas absolutas operativas. Las rutas `C:\Users\dgall\...` que aparecen en `INFORME_FASE2.md`, `informe_plantilla.md`, `AUDITORIA_REPOSITORIO.md`, `CONTRIBUCIONES_DANI.md` y en un comentario de `entrenar.py` son **registro de dónde se ejecutó** (el entorno de entrenamiento del autor) y no se usan.
