"""
manifiesto.py — huellas de integridad y manifiesto de procedencia (Fases 2 a 4).

Uso (desde cualquier carpeta; en Windows sustituye `python` por `py -3`; añade -X utf8 si la consola no es UTF-8):

    python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --huellas              (Fase 2, una sola vez, ANTES de analizar)
    python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --procedencia          (tras entrenar)
    python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --registrar            (Fase 4, una sola vez: crea manifiesto_integridad.json)
    python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --actualizar-adiciones (tras editar a propósito documentos o código declarados)
    python -B -X utf8 analisis_adicional/fase2_evolucion/manifiesto.py --comprobar            (verifica TODO; código de salida 0 sólo si todo coincide)

Qué se comprueba (cada categoría contra una lista EXPLÍCITA de archivos y huellas sha256; nunca contra "lo que haya en la carpeta"):

  original_marco     código, bitácora, notebook, resultados/ y figuras/ de Marco        manifiesto_huellas_inicio.json   (Fase 2, inmutable)
  fase1              todo lo que había en analisis_adicional/ al empezar la Fase 2       manifiesto_huellas_inicio.json   (Fase 2, inmutable)
  fase2_artefactos   .npz, resultados/, figuras/ y registro de la Fase 2                 manifiesto_integridad.json       (Fase 4, inmutable)
  fase2_codigo       scripts e informes de la Fase 2, con la huella anterior a la Fase 4  manifiesto_integridad.json       (editable con --actualizar-adiciones)
  adiciones_fase3/4  documentos y archivos de reproducibilidad incorporados después      manifiesto_integridad.json       (editable con --actualizar-adiciones)

Un archivo de analisis_adicional/ que no esté en ninguna lista (ni sea una salida regenerada declarada) es un FALLO: hay que declararlo
con --actualizar-adiciones, de modo que quede constancia. Las diferencias de sólo saltos de línea (CRLF/LF, p. ej. tras clonar en Linux
con Git) se aceptan y se cuentan aparte: el hash LF-normalizado de los archivos de texto también está registrado.
"""
import csv
import json
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import utilidades as u  # noqa: E402

HUELLAS = u.F2 / "manifiesto_huellas_inicio.json"
INTEGRIDAD = u.F2 / "manifiesto_integridad.json"
CAT_FASE3 = "adiciones_fase3"
CAT_FASE4 = "adiciones_fase4"
FASE3_DOCS = {"AUDITORIA_REPOSITORIO.md", "CONTRIBUCIONES_DANI.md", "MATRIZ_EVIDENCIAS.md", "PROPUESTA_ENTREGA.md", "RESUMEN_PARA_EQUIPO.md"}
# Salida de verificar_fase2.py que se reescribe en cada ejecución (contiene el estado de git del momento): no se hashea, pero se declara.
REGENERADOS = ["analisis_adicional/fase2_evolucion/resultados/verificacion_fase2.csv"]
# El propio manifiesto no puede contener su huella.
AUTORREFERENCIA = ["analisis_adicional/fase2_evolucion/manifiesto_integridad.json"]
RAIZ_F2 = "analisis_adicional/fase2_evolucion/"

MOTIVOS = {   # por qué existe cada archivo añadido (Fases 3-4)
    "AUDITORIA_REPOSITORIO.md": "Fase 3: auditoría del repositorio, riesgos de publicación y comandos propuestos",
    "CONTRIBUCIONES_DANI.md": "Fase 3: contribuciones con procedencia (original / reutilizado / reproducido / nuevo)",
    "MATRIZ_EVIDENCIAS.md": "Fase 3: matriz pregunta-hipótesis-experimento-resultado-evidencia",
    "PROPUESTA_ENTREGA.md": "Fase 3: contenido propuesto para el PDF de 2 páginas y la defensa",
    "RESUMEN_PARA_EQUIPO.md": "Fase 3: resumen para Marco y Cristian con decisiones pendientes",
    "README_REPRODUCCION.md": "Fase 4: entornos, orden de ejecución, verificación sin reentrenar y reentrenamiento opcional",
    "requirements_analisis.txt": "Fase 4: dependencias verificadas del análisis (Fases 1-2) y las figuras",
    "requirements_entrenamiento.txt": "Fase 4: dependencias verificadas del entrenamiento (Fase 2)",
}
CAMBIOS_FASE4 = {   # ediciones de la Fase 4 en archivos de la Fase 2 (ninguna cambia resultados ni algoritmos)
    "entrenar.py": "docstring sin ruta personal; INICIO sólo se registra si hay grupos pendientes (verificar_fase2.py V5 ya no añade líneas al registro)",
    "verificar_fase2.py": "V4 por categorías con listas declaradas; V5 localiza el intérprete con TALLER_RL_PYTHON o ~/.venvs (Windows y POSIX); V6 admite commits posteriores",
    "manifiesto.py": "comprobación por categorías (esta versión); --registrar y --actualizar-adiciones",
    "utilidades.py": "sha256_lf y aclaración del alcance de archivos_protegidos()",
    "informe_plantilla.md": "aclara que las rutas C:\\Users\\dgall\\... son del entorno del autor y apunta al README",
    "INFORME_FASE2.md": "idem informe_plantilla.md (mismo texto aplicado al informe ya ensamblado)",
    "analisis_fase2.py": "sólo docstring (comando genérico)",
    "ensamblar_informe.py": "sólo docstring (comando genérico)",
    "generar_tablas_informe.py": "sólo docstring (comando genérico)",
}


def git(*args):
    return subprocess.run(["git", *args], cwd=u.RAIZ, capture_output=True, text=True).stdout.strip()


def rel(p):
    return str(p.relative_to(u.RAIZ)).replace("\\", "/")


def archivos_analisis():
    """Todos los archivos de analisis_adicional/ (sin cachés de bytecode)."""
    return {rel(p): p for p in sorted(u.FASE1.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def hay_caches():
    return sorted(rel(p) for p in u.FASE1.rglob("*") if "__pycache__" in p.parts or p.suffix == ".pyc")


# ------------------------------------------------------------------------------------------------
# Fase 2: instantánea de inicio y procedencia (sin cambios de comportamiento)
# ------------------------------------------------------------------------------------------------
def huellas():
    if HUELLAS.exists():
        print(f"{HUELLAS.name} ya existe: no se sobrescribe (es la referencia de integridad).")
        return
    datos = dict(rama=git("branch", "--show-current"), head=git("rev-parse", "HEAD"), huellas=u.huellas_protegidas())
    HUELLAS.write_text(json.dumps(datos, indent=1), encoding="utf-8")
    u.registrar(f"HUELLAS de inicio guardadas: {len(datos['huellas'])} archivos protegidos, HEAD={datos['head'][:7]}")
    print(f"{len(datos['huellas'])} archivos protegidos registrados; HEAD={datos['head']}")


def procedencia():
    etapas = u.marco_etapas()
    fecha_marco = git("log", "-1", "--format=%ci", "--", "resultados/alpha.pkl")
    filas = []
    for etapa, runs in etapas.items():
        for r in runs:
            filas.append(dict(origen="Marco (histórico, commit 5a6e1ee)", fecha_generacion=fecha_marco, fuente=f"resultados/{etapa}.pkl",
                              grupo_o_etapa=etapa, variante=r["variante"], generador=r["generador"], algoritmo=r["algoritmo"], seed=r["seed"],
                              n_episodios=r["cfg"]["n_episodios"], alpha=r["cfg"]["alpha"], alpha_decay=r["cfg"]["alpha_decay"],
                              gamma=r["cfg"]["gamma"], eps_inicial=r["cfg"]["eps_inicial"], eps_final=r["cfg"]["eps_final"],
                              frac_exploracion=r["cfg"]["frac_exploracion"], n_bins=r["cfg"]["n_bins"], discretizacion=r["cfg"]["discretizacion"],
                              tiempo_s=round(float(r["tiempo"]), 2)))
    for g in u.grupos_existentes():
        for r in u.cargar_grupo(g):
            filas.append(dict(origen="Fase 2 (generado en esta ejecución, 2026-10-28)", fecha_generacion="2026-10-28", fuente=f"artefactos/{g}.npz",
                              grupo_o_etapa=g, variante="", generador=r["generador"], algoritmo=r["algoritmo"], seed=r["seed"],
                              n_episodios=r["cfg"]["n_episodios"], alpha=r["cfg"]["alpha"], alpha_decay=r["cfg"]["alpha_decay"],
                              gamma=r["cfg"]["gamma"], eps_inicial=r["cfg"]["eps_inicial"], eps_final=r["cfg"]["eps_final"],
                              frac_exploracion=r["cfg"]["frac_exploracion"], n_bins=r["cfg"]["n_bins"], discretizacion=r["cfg"]["discretizacion"],
                              tiempo_s=round(float(r["tiempo"]), 2)))
    u.asegurar()
    with open(u.RES / "manifiesto_corridas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    resumen = dict(
        evaluacion=dict(val=f"{u.N_VAL} series, seed {u.SEED_VAL}", test=f"{u.N_TEST} series, seed {u.SEED_TEST}", n_steps=500,
                        curva_validacion="10 primeras series de validación cada 100 episodios", semillas_entrenamiento=u.SEMILLAS),
        corridas_marco=sum(f["origen"].startswith("Marco") for f in filas), corridas_fase2=sum(f["origen"].startswith("Fase 2") for f in filas),
        grupos_fase2={g: len(u.cargar_grupo(g)) for g in u.grupos_existentes()},
        nota="Las corridas de Fase 2 se ejecutaron el 2026-10-28 con el código original de trader_rl.py en un venv aislado; NO son histórico. "
             "El grupo REP duplica a propósito 40 corridas de Marco para verificar la reproducción bit a bit.")
    (u.RES / "manifiesto_procedencia.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(resumen["corridas_marco"], "corridas de Marco;", resumen["corridas_fase2"], "de Fase 2")


# ------------------------------------------------------------------------------------------------
# Fase 4: manifiesto de integridad con listas explícitas por categoría
# ------------------------------------------------------------------------------------------------
def _entrada(ruta):
    return dict(sha256=u.sha256(ruta), sha256_lf=u.sha256_lf(ruta))


def _leer_hashes_previos(archivo):
    """Formato `sha256sum` (hash *ruta). Devuelve {ruta relativa a la raíz del repositorio: hash}."""
    previos = {}
    for linea in open(archivo, encoding="utf-8"):
        h, _, ruta = linea.rstrip("\n").partition(" ")
        previos[ruta.lstrip("*").replace("\\", "/")] = h
    return previos


def _clasificar(ref, todos):
    """Reparte los archivos de analisis_adicional/ que NO son de la instantánea de inicio en las categorías de la Fase 2 y de adiciones."""
    art, cod, adi = [], [], []
    for k in todos:
        if k in ref["huellas"] or k in REGENERADOS or k in AUTORREFERENCIA:
            continue
        if k.startswith(RAIZ_F2):
            sub = k[len(RAIZ_F2):]
            if "/" in sub or sub == "registro_ejecucion.log":
                art.append(k)      # artefactos/, resultados/, figuras/ y el registro de ejecución (sólo se añade, no se edita)
            else:
                cod.append(k)      # scripts, informes y la instantánea de inicio
        else:
            adi.append(k)
    return art, cod, adi


def _cargar_integridad():
    return json.loads(INTEGRIDAD.read_text(encoding="utf-8")) if INTEGRIDAD.exists() else None


def _adiciones(ruta_rel):
    nombre = ruta_rel.rsplit("/", 1)[-1]
    return dict(categoria=CAT_FASE3 if nombre in FASE3_DOCS else CAT_FASE4, motivo=MOTIVOS.get(nombre, "SIN MOTIVO DECLARADO"))


def _cambio(k, previos, sha_actual):
    """Qué le hizo la Fase 4 a un archivo de la Fase 2, comparando con su huella anterior (sha256_fase2)."""
    if not previos:
        return ""
    if k not in previos:
        return "nuevo en Fase 4"
    if previos[k] == sha_actual:
        return "sin cambios"
    return CAMBIOS_FASE4.get(k.rsplit("/", 1)[-1], "modificado sin descripción")


def registrar(hashes_previos=None):
    """Crea manifiesto_integridad.json. Se niega a sobrescribirlo (usa --actualizar-adiciones para documentos y código declarados)."""
    if INTEGRIDAD.exists():
        print(f"{INTEGRIDAD.name} ya existe: no se sobrescribe. Usa --actualizar-adiciones para documentos y código.")
        return False
    ref = json.loads(HUELLAS.read_text(encoding="utf-8"))
    todos = archivos_analisis()
    alterados = [k for k, h in ref["huellas"].items() if not (u.RAIZ / k).is_file() or u.sha256(u.RAIZ / k) != h]
    if alterados:
        print("No se registra sobre un estado alterado; archivos originales que no coinciden:", alterados)
        return False
    previos = _leer_hashes_previos(hashes_previos) if hashes_previos else {}
    art, cod, adi = _clasificar(ref, todos)
    datos = dict(
        descripcion="Manifiesto de integridad de la Fase 4. Complementa manifiesto_huellas_inicio.json (Fase 2). Ver manifiesto.py.",
        creado=time.strftime("%Y-%m-%d"), rama_al_crear=git("branch", "--show-current"), head_al_crear=git("rev-parse", "HEAD"),
        originales_lf={k: u.sha256_lf(u.RAIZ / k) for k in ref["huellas"]},
        fase2_artefactos={k: _entrada(todos[k]) for k in art},
        fase2_codigo={k: dict(sha256_fase2=previos.get(k), **_entrada(todos[k]), cambio_fase4=_cambio(k, previos, u.sha256(todos[k]))) for k in cod},
        adiciones={k: dict(**_adiciones(k), **_entrada(todos[k])) for k in adi},
        regenerados_excluidos=REGENERADOS, autorreferencia_excluida=AUTORREFERENCIA,
        historial=[dict(fecha=time.strftime("%Y-%m-%d %H:%M"), accion="registrar", detalle=f"{len(art)} artefactos, {len(cod)} de código/informes, {len(adi)} adiciones")])
    INTEGRIDAD.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"manifiesto_integridad.json creado: {len(ref['huellas'])} originales, {len(art)} artefactos Fase 2, {len(cod)} código/informes Fase 2, {len(adi)} adiciones")
    return True


def actualizar_adiciones():
    """Vuelve a fijar las huellas de código/informes de la Fase 2 y de las adiciones tras editarlos A PROPÓSITO. No toca originales ni artefactos."""
    m = _cargar_integridad()
    ref = json.loads(HUELLAS.read_text(encoding="utf-8"))
    todos = archivos_analisis()
    art, cod, adi = _clasificar(ref, todos)
    mal = [k for k in art if k not in m["fase2_artefactos"]] + [k for k in m["fase2_artefactos"] if k not in todos]
    if mal or any(u.sha256(todos[k]) != v["sha256"] for k, v in m["fase2_artefactos"].items() if k in todos):
        print("Los artefactos de la Fase 2 no coinciden con su registro; no se actualiza nada:", mal or "huellas distintas")
        return False
    cambios = []
    for k in cod:
        previo = m["fase2_codigo"].get(k)
        nuevo = _entrada(todos[k])
        if previo is None:
            m["fase2_codigo"][k] = dict(sha256_fase2=None, **nuevo, cambio_fase4="nuevo en Fase 4")
            cambios.append(k)
        elif previo["sha256"] != nuevo["sha256"]:
            previo.update(nuevo)
            if previo.get("sha256_fase2") and previo["sha256_fase2"] != nuevo["sha256"] and previo.get("cambio_fase4") in (None, "", "sin cambios"):
                previo["cambio_fase4"] = CAMBIOS_FASE4.get(k.rsplit("/", 1)[-1], "modificado sin descripción")
            cambios.append(k)
    for k in adi:
        previo = m["adiciones"].get(k)
        nuevo = _entrada(todos[k])
        if previo is None:
            m["adiciones"][k] = dict(**_adiciones(k), **nuevo)
            cambios.append(k)
        elif previo["sha256"] != nuevo["sha256"]:
            previo.update(nuevo)
            cambios.append(k)
    for categoria in ("fase2_codigo", "adiciones"):
        for k in [k for k in m[categoria] if k not in todos]:
            print("Declarado pero ausente (se elimina del registro y consta en el historial):", k)
            del m[categoria][k]
            cambios.append(f"ELIMINADO {k}")
    m["historial"].append(dict(fecha=time.strftime("%Y-%m-%d %H:%M"), accion="actualizar-adiciones", detalle=cambios))
    INTEGRIDAD.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"actualizado: {len(cambios)} cambios", *cambios, sep="\n  ")
    return True


def _evaluar(entradas):
    """entradas: {ruta relativa: (sha256, sha256_lf)}."""
    r = dict(n=len(entradas), identicos=0, solo_eol=0, cambiados=[], ausentes=[])
    for k, (sha, lf) in entradas.items():
        p = u.RAIZ / k
        if not p.is_file():
            r["ausentes"].append(k)
        elif u.sha256(p) == sha:
            r["identicos"] += 1
        elif lf and u.sha256_lf(p) == lf:
            r["solo_eol"] += 1
        else:
            r["cambiados"].append(k)
    return r


def evaluar_integridad():
    """Devuelve {categoría: resultado} más 'no_declarados' y 'caches'. Cada categoría se compara con su lista explícita."""
    ref = json.loads(HUELLAS.read_text(encoding="utf-8"))
    m = _cargar_integridad()
    lf = m["originales_lf"] if m else {}
    cats = {
        "original_marco": {k: (h, lf.get(k)) for k, h in ref["huellas"].items() if not k.startswith("analisis_adicional/")},
        "fase1": {k: (h, lf.get(k)) for k, h in ref["huellas"].items() if k.startswith("analisis_adicional/")},
    }
    declarados = set(ref["huellas"]) | set(REGENERADOS) | set(AUTORREFERENCIA)
    if m:
        cats["fase2_artefactos"] = {k: (v["sha256"], v["sha256_lf"]) for k, v in m["fase2_artefactos"].items()}
        cats["fase2_codigo"] = {k: (v["sha256"], v["sha256_lf"]) for k, v in m["fase2_codigo"].items()}
        for c in (CAT_FASE3, CAT_FASE4):
            cats[c] = {k: (v["sha256"], v["sha256_lf"]) for k, v in m["adiciones"].items() if v["categoria"] == c}
        declarados |= set(m["fase2_artefactos"]) | set(m["fase2_codigo"]) | set(m["adiciones"])
    res = {c: _evaluar(e) for c, e in cats.items()}
    res["no_declarados"] = [k for k in archivos_analisis() if k not in declarados]
    res["caches"] = hay_caches()
    res["_manifiesto_integridad"] = m is not None
    return res


def resumen_integridad(res):
    """(ok, líneas de texto)."""
    lineas, ok = [], res["_manifiesto_integridad"]
    if not ok:
        lineas.append("FALTA manifiesto_integridad.json (ejecuta --registrar)")
    for c, r in res.items():
        if c in ("no_declarados", "caches", "_manifiesto_integridad"):
            continue
        bien = not r["cambiados"] and not r["ausentes"]
        ok &= bien
        extra = f", {r['solo_eol']} sólo en saltos de línea" if r["solo_eol"] else ""
        lineas.append(f"{c:18s} {r['n']:3d} archivos: {r['identicos']} idénticos{extra}, {len(r['cambiados'])} modificados, {len(r['ausentes'])} ausentes")
        lineas += [f"  ! modificado: {k}" for k in r["cambiados"]] + [f"  ! ausente: {k}" for k in r["ausentes"]]
    ok &= not res["no_declarados"] and not res["caches"]
    lineas.append(f"no declarados      {len(res['no_declarados'])}; cachés de bytecode {len(res['caches'])}")
    lineas += [f"  ! sin declarar: {k}" for k in res["no_declarados"]] + [f"  ! caché: {k}" for k in res["caches"]]
    return ok, lineas


def comprobar():
    ref = json.loads(HUELLAS.read_text(encoding="utf-8"))
    ok, lineas = resumen_integridad(evaluar_integridad())
    print(*lineas, sep="\n")
    print("HEAD inicial:", ref["head"][:7], "| HEAD actual:", git("rev-parse", "HEAD")[:7], "| rama:", git("branch", "--show-current"))
    print("excluidos (se reescriben en cada verificación):", *REGENERADOS)
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--huellas" in args:
        huellas()
    if "--procedencia" in args:
        procedencia()
    if "--registrar" in args:
        previos = args[args.index("--hashes-previos") + 1] if "--hashes-previos" in args else None
        if not registrar(previos):
            sys.exit(1)
    if "--actualizar-adiciones" in args:
        if not actualizar_adiciones():
            sys.exit(1)
    if "--comprobar" in args:
        sys.exit(0 if comprobar() else 1)
