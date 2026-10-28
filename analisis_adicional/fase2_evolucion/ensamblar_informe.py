"""
ensamblar_informe.py — Fase 2: genera INFORME_FASE2.md sustituyendo los marcadores {{Tn}} de informe_plantilla.md por las
tablas de resultados/tablas_informe.md y {{VERIF}} por resultados/verificacion_fase2.csv. Sin cifras transcritas a mano.

    python -B -X utf8 analisis_adicional/fase2_evolucion/ensamblar_informe.py
"""
import sys

sys.dont_write_bytecode = True

import csv
import re

import utilidades as u

tablas = {}
for bloque in (u.RES / "tablas_informe.md").read_text(encoding="utf-8").split("### ")[1:]:
    cab, _, cuerpo = bloque.partition("\n\n")
    clave, _, titulo = cab.partition(" — ")
    tablas[clave.strip()] = (titulo.strip(), cuerpo.strip())

plantilla = (u.F2 / "informe_plantilla.md").read_text(encoding="utf-8")


def tabla(m):
    clave = m.group(1)
    titulo, cuerpo = tablas[clave]
    return f"**Tabla {clave} — {titulo}**\n\n{cuerpo}"


texto = re.sub(r"\{\{(T\d+b?c?)\}\}", tabla, plantilla)
ruta = u.RES / "verificacion_fase2.csv"
if ruta.exists():
    filas = list(csv.DictReader(open(ruta, encoding="utf-8")))
    verif = "| Comprobación | Elementos | Detalle | Resultado |\n|---|---|---|---|\n" + "\n".join(
        f"| {f['comprobacion']} | {f['n_elementos']} | {f['detalle']} | {f['resultado']} |" for f in filas)
else:
    verif = "(ejecutar verificar_fase2.py)"
texto = texto.replace("{{VERIF}}", verif)
assert "{{" not in texto, re.findall(r"\{\{[^}]*\}\}", texto)
(u.F2 / "INFORME_FASE2.md").write_text(texto, encoding="utf-8")
print("INFORME_FASE2.md escrito:", len(texto.split()), "palabras")
