"""Verifica que entrega.json sea válido y coherente con el registro de la búsqueda (sin GPU).

Comprueba:
  1. entrega.json es JSON válido con las claves grupo, config y semana.
  2. Cada ranura tiene un índice entero entre 0 y 7, y la temperatura es 0.0, 0.3 o 0.7.
  3. La configuración entregada es la de mayor precisión en resultados/registro_busqueda.csv
     (la misma que justifica el notebook de búsqueda).

Uso, desde la raíz del repositorio:  python pruebas/verificar_entrega.py
Requiere solo Python 3.9+.
"""

import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RANURAS = ["rol", "estrategia", "formato", "estilo", "cierre"]
TEMPERATURAS = {0.0, 0.3, 0.7}

errores = []

# 1. Estructura
entrega = json.loads((RAIZ / "entrega.json").read_text(encoding="utf-8"))
for clave in ("grupo", "config", "semana"):
    if clave not in entrega:
        errores.append(f"falta la clave '{clave}'")
config = entrega.get("config", {})

# 2. Valores permitidos
for r in RANURAS:
    v = config.get(r)
    if not isinstance(v, int) or not 0 <= v <= 7:
        errores.append(f"{r} = {v!r} no es un índice entero entre 0 y 7")
if float(config.get("temperatura", -1)) not in TEMPERATURAS:
    errores.append(f"temperatura = {config.get('temperatura')!r} no está en {sorted(TEMPERATURAS)}")
if not isinstance(entrega.get("semana"), int):
    errores.append("semana debe ser un entero")

# 3. Coherencia con el registro de la búsqueda
with open(RAIZ / "resultados" / "registro_busqueda.csv", encoding="utf-8") as f:
    filas = list(csv.DictReader(f))
mejor = max(filas, key=lambda x: float(x["precision"]))   # la primera con la precisión máxima
config_mejor = {r: int(mejor[r]) for r in RANURAS}
config_mejor["temperatura"] = float(mejor["temperatura"])
config_entregada = {r: config.get(r) for r in RANURAS}
config_entregada["temperatura"] = float(config.get("temperatura", -1))
if config_entregada != config_mejor:
    errores.append(f"entrega.json {config_entregada} no coincide con la mejor del registro {config_mejor}")

print(f"Evaluaciones en el registro: {len(filas)}")
print(f"Mejor del registro: {config_mejor}  precisión {float(mejor['precision']):.1%} (paso {mejor['paso']})")
print(f"Configuración entregada: {config_entregada}")
if errores:
    print("\nERRORES:")
    for e in errores:
        print("  -", e)
    sys.exit(1)
print("OK: entrega.json es válido y coincide con la mejor configuración del registro.")
