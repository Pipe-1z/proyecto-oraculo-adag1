"""Prueba sintética del ascenso por coordenadas guiado por trazas (sin GPU).

Reproduce la lógica de la sección 7 del notebook de búsqueda contra un
oráculo sintético con estructura conocida, para verificar que el algoritmo:

  1. termina (converge o llega a MAX_PASADAS),
  2. nunca empeora la mejor precisión (ascenso monótono),
  3. respeta la cota de evaluaciones 1 + I * sum(k_p - 1),
  4. encuentra el óptimo cuando las ranuras son independientes.

Uso:  python pruebas/prueba_sintetica.py
Requiere solo Python 3.9+ (sin librerías externas).
"""

import itertools
import random
from collections import Counter

RANURAS = ["rol", "estrategia", "formato", "estilo", "cierre"]
K = 8                  # opciones por ranura
N = 100                # instancias sintéticas
MAX_PASADAS = 3
UMBRAL_TRAZA = 3
SEMILLA = 7

# ---------- Oráculo sintético ----------
# Cada ranura actúa sobre su propia familia de restricciones y tiene una sola
# opción "limpia"; cualquier otra opción rompe la familia de esa ranura.
# Cada instancia trae entre 1 y 3 familias y pasa si ninguna está rota
# (todo o nada). Las 5 primeras instancias traen una sola familia cada una,
# así que arreglar cualquier ranura mejora la precisión: las ranuras son
# independientes y el ascenso por coordenadas debe hallar el óptimo.
FAMILIA_DE = {"rol": "first_word", "estrategia": "length", "formato": "format",
              "estilo": "case", "cierre": "last_word"}
FAMILIAS = list(FAMILIA_DE.values())
rng = random.Random(SEMILLA)
LIMPIA = {r: rng.randrange(K) for r in RANURAS}
INSTANCIAS = [[f] for f in FAMILIAS] + [rng.sample(FAMILIAS, rng.choice([1, 2, 3])) for _ in range(N - len(FAMILIAS))]

CONTADOR_CONSULTAS = 0
CACHE = {}


def evaluar(c):
    """Devuelve (precisión, Counter de fallos por familia). Cuesta N consultas si no está en caché."""
    global CONTADOR_CONSULTAS
    clave = tuple(c[r] for r in RANURAS)
    if clave in CACHE:
        return CACHE[clave]
    CONTADOR_CONSULTAS += N
    rotas = {FAMILIA_DE[r] for r in RANURAS if c[r] != LIMPIA[r]}
    fallos = Counter()
    aciertos = 0
    for fams in INSTANCIAS:
        violada = next((f for f in fams if f in rotas), None)
        if violada:
            fallos[violada] += 1
        else:
            aciertos += 1
    CACHE[clave] = (aciertos / N, fallos)
    return CACHE[clave]


# ---------- Ascenso por coordenadas con poda por trazas (misma lógica del notebook) ----------
def ascenso(inicial):
    actual = dict(inicial)
    p_actual, fallos_actual = evaluar(actual)
    evaluaciones = 1
    historial = [p_actual]
    descartadas = {r: set() for r in RANURAS}
    for pasada in range(1, MAX_PASADAS + 1):
        mejoro = False
        for ranura in RANURAS:
            base_p, base_f = p_actual, fallos_actual
            mejor_op, mejor_p, mejor_f = actual[ranura], p_actual, fallos_actual
            for op in range(K):
                if op == actual[ranura] or op in descartadas[ranura]:
                    continue
                cand = dict(actual, **{ranura: op})
                p, f = evaluar(cand)
                evaluaciones += 1
                historial.append(max(historial[-1], p))
                empeora = any(f[x] - base_f.get(x, 0) >= UMBRAL_TRAZA for x in f)
                if empeora and p <= base_p:
                    descartadas[ranura].add(op)
                if p > mejor_p:
                    mejor_op, mejor_p, mejor_f = op, p, f
            if mejor_op != actual[ranura]:
                actual[ranura] = mejor_op
                p_actual, fallos_actual = mejor_p, mejor_f
                mejoro = True
        if not mejoro:
            return actual, p_actual, pasada, evaluaciones, historial
    return actual, p_actual, MAX_PASADAS, evaluaciones, historial


if __name__ == "__main__":
    inicial = {r: 0 for r in RANURAS}
    mejor, p, pasadas, evals, hist = ascenso(inicial)

    optimo = max(evaluar(dict(zip(RANURAS, t)))[0]
                 for t in itertools.product(range(K), repeat=len(RANURAS)))

    cota = 1 + MAX_PASADAS * len(RANURAS) * (K - 1)
    print("Configuración limpia (oculta):", LIMPIA)
    print("Configuración encontrada:     ", mejor)
    print(f"Precisión encontrada: {p:.2f}   Óptimo global (fuerza bruta): {optimo:.2f}")
    print(f"Pasadas: {pasadas}   Evaluaciones: {evals}   Cota teórica: {cota}")

    assert all(a <= b for a, b in zip(hist, hist[1:])), "La mejor precisión no debe bajar"
    assert evals <= cota, "Se superó la cota 1 + I*sum(k_p - 1)"
    assert mejor == LIMPIA and abs(p - 1.0) < 1e-9, "Con ranuras independientes debe hallar el óptimo"
    print("OK: las 3 verificaciones pasaron.")
