# Proyecto Oráculo — ADAG1

**Análisis y Diseño de Algoritmos · 2026-B**
**Autor:** Juan Felipe Henao (trabajo individual) · **Subproyecto:** Proyecto Oráculo (Grupo 1)
**Versión:** primera entrega (solución base), rama `main`

Búsqueda de una configuración de prompt (un índice por ranura `rol`, `estrategia`, `formato`, `estilo`, `cierre`, más la temperatura) que maximiza la precisión de un modelo de lenguaje sobre instancias con restricciones verificables, consultando el oráculo del curso.

## Resultado

| | |
|---|---|
| Configuración entregada | `{"rol": 0, "estrategia": 0, "formato": 0, "estilo": 5, "cierre": 1, "temperatura": 0.0}` |
| Precisión en búsqueda (100 instancias) | 14,0 % |
| Precisión en validación (87 instancias, `oraculo.validar`) | 10,3 % |
| Método | Ascenso por coordenadas guiado por trazas de fallos |
| Costo | 60 configuraciones distintas evaluadas (68 evaluaciones, 6.800 consultas, ≈ 3,4 h de GPU T4) |

Es la **mejor configuración encontrada**, no un óptimo global demostrado: se exploró el 0,18 % de las 32.768 configuraciones.

## Contenido del repositorio

```
proyecto-oraculo-adag1/
├── README.md                      este archivo
├── entrega.json                   configuración ganadora (grupo, semana, índices y temperatura)
├── notebooks/
│   ├── notebook_busqueda.ipynb    cómo se obtuvo la configuración (con todas las salidas)
│   └── notebook_evaluacion.ipynb  carga entrega.json y la evalúa sobre datos de prueba
├── resultados/
│   └── registro_busqueda.csv      las 68 evaluaciones con su precisión y fallos por familia
├── pruebas/
│   ├── verificar_entrega.py       valida entrega.json contra el registro (sin GPU)
│   └── prueba_sintetica.py        prueba del algoritmo con un oráculo sintético (sin GPU)
└── docs/
    └── manual_tecnico.md          instalación, arquitectura, funcionamiento, uso y límites
```

## Requisitos

- **Notebooks:** Google Colab con GPU **T4** (Python 3.12 o superior en Colab). La primera celda de cada notebook instala las dependencias (`transformers`, `accelerate`, `bitsandbytes`, `nltk`, `spacy`, `emoji`, `langdetect`, `immutabledict`) y descarga `oraculo.py`, `ayudas.py` y `datos_visibles.json` del [repositorio del curso](https://github.com/DanielMelo404/Proyecto-Oraculo-AyD).
- **Pruebas:** Python 3.9 o superior, sin librerías externas.

## Cómo ejecutar

### Evaluar la configuración entregada
1. Abrir `notebooks/notebook_evaluacion.ipynb` en Colab y elegir GPU T4.
2. Ejecutar la celda 1 y reiniciar la sesión cuando lo pida.
3. Ejecutar todas las celdas. Si hay un archivo `.json` de instancias en `/content`, se evalúa sobre él; si no, se evalúa sobre la validación pública.

No hay que editar código, rutas ni parámetros.

### Reproducir la búsqueda
Abrir `notebooks/notebook_busqueda.ipynb` en Colab con GPU T4 y ejecutar en orden. El caché y el registro se guardan en Google Drive, así que una desconexión no hace perder lo ya evaluado. La corrida completa tomó unas 3,4 h de GPU.

### Ejemplo pequeño sin GPU
Desde la raíz del repositorio:

```bash
python pruebas/verificar_entrega.py
python pruebas/prueba_sintetica.py
```

Salida esperada:

```
OK: entrega.json es válido y coincide con la mejor configuración del registro.
...
Pasadas: 2   Evaluaciones: 64   Cota teórica: 106
OK: las 3 verificaciones pasaron.
```

## Reproducibilidad

- Semilla fija (1) y temperatura 0,0 (generación determinista).
- Parámetros de la búsqueda en una sola celda del notebook de búsqueda.
- La prueba sintética usa la semilla 7 y su resultado esperado está arriba.
- `entrega.json`, el notebook de búsqueda y el notebook de evaluación usan la misma configuración; `pruebas/verificar_entrega.py` lo comprueba.

Más detalle en [`docs/manual_tecnico.md`](docs/manual_tecnico.md).
