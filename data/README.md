# Serie real del notebook 03

- **Serie:** SP500, cierre diario del índice S&P 500, sin dividendos.
- **Fuente:** FRED / S&P Dow Jones Indices, <https://fred.stlouisfed.org/series/SP500>.
- **Descarga:** 2026-10-28; periodo fijo 2017-01-03 a 2025-12-31.
- **Ficheros:** CSV normalizado con 2 262 sesiones y JSON con URL de descarga y SHA-256.
- **Limpieza:** eliminar cierres ausentes de festivos; ordenar fechas; no interpolar ni rellenar.
- **Derechos:** la fuente atribuye los datos a S&P Dow Jones Indices y publica sus condiciones
  en las notas de la serie. La copia local documenta el experimento académico; no se declara
  que la serie sea de dominio público.

## Uso experimental

Train: 2017–2021 (1 259 sesiones). Validación: 2022–2023 (501). Test: 2024–2025 (502).
Los bins se ajustan solo en train. Se entrenan ventanas cronológicas de 252 cierres muestreadas
únicamente dentro de train; las ventanas pueden solaparse. Validación y test mantienen su orden
completo. Para iniciar un split se incluyen cinco cierres anteriores como calentamiento; los
indicadores usan hasta 30 precios pasados, nunca futuros.

Se simula exposición fraccionaria al índice con la contabilidad del taller. El índice no es
directamente negociable; el análisis no incluye dividendos, spread, deslizamiento ni intereses
del efectivo. Las semillas son distintos entrenamientos sobre la misma historia, no historias
financieras independientes. El notebook no descarga datos al ejecutarse y verifica el SHA-256
antes de utilizarlos.
