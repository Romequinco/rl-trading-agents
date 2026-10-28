# 03 · Conclusiones de los extras

Ejecución completa: 5 semillas, 800 episodios, 200 historias sintéticas de test. Configuración fija y comparación emparejada.

## Estado enriquecido

- **Reversión a la media · qlearning:** mejora con intervalo por encima de cero; mediana del delta +23.1 puntos porcentuales, IC95 condicional [+20.5, +26.5] puntos.
- **Reversión a la media · sarsa:** mejora con intervalo por encima de cero; mediana del delta +23.7 puntos porcentuales, IC95 condicional [+20.1, +28.2] puntos.
- **Regímenes · qlearning:** mejora con intervalo por encima de cero; mediana del delta +7.8 puntos porcentuales, IC95 condicional [+5.5, +10.6] puntos.
- **Regímenes · sarsa:** mejora con intervalo por encima de cero; mediana del delta +7.8 puntos porcentuales, IC95 condicional [+5.7, +9.8] puntos.
- **Tendencia persistente · qlearning:** mejora con intervalo por encima de cero; mediana del delta +19.1 puntos porcentuales, IC95 condicional [+16.1, +23.4] puntos.
- **Tendencia persistente · sarsa:** mejora con intervalo por encima de cero; mediana del delta +18.4 puntos porcentuales, IC95 condicional [+15.8, +20.1] puntos.
- **Difusión + saltos · qlearning:** sin mejora concluyente con este protocolo; mediana del delta +0.0 puntos porcentuales, IC95 condicional [+0.0, +0.0] puntos.
- **Difusión + saltos · sarsa:** sin mejora concluyente con este protocolo; mediana del delta +0.0 puntos porcentuales, IC95 condicional [+0.0, +0.0] puntos.

Los intervalos condicionan a las políticas entrenadas y no corrigen comparaciones múltiples. El estado crece de 20 a 180 celdas sin aumentar el presupuesto; esto mezcla información adicional y coste estadístico.

## Mercados extra

- **Tendencia persistente:** Q enriquecido +123.8%, SARSA enriquecido +128.7%, buy & hold +22.8%; efectivo 0 %.
- **Difusión + saltos:** Q enriquecido +0.0%, SARSA enriquecido +0.0%, buy & hold -22.4%; efectivo 0 %.

La tendencia contiene persistencia aprovechable; los saltos son independientes. Superar buy & hold permaneciendo fuera no demuestra anticipación de saltos.

## Serie real

- Buy & hold en el recorrido completo 2024–2025: **+44.2%**.
- **qlearning_base:** mediana +45.9%, rango entre semillas [+45.9%, +45.9%], drawdown mediano 18.9%, 1 operaciones.
- **qlearning_enriquecido:** mediana +28.3%, rango entre semillas [+27.7%, +46.4%], drawdown mediano 11.4%, 79 operaciones.
- **sarsa_base:** mediana +45.9%, rango entre semillas [+45.9%, +45.9%], drawdown mediano 18.9%, 1 operaciones.
- **sarsa_enriquecido:** mediana +29.9%, rango entre semillas [+17.8%, +38.5%], drawdown mediano 12.8%, 77 operaciones.
- **Balance de la ampliación en qlearning:** menos retorno mediano y menor drawdown. La evaluación exige leer ambas medidas conjuntamente.
- **Balance de la ampliación en sarsa:** menos retorno mediano y menor drawdown. La evaluación exige leer ambas medidas conjuntamente.

Los agentes base hacen como máximo una operación en mediana. Su diferencia frente a buy & hold puede proceder del momento de entrada; no implica haber aprendido una estrategia activa de anticipación.

La historia real de test es única y los entrenamientos reutilizan ventanas del pasado. No se puede inferir una ventaja persistente a partir de estos dos años ni de cinco semillas sobre los mismos precios. Los bloques muestran variación temporal, no nuevas historias independientes.

## Límites y lectura crítica

- Se conserva la recompensa del taller: suma retornos simples con comisión, mientras que el balance compone. Una mayor recompensa de entrenamiento no garantiza mayor rentabilidad final.
- Los indicadores ampliados no garantizan un estado markoviano. La comparación no es una búsqueda exhaustiva de estados o hiperparámetros.
- Se añaden MA30 y volatilidad conjuntamente: esta prueba no permite atribuir la mejora a cada indicador por separado.
- La convergencia es un resumen descriptivo de la curva; puede reflejar el descenso de exploración y no una estrategia estable.
- La evaluación real usa el índice de precios, sin dividendos, spread, deslizamiento ni intereses del efectivo; no es un activo directamente negociable y no constituye validación operativa.

Fuente de la serie real: https://fred.stlouisfed.org/series/SP500. 2262 cierres; huella de los datos: `39dc70d1cc353c0e89a1ef9d6c99be00930859a6460a553d2ef2ac5a8496f1dd`.
