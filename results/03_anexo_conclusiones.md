# 03 · Anexo: pruebas de robustez

Ejecución completa: 5 semillas, 800 episodios de referencia, 200 historias sintéticas de test; 360 entrenamientos. Q-Learning y estado enriquecido salvo el factor estudiado; configuraciones fijadas antes del test.

## Qué indicador aporta

- **Reversión a la media:** base +18.1% · +MA30 +44.0% · +volatilidad +25.5% · ambos +42.8%. Mejor indicador por separado: MA30.
- **Regímenes:** base +0.0% · +MA30 +0.0% · +volatilidad +8.5% · ambos +7.8%. Mejor indicador por separado: volatilidad.
- **Tendencia persistente:** base +104.5% · +MA30 +120.6% · +volatilidad +109.0% · ambos +123.8%. Mejor indicador por separado: MA30.
- **S&P 500 · real:** base +45.9% · +MA30 +37.2% · +volatilidad +41.8% · ambos +28.3%. Mejor indicador por separado: volatilidad.

## Presupuesto, placebos, comisión, algoritmo y recompensa

- **Presupuesto:** con ×4 episodios el enriquecido pasa de Reversión a la media +42.8% → +42.8%, Regímenes +7.8% → +8.2%, Tendencia persistente +123.8% → +126.0%, S&P 500 · real +28.3% → +28.7%. Más entrenamiento no cierra la distancia en el S&P 500.
- **Placebos:** browniano base +0.0% y enriquecido +0.0%; S&P 500 barajado base +44.0% y enriquecido +11.1%, con buy & hold +44.2% en ambas series.
- **Comisión · Reversión a la media:** 0.00 % → +45.8% (61 ops.) · 0.05 % → +45.1% (45 ops.) · 0.10 % → +42.8% (27 ops.) · 0.25 % → +38.3% (16 ops.) · 0.50 % → +23.0% (6 ops.); buy & hold -0.3%.
- **Comisión · Regímenes:** 0.00 % → +9.6% (44 ops.) · 0.05 % → +8.9% (18 ops.) · 0.10 % → +7.8% (12 ops.) · 0.25 % → +0.0% (2 ops.) · 0.50 % → +0.0% (0 ops.); buy & hold -12.5%.
- **Comisión · Tendencia persistente:** 0.00 % → +128.0% (30 ops.) · 0.05 % → +126.5% (15 ops.) · 0.10 % → +123.8% (13 ops.) · 0.25 % → +123.6% (8 ops.) · 0.50 % → +79.6% (4 ops.); buy & hold +22.8%.
- **Algoritmo:** rango entre algoritmos de Reversión a la media [+42.3%, +43.9%], Regímenes [+6.5%, +7.8%], Tendencia persistente [+123.8%, +128.7%], S&P 500 · real [+28.3%, +32.5%].
- **Recompensa log:** Tendencia + ciclo +41092.3% → +43009.3%, Reversión a la media +42.8% → +42.8%, Regímenes +7.8% → +8.1%, Tendencia persistente +123.8% → +126.5%, S&P 500 · real +28.3% → +28.7%.

## Walk-forward anual del S&P 500

- **2019:** buy & hold +28.6% · base +16.5% · enriquecido +6.6% (caída máxima 6.8% · 7.1% · 7.9%).
- **2020:** buy & hold +15.2% · base +10.4% · enriquecido +1.0% (caída máxima 33.9% · 27.2% · 34.2%).
- **2021:** buy & hold +28.7% · base +25.3% · enriquecido +7.0% (caída máxima 5.2% · 5.2% · 5.9%).
- **2022:** buy & hold -20.0% · base -18.4% · enriquecido -20.7% (caída máxima 25.5% · 24.3% · 22.2%).
- **2023:** buy & hold +24.6% · base +21.3% · enriquecido +11.0% (caída máxima 10.3% · 10.3% · 10.0%).
- **2024:** buy & hold +23.9% · base +23.9% · enriquecido +17.4% (caída máxima 8.5% · 8.5% · 5.8%).
- **2025:** buy & hold +16.5% · base +14.9% · enriquecido +15.8% (caída máxima 18.9% · 18.9% · 12.0%).

## Lectura conjunta

- La mejora del estado enriquecido en sintéticos procede de **un indicador por mercado**, el que su mecanismo predice: distancia a MA30 en reversión y tendencia, volatilidad en regímenes.
- El estado de 180 celdas no está limitado por el entrenamiento, y el resultado apenas depende del algoritmo TD o de la forma de la recompensa: **lo que decide es qué información recibe el agente**.
- Los placebos se comportan como deben: sin señal, el agente no opera o se limita a comprar y mantener.
- En el S&P 500 ningún estado bate de forma consistente a buy & hold en siete años de test. Los indicadores reducen a veces la caída máxima, a costa de rentabilidad.
- Las comparaciones no corrigen por pruebas múltiples, y las historias reales siguen sin ser independientes entre sí.
