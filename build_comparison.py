import json
from pathlib import Path
import textwrap

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
path = ROOT / 'notebooks/03_extras.ipynb'
notebook = nbformat.read(path, as_version=4)

intro = '''### Base frente a estado enriquecido: qué cambia en los resultados

**Base:** señal de tendencia corta y posición (20 estados). **Enriquecido:** añade distancia a la media de 30 precios y volatilidad de 20 retornos (hasta 180 estados). Se añaden ambos indicadores a la vez; esta comparación no identifica su efecto por separado.

Se enfrentan las variantes **del experimento 03**, con los mismos hiperparámetros, 800 episodios por entrenamiento, cinco semillas y las mismas historias de test en la ejecución completa. Los modelos ajustados de 01–02 tienen otras configuraciones y no son el control de esta prueba. Añadir mercados sintéticos y datos reales amplía la evaluación; no es una característica adicional que reciba el agente.

Las tablas colocan las dos variantes una al lado de la otra. **Δ retorno = enriquecido − base**; positivo significa más rentabilidad. **Reducción del drawdown = base − enriquecido**; positivo significa una caída máxima menor. Ambas diferencias se expresan en **puntos porcentuales (pp)**: pasar del 10 % al 15 % son +5 pp.

El retorno resume primero cada entrenamiento y después toma la mediana entre semillas. En sintéticos, el drawdown es la mediana entre semillas del drawdown máximo medio de las 200 historias; en el S&P 500 es la mediana entre semillas sobre el único recorrido real de 2024–2025. Las diferencias de estas tablas son **restas de los resúmenes**, distintas de la mediana de diferencias emparejadas de la sección siguiente. En modo rápido se usa el presupuesto reducido indicado al inicio.
'''

table_code = '''
rows = []
for market in MARKETS_03:
    for algorithm in ALGORITHMS:
        base_row = final.loc[(final.mercado == market) & (final.agente == f"{algorithm}_base")].squeeze()
        rich_row = final.loc[(final.mercado == market) & (final.agente == f"{algorithm}_enriquecido")].squeeze()
        rows.append(dict(
            mercado=market, algoritmo=algorithm,
            retorno_base=base_row.retorno_mediana,
            retorno_enriquecido=rich_row.retorno_mediana,
            delta_retorno_pp=100 * (rich_row.retorno_mediana - base_row.retorno_mediana),
            drawdown_base=base_row.max_drawdown,
            drawdown_enriquecido=rich_row.max_drawdown,
            reduccion_drawdown_pp=100 * (base_row.max_drawdown - rich_row.max_drawdown),
        ))
comparison = pd.DataFrame(rows)
comparison.to_csv(RESULTS_DIR / f"03_base_vs_enriquecido{SUF}.csv", index=False)

readable = comparison.assign(
    mercado=comparison.mercado.map(LABELS),
    algoritmo=comparison.algoritmo.map({"qlearning": "Q-Learning", "sarsa": "SARSA"}),
).set_index(["mercado", "algoritmo"])
readable.index.names = ["Mercado", "Algoritmo"]
for columns, caption in (
    (["retorno_base", "retorno_enriquecido", "delta_retorno_pp"], "Rentabilidad: positivo en la diferencia significa más retorno"),
    (["drawdown_base", "drawdown_enriquecido", "reduccion_drawdown_pp"], "Caída máxima: positivo en la reducción significa menos riesgo observado"),
):
    labels = {
        "retorno_base": "Retorno base", "retorno_enriquecido": "Retorno enriquecido",
        "delta_retorno_pp": "Δ retorno (pp)", "drawdown_base": "Drawdown base",
        "drawdown_enriquecido": "Drawdown enriquecido",
        "reduccion_drawdown_pp": "Reducción drawdown (pp)",
    }
    view = readable[columns].rename(columns=labels)
    formats = {labels[c]: ("{:+.1f}" if c.endswith("_pp") else "{:.1%}") for c in columns}
    display(view.style.format(formats).map(
        lambda value: "color: #187344" if value > 1e-9 else ("color: #B23A38" if value < -1e-9 else ""),
        subset=[labels[columns[-1]]],
    ).set_caption(caption))
'''

plot_code = '''
fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
y = np.arange(len(comparison))
row_labels = [f"{LABELS[r.mercado]} · {'Q' if r.algoritmo == 'qlearning' else 'SARSA'}"
              for r in comparison.itertuples()]
for ax, column, title in (
    (axes[0], "delta_retorno_pp", "Cambio de rentabilidad\\nEnriquecido − base"),
    (axes[1], "reduccion_drawdown_pp", "Reducción de la caída máxima\\nBase − enriquecido"),
):
    values = comparison[column].to_numpy()
    colors = ["#187344" if value > 1e-9 else ("#B23A38" if value < -1e-9 else "#6B7280") for value in values]
    ax.barh(y, values, color=colors, height=.65)
    ax.scatter(np.zeros(len(y)), y, color="#6B7280", s=12, zorder=3)
    ax.axvline(0, color="#374151", linewidth=.8)
    for yi, value in zip(y, values):
        ax.annotate(f"{value:+.1f}", (value, yi), xytext=(5 if value >= 0 else -5, 0),
                    textcoords="offset points", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    span = max(float(np.ptp(np.r_[values, 0])), 1.)
    ax.set_xlim(min(0, values.min()) - .2 * span, max(0, values.max()) + .2 * span)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Puntos porcentuales · positivo = mejora")
    ax.grid(axis="y", visible=False)
    ax.axhspan(7.5, 9.5, color="#E8EEF5", alpha=.5, zorder=0)
axes[0].set_yticks(y, row_labels)
axes[0].invert_yaxis()
fig.suptitle("Qué aporta el estado enriquecido frente al base", fontweight="bold")
fig.text(.5, .015, "Verde: mejora · rojo: empeora · gris: sin cambio. S&P 500: un único recorrido real de test.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .04, 1, .95))
figure(fig, "03_base_vs_enriquecido")
plt.show()
'''

report_code = '''
report = ["# Comparación directa: modelos base y estado enriquecido", "",
          f"Resultados del experimento 03: {len(SEEDS)} semillas, {N_EP} episodios por entrenamiento y "
          f"{N_TEST} historias sintéticas de test; un único recorrido real de 2024–2025.", "",
          "Base: tendencia corta y posición (20 estados). Enriquecido: añade distancia MA30 y volatilidad20 "
          "(hasta 180 estados). Mismos hiperparámetros, presupuesto e historias de test para ambas variantes. "
          "Los modelos ajustados de 01–02 no son el control de esta comparación.", "",
          "## Rentabilidad", "",
          "| Mercado | Algoritmo | Base | Enriquecido | Δ retorno (pp) |",
          "|---|---|---:|---:|---:|"]
for r in comparison.itertuples():
    algorithm_name = "Q-Learning" if r.algoritmo == "qlearning" else "SARSA"
    report.append(f"| {LABELS[r.mercado]} | {algorithm_name} | {r.retorno_base:+.1%} | "
                  f"{r.retorno_enriquecido:+.1%} | {r.delta_retorno_pp:+.1f} |")
report += ["", "## Caída máxima (drawdown)", "",
           "| Mercado | Algoritmo | Base | Enriquecido | Reducción drawdown (pp) |",
           "|---|---|---:|---:|---:|"]
for r in comparison.itertuples():
    algorithm_name = "Q-Learning" if r.algoritmo == "qlearning" else "SARSA"
    report.append(f"| {LABELS[r.mercado]} | {algorithm_name} | {r.drawdown_base:.1%} | "
                  f"{r.drawdown_enriquecido:.1%} | {r.reduccion_drawdown_pp:+.1f} |")
report += ["", "## Lectura de los resultados", ""]
for r in comparison.itertuples():
    algorithm_name = "Q-Learning" if r.algoritmo == "qlearning" else "SARSA"
    returns = "más retorno" if r.delta_retorno_pp > 1e-9 else ("menos retorno" if r.delta_retorno_pp < -1e-9 else "el mismo retorno")
    risk = "menor drawdown" if r.reduccion_drawdown_pp > 1e-9 else ("mayor drawdown" if r.reduccion_drawdown_pp < -1e-9 else "el mismo drawdown")
    report.append(f"- **{LABELS[r.mercado]} · {algorithm_name}:** {returns} y {risk}.")
report += ["", "Las diferencias son restas de los resúmenes, expresadas en puntos porcentuales: "
           "Δ retorno = enriquecido − base; reducción del drawdown = base − enriquecido. "
           "En ambas columnas, positivo indica mejora. No son las medianas de diferencias emparejadas "
           "ni los intervalos bootstrap de `03_estado_deltas.csv`.", "",
           "El retorno toma primero la mediana entre historias de cada entrenamiento y después la mediana "
           "entre semillas. El drawdown toma la media entre historias de cada entrenamiento y después la "
           "mediana entre semillas. En la serie real solo hay una historia por política: cinco semillas no "
           "equivalen a cinco mercados independientes. Más indicadores no garantizan más rentabilidad; "
           "ambos se incorporan conjuntamente y no se mide su contribución individual.", "",
           f"![Diferencias de retorno y drawdown](figures/03_base_vs_enriquecido{SUF}.png)", ""]
comparison_report = "\\n".join(report)
(RESULTS_DIR / f"03_base_vs_enriquecido{SUF}.md").write_text(comparison_report, encoding="utf-8")
display(Markdown("\\n".join(report[report.index("## Lectura de los resultados"):report.index("## Lectura de los resultados") + 12])))
'''

new_cells = [nbformat.v4.new_markdown_cell(intro)] + [
    nbformat.v4.new_code_cell(textwrap.dedent(code).strip())
    for code in (table_code, plot_code, report_code)
]

# Execute only the added section against the existing full results; no retraining.
bootstrap = '''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from src.experiment import RESULTS_DIR
from src.style import setup, savefig
from src.extras import SYNTHETIC_MARKETS, LABELS, ALGORITHMS
setup()
SEEDS = list(range(5))
N_EP, N_TEST, SUF = 800, 200, ""
MARKETS_03 = [*SYNTHETIC_MARKETS, "sp500_real"]
final = pd.read_csv(RESULTS_DIR / "03_extras.csv")
def figure(fig, name):
    return savefig(fig, name + SUF)
'''
execution = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(bootstrap), *new_cells])
NotebookClient(execution, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
position = next(i for i, c in enumerate(notebook.cells) if c.source.startswith('### ¿Cuánto aporta el estado?'))
notebook.cells[position:position] = execution.cells[1:]
nbformat.write(notebook, path)

# Keep a short, immediately visible comparison in the repository entry point.
report_path = ROOT / 'results/03_base_vs_enriquecido.md'
report_text = report_path.read_text(encoding='utf-8')
return_table = report_text.split('## Rentabilidad\n\n', 1)[1].split('\n\n## Caída', 1)[0]
section = '''## Comparación directa: base frente a estado enriquecido

El estado enriquecido añade **distancia a MA30 y volatilidad20** al estado base de tendencia corta y posición.
La comparación usa las dos variantes del experimento 03: mismos hiperparámetros, 800 episodios por entrenamiento,
cinco semillas y las mismas historias de test. No se compara contra las configuraciones ajustadas de 01–02.
Los valores son los resúmenes de retorno de `results/03_extras.csv`; **Δ = enriquecido − base**, en puntos porcentuales.

''' + return_table + '''

Mejora el retorno en reversión, regímenes y tendencia persistente; en difusión con saltos no hay cambio.
En el S&P 500, el enriquecido reduce el retorno y también la caída máxima: Q-Learning pasa de 18,9 % a 11,4 %
de drawdown y SARSA de 18,9 % a 12,8 %. Añadir información no mejora todas las medidas en todos los mercados.
La serie real es un único recorrido de test de 2024–2025; las semillas no son historias independientes.

La [comparación completa](results/03_base_vs_enriquecido.md) incluye las tablas de rentabilidad y drawdown y un gráfico
de las diferencias. También se muestra en el apartado **«Base frente a estado enriquecido»** de
`notebooks/03_extras.ipynb`, que regenera el [CSV de diferencias](results/03_base_vs_enriquecido.csv), el informe y la figura.
Estas diferencias entre resúmenes no son la mediana de diferencias emparejadas de `03_estado_deltas.csv`.

'''
readme_path = ROOT / 'README.md'
readme = readme_path.read_text(encoding='utf-8')
readme = readme.replace('## Estado\n', section + '## Estado\n', 1)
readme_path.write_text(readme, encoding='utf-8')
print('Added executed comparison section, report, CSV, figure and README summary.')
