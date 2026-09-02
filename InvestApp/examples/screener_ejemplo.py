"""Ejemplo del Market Screener: barrer muchas acciones y devolver
las que están en momento de compra o de venta."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from investapp.screener import MarketScreener
from investapp.strategies import ConfluenceStrategy, HybridStrategy

# 1. Crear screener con la lista de tickers
screener = MarketScreener(tickers_file=str(Path(__file__).resolve().parents[1] / "tickers.csv"), verbose=True)

# 2. Barrer todas las acciones con la estrategia Hybrid
resultado = screener.scan(HybridStrategy(), period="2y")

# 3. Ver en terminal (tabla con señales)
resultado.mostrar_cli()

# 4. Filtrar activos en momento de COMPRA
compras = resultado.filtrar("COMPRA")
ventas = resultado.filtrar("VENTA")
print(f"\nCOMPRA: {[s.ticker for s in compras.signals]}")
print(f"VENTA: {[s.ticker for s in ventas.signals]}")

# 5. Cruzar dos estrategias: activos en COMPRA según hybrid Y confluence
cruce = resultado.intersectar(
    screener.scan(ConfluenceStrategy(), period="2y").filtrar("COMPRA")
)
print(f"\nEn COMPRA en ambas estrategias: {cruce.lista_tickers()}")

# 6. Reportes exportables
resultado.exportar_csv("reporte_senales.csv")
resultado.exportar_html("reporte_senales.html", title="Reporte de señales (Hybrid)")
print("\nReportes guardados: reporte_senales.csv, reporte_senales.html")