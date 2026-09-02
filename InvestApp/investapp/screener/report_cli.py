"""Reporte en terminal (CLI) de los resultados del screener."""
from __future__ import annotations

from typing import Optional

import sys

# Mejorar renderizado de acentos en consolas Windows modernas
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WIDTH = 100


def _colorize(text: str, senal: str) -> str:
    """Colorea texto según ANSI (verde compra, rojo venta, gris neutro)."""
    try:
        if senal == "COMPRA":
            return f"\033[92m{text}\033[0m"
        if senal == "VENTA":
            return f"\033[91m{text}\033[0m"
        return f"\033[90m{text}\033[0m"
    except Exception:
        return text


def print_table(result, top: Optional[int] = None) -> None:
    """Imprime la tabla de señales."""
    rows = result.signals
    if top is not None:
        rows = rows[:top]

    strategy = result.strategy_name
    header = f" Market Screener | Estrategia: {strategy} "
    line = "-" * WIDTH
    print(line)
    print(_colorize(_pad(header, WIDTH), "NEUTRO"))
    print(line)

    cols = [
        ("Ticker", 9),
        ("Señal", 8),
        ("Precio", 10),
        ("Entrada", 10),
        ("SL", 10),
        ("TP", 10),
        ("R/R", 6),
        ("Fuerza", 8),
        ("Régimen", 12),
    ]
    header_line = "".join(_pad(c, w) for c, w in cols)
    print(_colorize(header_line, "NEUTRO"))
    print(line)

    if not rows:
        print("  Sin señales en este barrido.")
        print(line)
        return

    sorted_rows = sorted(rows, key=lambda r: (r.senal != "COMPRA", r.senal != "COMPRA" and r.senal != "VENTA", -r.strength))
    for r in sorted_rows:
        tp = f"{r.take_profit:.2f}" if r.take_profit else "—"
        row_line = "".join(
            [
                _pad(r.ticker, 9),
                _pad(r.senal, 8),
                _pad(f"{r.precio:.2f}", 10),
                _pad(f"{r.entrada:.2f}" if r.entrada else "—", 10),
                _pad(f"{r.stop_loss:.2f}" if r.stop_loss else "—", 10),
                _pad(tp, 10),
                _pad(f"{r.rr:.2f}", 6),
                _pad(f"{r.strength:.2f}", 8),
                _pad(r.regime, 12),
            ]
        )
        print(_colorize(row_line, r.senal))

    print(line)
    compras = sum(1 for r in rows if r.senal == "COMPRA")
    ventas = sum(1 for r in rows if r.senal == "VENTA")
    print(f"  Total: {len(rows)} señales | COMPRA: {compras} | VENTA: {ventas}")
    print(line)

    # Detalle de motivos para cada señal
    for r in rows:
        if r.reasons:
            print(f"\n  {_colorize(r.ticker, r.senal)} ({r.senal}):")
            for reason in r.reasons:
                print(f"    · {reason}")


def _pad(text: str, width: int) -> str:
    t = str(text)
    return t.ljust(width)[:width]