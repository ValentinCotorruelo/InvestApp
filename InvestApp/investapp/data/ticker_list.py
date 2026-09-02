"""Gestión de la lista manual de tickers."""
import csv
import json
from pathlib import Path
from typing import Iterable


class TickerList:
    """Administra una lista manual de símbolos de acciones.

    Soporta carga/guardado en CSV y JSON.
    """

    def __init__(self, tickers: Iterable[str] | None = None):
        self._tickers: list[str] = []
        if tickers:
            self.add_many(tickers)

    def add_many(self, tickers: Iterable[str]) -> None:
        for t in tickers:
            self.add(t)

    def add(self, ticker: str) -> bool:
        """Agrega un ticker normalizado. No duplica. Devuelve True si se agregó."""
        t = self._normalize(ticker)
        if not t:
            return False
        if t not in self._tickers:
            self._tickers.append(t)
            return True
        return False

    def remove(self, ticker: str) -> bool:
        t = self._normalize(ticker)
        if t in self._tickers:
            self._tickers.remove(t)
            return True
        return False

    def get(self) -> list[str]:
        return list(self._tickers)

    def __len__(self) -> int:
        return len(self._tickers)

    def __iter__(self):
        return iter(self._tickers)

    @staticmethod
    def _normalize(ticker: str) -> str:
        return ticker.strip().upper() if isinstance(ticker, str) else ""

    # ---- Persistencia ----

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker"])
            for t in self._tickers:
                writer.writerow([t])

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tickers": self._tickers}, f, indent=2)

    @classmethod
    def load_csv(cls, path: str | Path) -> "TickerList":
        path = Path(path)
        tl = cls()
        if not path.exists():
            return tl
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0 and row and row[0].strip().lower() == "ticker":
                    continue  # header
                if row and row[0].strip():
                    tl.add(row[0])
        return tl

    @classmethod
    def load_json(cls, path: str | Path) -> "TickerList":
        path = Path(path)
        tl = cls()
        if not path.exists():
            return tl
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tl.add_many(data.get("tickers", []))
        return tl

    @classmethod
    def load(cls, path: str | Path) -> "TickerList":
        """Carga CSV o JSON según extensión."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.load_json(path)
        return cls.load_csv(path)
