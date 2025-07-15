from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

INPUT_PATH = Path("merged.csv")
OUTPUT_PATH = Path("merged_clear.csv")

"""На данном этапе мы удаляем дубликаты првоеряем SMILES на валидность, в результате получаем очищенный датасет.
Для работы требуется merged.csv"""

def valid_smiles(smiles: str | float | None) -> bool:
    if not isinstance(smiles, str) or not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False

def clean_ic50(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", quotechar='"', low_memory=False)

    # Удаляем дубликаты по SMILES
    df = df.drop_duplicates(subset="Smiles", keep="first")

    # Валидируем SMILES
    df = df[df["Smiles"].map(valid_smiles)]

    return df[["Smiles", "Value"]]


if __name__ == "__main__":
    cleaned = clean_ic50(INPUT_PATH)
    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранён файл: {OUTPUT_PATH} ({len(cleaned)} строк)")
