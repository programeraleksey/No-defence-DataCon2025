#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resalt_halide_top10.py
──────────────────────
• вход : top10.csv   (должна лежать в той же папке)
• выход: top10_resalt.csv (добавлена колонка SMILES_resalt)
Пересоливает Br⁻ ↔ Cl⁻, free‑base не трогает.
"""

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops

IN_CSV   = Path("top10.csv")
OUT_CSV  = Path("top10_resalt.csv")
SMI_COL  = "SMILES"          # имя колонки со SMILES
HALIDES  = {"Cl": "[Cl-]", "Br": "[Br-]"}

def resalt_smiles(smiles: str) -> str:
    """Br⁻ ↔ Cl⁻, иначе возвращает исходный SMILES."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return smiles

    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    ions  = [f for f in frags if f.GetNumAtoms() == 1 and
             f.GetAtomWithIdx(0).GetFormalCharge() == -1 and
             f.GetAtomWithIdx(0).GetSymbol() in ("Cl", "Br")]
    cations = [f for f in frags if f not in ions]

    if not ions:                           # не соль
        return Chem.MolToSmiles(mol, canonical=True)

    hal = ions[0].GetAtomWithIdx(0).GetSymbol()  # 'Cl' or 'Br'
    target = "Cl" if hal == "Br" else "Br"       # меняем на обратный

    cat = cations[0]
    new_halide = Chem.MolFromSmiles(HALIDES[target])
    combo = Chem.CombineMols(cat, new_halide)
    rdmolops.SanitizeMol(combo)
    return Chem.MolToSmiles(combo, canonical=True)

def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Файл {IN_CSV} не найден в текущей папке.")
    df = pd.read_csv(IN_CSV)
    if SMI_COL not in df.columns:
        raise SystemExit(f"Колонка '{SMI_COL}' отсутствует в {IN_CSV}.")
    df["SMILES_resalt"] = df[SMI_COL].astype(str).map(resalt_smiles)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ пересолено {len(df)} строк  →  {OUT_CSV}")
    diff = df[df["SMILES"] != df["SMILES_resalt"]]
    print("\nИзменённые строки:")
    print(diff[[SMI_COL, "SMILES_resalt"]])

if __name__ == "__main__":
    main()
