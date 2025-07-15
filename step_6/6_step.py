#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_and_screen.py
──────────────────────
1. Берёт fine‑tuned чек‑пойнт DrugGPT (drugGPT_finetuned/)
2. Генерирует N_SMILES молекул nucleus‑sampling'ом
3. Фильтрует: pIC50 > 6, QED > 0.7, 2 < SA < 6, нет токсофоров, ≤1 Lipinski‑viol
4. Сохраняет финальный CSV
"""
from pathlib import Path
import random, math, json, gzip, pickle, base64, warnings
from typing import List

# ────────── Библиотеки ───────────────────────────────────────────────────
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Lipinski, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from transformers import AutoTokenizer, AutoModelForCausalLM
import joblib, lightgbm as lgb
from tqdm.auto import tqdm

# ────────── Параметры ────────────────────────────────────────────────────
MODEL_DIR   = Path("drugGPT_finetuned")          # чек‑пойнт после fine‑tune
LGBM_PATH   = Path("lgbm.joblib")          # 127‑дескрипторов
OUT_CSV     = Path("generated_filtered.csv")
OUT_SMI = Path("generated_valid.smi")

N_SMILES    = 15_000
BATCH_GEN   = 8
TOP_P       = 0.95
TEMPERATURE = 1.0
MAX_LEN     = 120
SEED        = 42

# ────────── Случайность (фиксация) ───────────────────────────────────────
random.seed(SEED); np.random.seed(SEED)

# ────────── 1.  Загрузка RDKit‑списка дескрипторов ──────────────────────
RD_NAMES: List[str] = [
   "MaxAbsEStateIndex","MinAbsEStateIndex","MinEStateIndex","qed","SPS",
   "MaxPartialCharge","MinPartialCharge","FpDensityMorgan1","BCUT2D_MWHI",
   "BCUT2D_LOGPLOW","BCUT2D_MRHI","BCUT2D_MRLOW","AvgIpc","BalabanJ","Ipc",
   "PEOE_VSA1","PEOE_VSA10","PEOE_VSA11","PEOE_VSA12","PEOE_VSA13",
   "PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA5","PEOE_VSA6",
   "PEOE_VSA8","PEOE_VSA9","SMR_VSA10","SMR_VSA2","SMR_VSA3","SMR_VSA4",
   "SMR_VSA6","SMR_VSA8","SlogP_VSA1","SlogP_VSA10","SlogP_VSA11",
   "SlogP_VSA12","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5",
   "SlogP_VSA6","SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","VSA_EState1",
   "VSA_EState10","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5",
   "VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9","fr_Al_COO",
   "fr_Al_OH","fr_Al_OH_noTert","fr_ArN","fr_Ar_COO","fr_Ar_N","fr_Ar_NH",
   "fr_Ar_OH","fr_Ar_ring","fr_ArNHR","fr_Ar_OHNoCOO","fr_COO","fr_COO2",
   "fr_C_O","fr_C_O_noCOO","fr_C_S","fr_F","fr_Imine","fr_NH0","fr_NH1",
   "fr_NH2","fr_N_O","fr_Ndealkylation1","fr_Ndealkylation2","fr_Nhpyrrole",
   "fr_Npyridinium","fr_OCN","fr_OH","fr_OH_noO","fr_SH","fr_SMe",
   "fr_aldehyde","fr_alkyl_carbamate","fr_alkyl_halide","fr_allylic_oxid",
   "fr_amide","fr_amidine","fr_aniline","fr_aryl_methyl","fr_azide",
   "fr_azo","fr_barbitur","fr_benzene","fr_benzodiazepine","fr_bicyclic",
   "fr_diazo","fr_dihydropyridine","fr_epoxide","fr_ester","fr_ether",
   "fr_furan","fr_guanido","fr_halogen","fr_hdrzine","fr_hdrzone","fr_imide",
   "fr_imidazole","fr_isocyan","fr_isothiocyan","fr_ketone",
   "fr_ketone_Topliss","fr_lactam","fr_lactone","fr_methoxy","fr_morpholine",
   "fr_nitrile","fr_nitro","fr_nitro_arom","fr_nitroso","fr_oxazole",
   "fr_oxime","fr_para_hydroxylation"
]
assert len(RD_NAMES) == 127, "descriptor_order.json должен содержать ровно 127 имён!"

# ────────── 2.  Загрузка токенизатора/модели (левый padding) ────────────
tok = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side='left')
tok.pad_token = tok.eos_token
gpt = AutoModelForCausalLM.from_pretrained(MODEL_DIR).eval()

# ────────── 3.  Генерация ------------------------------------------------
def generate_smiles(n=N_SMILES) -> List[str]:
    out=[]
    batches=(n+BATCH_GEN-1)//BATCH_GEN
    for _ in tqdm(range(batches), desc="Generate"):
        ids = tok([tok.eos_token]*BATCH_GEN, return_tensors="pt",
                  add_special_tokens=False).input_ids
        gen = gpt.generate(ids, max_length=MAX_LEN, do_sample=True,
                           top_p=TOP_P, temperature=TEMPERATURE,
                           pad_token_id=tok.eos_token_id)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        if len(out)>=n: break
    return [s for s in out if Chem.MolFromSmiles(s)]

def save_smiles(smiles: list[str], path: Path):
    """Пишет список SMILES в .smi (одна строка ‑ одна молекула)."""
    path.write_text("\n".join(smiles) + "\n", encoding="utf‑8")
    print(f"✓ SMILES сохранены → {path}  ({len(smiles)} строк)")

# ────────── 4.  RDKit‑127 дескрипторов ----------------------------------
calc127 = MoleculeDescriptors.MolecularDescriptorCalculator(RD_NAMES)
def df_descriptors(smiles: List[str]) -> pd.DataFrame:
    rows, keep=[],[]
    for s in tqdm(smiles, desc="Descriptors"):
        mol=Chem.MolFromSmiles(s); rows.append(calc127.CalcDescriptors(mol)); keep.append(s)
    df=pd.DataFrame(rows, columns=[f"Column_{i}" for i in range(127)])
    df.insert(0,"SMILES",keep)
    return df

# ────────── 5.  Предсказание pIC50 --------------------------------------
booster: lgb.Booster = joblib.load(LGBM_PATH)

def add_pred(df: pd.DataFrame):
    X=df.drop(columns="SMILES").values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # feature names
        df["Pred_pIC50"]=booster.predict(X)
    return df

# ────────── 6.  SA‑Score (чистая Python копия) ---------------------------
from sascorer import calculateScore   # или scoreMol в старых версиях RDKit
def sa_score(mol: Chem.Mol) -> float:
    return calculateScore(mol)

# ────────── 7.  SMARTS токсифоры (80 правил BRENK‑core) ------------------
from tox_smarts import TOX_PATTERNS   # файл со словарём: name→Mol
def has_tox(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True   # невалидный SMILES → считаем токсичным
    return any(mol.HasSubstructMatch(patt) for patt in TOX_PATTERNS.values())

# ────────── 8.  Lipinski violations -------------------------------------
def lipinski_violations(mol: Chem.Mol) -> int:
    return sum([
        Lipinski.HeavyAtomCount(mol)>50,
        Descriptors.MolWt(mol)>500,
        Lipinski.NumHDonors(mol)>5,
        Lipinski.NumHAcceptors(mol)>10,
        Descriptors.MolLogP(mol)>5])

# ────────── 9.  Обогащение и фильтр -------------------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет QED, SA, токсофоры, Lipinski‑нарушения
    и поэтапно фильтрует/печатает статистику.
    """
    # ── 1.  расчёт дескрипторов drug‑likeness ────────────────────────────
    qeds, sas, toxs, lips = [], [], [], []
    for smi in tqdm(df.SMILES, desc="QED/SA/Lipinski/Tox"):
        mol = Chem.MolFromSmiles(smi)
        qeds.append(QED.qed(mol))
        sas.append(sa_score(mol))
        toxs.append(has_tox(smi))
        lips.append(lipinski_violations(mol))

    df["QED"]      = qeds
    df["SA"]       = sas
    df["HasTox"]   = toxs
    df["LipViol"]  = lips

    # ── 2.  пошаговая фильтрация с отчётом ──────────────────────────────
    def report(stage: str, mask: pd.Series) -> pd.Series:
        """Печатает, сколько осталось, и возвращает маску."""
        print(f"  ↳ после {stage:<15}: {mask.sum():>4} / {len(df)}")
        return mask

    m = pd.Series(True, index=df.index)

    # каждый критерий — отдельная маска
    m &= report("pIC50>3",          df["Pred_pIC50"] > 3)
    m &= report("QED>0.5",          df["QED"] > 0.5)
    m &= report("2<SA<6",         df["SA"].between(2, 6))
    m &= report("Нет токсофоров",     ~df["HasTox"])
    m &= report("Lipinski≤1",       df["LipViol"] <= 1)

    # ── 3.  возврат отфильтрованной таблицы ─────────────────────────────
    df_final = df[m].reset_index(drop=True)
    print(f"✓ Итог после всех фильтров: {len(df_final)} из {len(df)}\n")
    return df_final

# ────────── 10.  pipeline ------------------------------------------------
def main():
    smiles=generate_smiles()
    save_smiles(smiles, OUT_SMI)
    print(f"✓ Валидных: {len(smiles)} / {N_SMILES}")
    df=df_descriptors(smiles)
    print(f"✓ Дескрипторы рассчитаны для {len(df)} молекул")
    df=add_pred(df)
    print(f"✓ Добавлены Pred_pIC50({len(df)} строк)")
    df=enrich(df)
    print(f"✓ После фильтров drug‑likeness осталось {len(df)} молекул")
    df.to_csv(OUT_CSV,index=False)
    print(f"✓ Отобрано {len(df)} молекул  →  {OUT_CSV}")

if __name__=="__main__":
    main()
