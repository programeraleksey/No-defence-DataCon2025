import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rfpg

"""На данном этапе мы создаём 2D дескрипторы, а также отпечатки для дальнейшего обучения моделей, смотрим кореляцию и
убираем плохо корелируемые дескрипторы. Для работы требуется merged clear.csv"""

IN = "merged_clear.csv"
OUT = "DataSet_with_2D.csv"
R = 0.70

df = pd.read_csv(IN)

desc_names = [name for name, _ in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)


def calc_all_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        return list(calc.CalcDescriptors(mol))
    except Exception:
        return [np.nan] * len(desc_names)


def calc_fps(smiles):
    mol = Chem.MolFromSmiles(smiles)

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    ecfp_fp  = gen_ecfp.GetFingerprint(mol)

    return pd.Series([maccs_fp.ToBitString(), ecfp_fp.ToBitString()])


tqdm.pandas()
df_desc = df["Smiles"].progress_apply(calc_all_desc)
df_desc = pd.DataFrame(df_desc.tolist(), columns=desc_names)
df_final = pd.concat([df.reset_index(drop=True), df_desc], axis=1)


gen_ecfp = rfpg.GetMorganGenerator(radius=2, fpSize=2048)
df_final[["Maccs", "ECFP"]] = df_final["Smiles"].progress_apply(calc_fps)

df = df_final.dropna(axis=0, how="any")

cols_constant = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
df = df.drop(columns=cols_constant)

num_cols = df.select_dtypes(include=[np.number]).columns

corr = df[num_cols].corr().abs()                 # |r|
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > R)]
df_reduced = df.drop(columns=to_drop)
df_reduced.to_csv(OUT, index=False)