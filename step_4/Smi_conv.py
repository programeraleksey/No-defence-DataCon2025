from pathlib import Path
import pandas as pd

"""На данном этапе конвертируем csv формат до smi. А также оставляем только Value > 6.0. Это сделанно чтобы модель
дообучалась только на хороших данных. Для работы нужен DataSet_with_2D.csv"""

INPUT_FILE     = Path("DataSet_with_2D.csv")
OUTPUT_FILE    = Path("DataSet_for_training.smi")
SMILES_COLUMN  = "Smiles"
PIC50_COLUMN   = "Value"
PIC50_THRESHOLD = 6.0



def load_table(path: Path) -> pd.DataFrame:
    """Определяет формат по расширению и загружает таблицу."""
    ext = path.suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Неподдерживаемый формат '{ext}'. "
        "Ожидается CSV, Excel или Parquet."
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Файл '{INPUT_FILE}' не найден. "
            "Скорректируйте переменную INPUT_FILE в начале скрипта."
        )

    df = load_table(INPUT_FILE)

    # нормализуем регистр имён колонок
    cols_lower = {c.lower(): c for c in df.columns}
    try:
        smiles_col  = cols_lower[SMILES_COLUMN.lower()]
        pic50_col   = cols_lower[PIC50_COLUMN.lower()]
    except KeyError as e:
        raise KeyError(
            f"Колонка {e.args[0]} не найдена в таблице: "
            f"{list(df.columns)[:10]} …"
        )

    # отбор строк с Value > порога
    filtered_smiles = (
        df[df[pic50_col].astype(float) > PIC50_THRESHOLD][smiles_col]
        .astype(str)          # гарантируем строковый тип
        .str.strip()          # убираем лишние пробельные символы
    )

    # записываем .smi (каждый SMILES в отдельной строке)
    OUTPUT_FILE.write_text("\n".join(filtered_smiles) + "\n", encoding="utf-8")

    print(
        f"Сохранено {len(filtered_smiles):,} строк в '{OUTPUT_FILE}'. "
        "Файл готов для дообучения."
    )


if __name__ == "__main__":
    main()