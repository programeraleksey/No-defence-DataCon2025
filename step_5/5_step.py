#!/usr/bin/env python
from pathlib import Path
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)

BASE      = "StanfordAIMI/DrugGPT"
DATA_SMI  = "DataSet_for_training.smi"
OUT_DIR   = "drugGPT_finetuned"
EPOCHS    = 2
BATCH     = 8
LR        = 5e-5
MAX_LEN   = 120
SEED      = 42

def main():
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="right")
    tok.pad_token = tok.eos_token
    smiles = [s.strip() for s in Path(DATA_SMI).read_text().splitlines() if s.strip()]
    ds = Dataset.from_dict({"text": smiles})
    ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=MAX_LEN),
                batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(BASE)
    args = TrainingArguments(
        output_dir=OUT_DIR, overwrite_output_dir=True,
        num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
        learning_rate=LR, fp16=True, seed=SEED,
        save_strategy="epoch", save_total_limit=1, report_to="none")
    Trainer(model, args, ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)
    print("✓ fine‑tune завершён, чек‑пойнт сохранён:", OUT_DIR)

if __name__ == "__main__":
    main()
