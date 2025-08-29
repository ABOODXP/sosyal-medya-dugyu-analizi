import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd

# 1. تحميل البيانات (مثال: ملف CSV)
df = pd.read_csv("sentiment_data_1.csv")  # ✅ عدل المسار إلى بياناتك
df = df[["text", "label"]]  # تأكد من وجود عمود "text" و "label"

# 2. تقسيم البيانات
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# 3. تحميل Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# 4. تحويل النصوص إلى تنسيقات BERT
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx]),
        }

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

# 5. تحميل النموذج
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)

# 6. إعدادات التدريب مع TensorBoard
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="steps",
    logging_dir=None,                  # ✅ يسجل في TensorBoard
    logging_steps=10,                      # ✅ كل 10 خطوات يسجل
    report_to="none",               # ✅ مهم جداً
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True
)

# 7. المدرب
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 8. بدء التدريب
trainer.train()

# 9. حفظ النموذج المدرب
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")
