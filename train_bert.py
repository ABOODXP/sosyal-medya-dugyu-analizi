import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# تحميل البيانات - مثال على بيانات IMDb، استبدله ببياناتك الخاصة إذا لزم الأمر
dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))  # تقليل الحجم للتجربة
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

# التحميل المسبق للنموذج والـ Tokenizer
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# دالة لتحويل النصوص إلى مدخلات نموذج BERT
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# حذف الأعمدة غير المطلوبة
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# إعدادات التدريب لعرض النتائج في الـ Terminal فقط
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=None,
    report_to="none",  # لا يستخدم TensorBoard
    logging_steps=10,
    logging_first_step=True,
    logging_strategy="steps",
    load_best_model_at_end=True
)

# دالة تقييم
def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.argmax(torch.tensor(pred.predictions), axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# تشغيل التدريب
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./results/checkpoint-750")
model = BertForSequenceClassification.from_pretrained(model_name_or_path)

result = trainer.evaluate()
print(result)