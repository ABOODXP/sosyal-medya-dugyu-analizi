import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from fpdf import FPDF

# تحميل البيانات بنفس الطريقة
dataset = load_dataset("imdb")
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

model_name_or_path = "./results/checkpoint-750"  # هذا هو مجلد حفظ النموذج بعد التدريب

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = BertForSequenceClassification.from_pretrained(model_name_or_path)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
    do_train=False,
    do_eval=True,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.argmax(torch.tensor(pred.predictions), axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

result = trainer.evaluate()
print(result)

os.makedirs("static/reports", exist_ok=True)

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Model Evaluation Report", ln=True, align='C')
pdf.ln(10)
pdf.cell(200, 10, txt=f"Accuracy: {result['eval_accuracy']:.4f}", ln=True)
pdf.cell(200, 10, txt=f"Precision: {result['eval_precision']:.4f}", ln=True)
pdf.cell(200, 10, txt=f"Recall: {result['eval_recall']:.4f}", ln=True)
pdf.cell(200, 10, txt=f"F1 Score: {result['eval_f1']:.4f}", ln=True)

pdf_path = "static/reports/evaluation_report.pdf"
pdf.output(pdf_path)