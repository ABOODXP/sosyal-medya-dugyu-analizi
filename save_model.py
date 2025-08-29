from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model.save_pretrained("fine_tuned_bert")
tokenizer.save_pretrained("fine_tuned_bert")