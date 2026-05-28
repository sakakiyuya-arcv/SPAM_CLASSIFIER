import torch
import os
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def _clean_text_light(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TransformerDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'transformer_model.pth')

        bert_model_dir = os.path.join(current_dir, 'my_bert_model')
        self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            bert_model_dir,
            num_labels=2
        )
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        # self.model = DistilBertForSequenceClassification.from_pretrained(
        #     'distilbert-base-multilingual-cased', 
        #     num_labels=2
        # )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, raw_emails):
        if isinstance(raw_emails, str):
            raw_emails = [raw_emails]
            
        cleaned_texts = [_clean_text_light(text) for text in raw_emails]

        inputs = self.tokenizer(
            cleaned_texts, return_tensors='pt', padding='max_length', 
            truncation=True, max_length=256 
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            spam_probs = probabilities[:, 1].cpu().tolist()

        if len(raw_emails) == 1:
            return spam_probs[0] * 100
        return [prob * 100 for prob in spam_probs]