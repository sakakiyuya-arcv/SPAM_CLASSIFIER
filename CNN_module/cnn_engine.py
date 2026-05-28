import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from transformers import DistilBertTokenizer, DistilBertModel

def _clean_text_light(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # text = re.sub(r'http\S+', 'httpaddr', text)
    return text

class CNN_Spam_Model(nn.Module):
    def __init__(self, n_filters, output_dim, dropout, pad_idx):
        super(CNN_Spam_Model, self).__init__()
        
        bert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        self.embedding = nn.Embedding.from_pretrained(
            bert.embeddings.word_embeddings.weight, 
            freeze=False, 
            padding_idx=pad_idx
        )
        
        embedding_dim = 768 
        self.conv3 = nn.Conv1d(embedding_dim, n_filters, 3)
        self.conv5 = nn.Conv1d(embedding_dim, n_filters, 5)
        self.conv7 = nn.Conv1d(embedding_dim, n_filters, 7)
        self.conv9 = nn.Conv1d(embedding_dim, n_filters, 9) 
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * 4, output_dim) 
        
    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1) 
        
        convolved3 = F.relu(self.conv3(embedded)) 
        convolved5 = F.relu(self.conv5(embedded)) 
        convolved7 = F.relu(self.conv7(embedded)) 
        convolved9 = F.relu(self.conv9(embedded))
         
        pooled3 = F.max_pool1d(convolved3, convolved3.shape[2]).squeeze(2) 
        pooled5 = F.max_pool1d(convolved5, convolved5.shape[2]).squeeze(2) 
        pooled7 = F.max_pool1d(convolved7, convolved7.shape[2]).squeeze(2) 
        pooled9 = F.max_pool1d(convolved9, convolved9.shape[2]).squeeze(2)
        
        cat = torch.cat((pooled3, pooled5, pooled7, pooled9), dim=1) 
        return self.fc(self.dropout(cat))

class CNNDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        model_path = os.path.join(current_dir, 'cnn_model.pth')
        
        self.model = CNN_Spam_Model(
            n_filters=128, 
            output_dim=1, 
            dropout=0.4,
            pad_idx=self.tokenizer.pad_token_id
        )
        
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
            output = self.model(inputs['input_ids'])
            spam_probs = torch.sigmoid(output).squeeze(1).cpu().tolist()

        if not isinstance(spam_probs, list):
            spam_probs = [spam_probs]

        if len(raw_emails) == 1:
             return spam_probs[0] * 100
        return [prob * 100 for prob in spam_probs]