import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import spikegen
import re
import pickle 

def _clean_text_light(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SpamFilterSNN(nn.Module):
    def __init__(self, vocab_size=5000, hidden_size=128, num_classes=2, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []
        num_steps = x.size(0) 

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

class SNNDetector:
    def __init__(self, model_path='snn_model.pth', vectorizer_path='tfidf_vectorizer.pkl', num_steps=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        model_path = os.path.join(current_dir, 'snn_model.pth')
        vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
        
        self.model = SpamFilterSNN(vocab_size=5000, num_classes=2).to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        self.num_steps = num_steps

    def predict(self, raw_emails, seed=42): 
        if not raw_emails:
            return [0.0] if isinstance(raw_emails, list) else 0.0

        if isinstance(raw_emails, str):
            raw_emails = [raw_emails]
        
        cleaned_texts = [_clean_text_light(email) for email in raw_emails]
        
        inputs = self.vectorizer.transform(cleaned_texts).toarray()
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
                
            spike_data = spikegen.rate(inputs, num_steps=self.num_steps)
            
            spk_out, _ = self.model(spike_data)
            
            total_spikes = spk_out.sum(dim=0) 
            
            probs = F.softmax(total_spikes, dim=1)
            
            spam_probs = probs[:, 1].cpu().tolist()

        if len(raw_emails) == 1:
             return spam_probs[0] * 100
        return [prob * 100 for prob in spam_probs]