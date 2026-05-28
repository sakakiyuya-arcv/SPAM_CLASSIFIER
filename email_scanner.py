import joblib
import pandas as pd
import math
from collections import Counter

class EmailAddressScanner:
    def __init__(self, model_path='rf_email_spam_model.pkl'):
        """Khởi tạo máy quét và nạp não bộ AI vào RAM"""
        try:
            self.model = joblib.load(model_path)
            # print(f"✅ Đã nạp thành công AI Quét Địa Chỉ từ: {model_path}")
        except FileNotFoundError:
            # print(f"❌ LỖI: Không tìm thấy file {model_path}! Hãy đảm bảo nó nằm cùng thư mục.")
            self.model = None

        # KHÓA CỨNG: Đảm bảo thứ tự cột nạp vào AI luôn chuẩn xác 100%
        self.feature_columns = [
            'local_length', 'domain_length', 'num_digits', 
            'digit_ratio', 'entropy', 'is_free_mail', 'has_suspicious_tld'
        ]

    def _calculate_entropy(self, text):
        """Tính độ hỗn loạn của chuỗi (Private Method)"""
        if not text: return 0
        counts = Counter(text)
        probs = [c / len(text) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def _extract_features(self, email):
        """Băm địa chỉ email thành các con số đặc trưng (Private Method)"""
        email = str(email).lower().strip()
        try:
            local_part, domain = email.split('@')
        except ValueError:
            return None 

        features = {
            'local_length': len(local_part),
            'domain_length': len(domain),
            'num_digits': sum(c.isdigit() for c in local_part),
            'digit_ratio': sum(c.isdigit() for c in local_part) / len(local_part) if local_part else 0,
            'entropy': self._calculate_entropy(local_part),
            'is_free_mail': 1 if domain in ['gmail.com', 'yahoo.com', 'hotmail.com'] else 0,
            'has_suspicious_tld': 1 if any(domain.endswith(tld) for tld in ['.tk', '.xyz', '.top', '.vip', '.biz']) else 0
        }
        return features

    def scan(self, email):
        """Hàm chính để hệ thống bên ngoài gọi vào"""
        if self.model is None:
            return {"error": "Hệ thống AI chưa sẵn sàng!"}

        feats = self._extract_features(email)
        if not feats:
            return {"error": f"Email sai định dạng: {email}"}
        
        # Tạo bảng và ép vào đúng khuôn (thứ tự cột) đã thiết lập
        df_input = pd.DataFrame([feats])[self.feature_columns]
        
        # Dự đoán
        spam_prob = self.model.predict_proba(df_input)[0][1] * 100
        
        # Trả về một Dictionary cực kỳ dễ xử lý cho file main
        return {
            "email": email,
            "spam_prob": round(spam_prob, 2),
            "is_spam": spam_prob > 60.0,
            "status": "🚨 SPAM" if spam_prob > 60.0 else "✅ AN TOÀN"
        }