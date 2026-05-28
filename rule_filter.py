import json
import os

class EmailBlacklistFilter:
    def __init__(self, db_path='blacklist.json'):
        self.db_path = db_path
        self.banned_emails = set()
        self.banned_domains = set()
        
        self.load_database()

    def load_database(self):
        """Đọc file JSON và chuyển thành Set (Tập hợp) để truy xuất siêu nhanh O(1)"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.banned_emails = set(data.get("banned_emails", []))
                self.banned_domains = set(data.get("banned_domains", []))
        else:
            self.banned_emails = {"lode88@tk.com", "chovay_nhanh@gmail.com", "quangcao_rac@yahoo.com"}
            self.banned_domains = {"@scam.vn", "@offer.tk", "@ru.mail"}
            self.save_database()

    def save_database(self):
        """Ghi đè dữ liệu từ RAM xuống file JSON"""
        data = {
            "banned_emails": list(self.banned_emails),
            "banned_domains": list(self.banned_domains)
        }
        with open(self.db_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    def add_to_blacklist(self, item):
        """
        Hàm dùng để AI gọi khi bắt được thư rác mới.
        Truyền vào "@domain.com" để cấm tên miền, hoặc "email@domain.com" để cấm đích danh.
        """
        item_clean = str(item).lower().strip()
        
        if item_clean.startswith("@"):
            self.banned_domains.add(item_clean)
        else:
            self.banned_emails.add(item_clean)
            
        self.save_database()

    def check_spam(self, sender_email):
        """Kiểm tra xem email có nằm trong sổ đen không"""
        email_clean = str(sender_email).lower().strip()

        if email_clean in self.banned_emails:
            return True, f"[BLOCK] Email '{email_clean}' nằm trong danh sách đen."

        if "@" in email_clean:
            domain = "@" + email_clean.split("@")[1] 
            if domain in self.banned_domains:
                return True, f"[BLOCK] Tên miền '{domain}' bị cấm toàn hệ thống."

        return False, "[PASS] Địa chỉ email an toàn."