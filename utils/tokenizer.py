import sqlite3
import re
from utils.constants import DB, ethan_uname
import bidict

class Tokenizer:
    def __init__(self):
        self.bidict_word_id = bidict()
        self.bidict_word_id["<pad>"] = 0
        self.bidict_word_id["<sos>"] = 1
        self.bidict_word_id["<eos>"] = 2
        self.bidict_word_id["<unk>"] = 3
    
    def add_token(self, word):
        if word not in self.bidict_word_id:
            wid = len(self.bidict_word_id)
            self.bidict_word_id[word] = wid
    
    def encode(self, message: str):
        return [self.bidict_word_id[word] for word in message]
    
    def decode(self, tokens: str):
        return [self.bidict_word_id.inv[wid] for wid in tokens]
    
    def add_tokens_from_message(self, msg: str, user_map: dict):
        # handle links
        links = re.findall(r'https?://\S+', msg)
        for link in links:
            self.add_token(link)

        msg = re.sub(r'https?://\S+', '', msg) # remove them

        msg = msg.lower()

        if re.search(r'\d{18}', msg): # replace user ids with their names
            msg = re.sub(r'\b(' + '|'.join(map(re.escape, user_map.keys())) + r')\b', lambda m: user_map[m.group(0)], msg)

        stripped = re.sub(r'[^a-zA-z\s1-9]+', '', msg) # strip output

        for i in stripped.split():
            self.add_token(i)


def build_tokenizer(username):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    user_table_info = c.execute("PRAGMA table_info(users)").fetchall()
    user_map = {str(uid): name for uid, name, _, _, _ in c.execute("SELECT * FROM users").fetchall()}
    uid = c.execute(f"SELECT id FROM users WHERE name='{username}'").fetchone()
    uid = uid[0]
    ethan_msgs = c.execute(f"SELECT text FROM messages WHERE sender_id={uid} ORDER BY timestamp ASC").fetchall()
    conn.commit()
    conn.close()

if __name__ == "__main__":
    build_tokenizer(ethan_uname)
