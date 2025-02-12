import sqlite3
import re
import pickle
from util import Tokenizer

DB = "/home/antonio/dht/archive.dht"
ethan_id = 791492026986135592

conn = sqlite3.connect(DB)
c = conn.cursor()
user_table_info = c.execute("PRAGMA table_info(users)").fetchall()
user_map = {str(uid): name for uid, name, _, _, _ in c.execute("SELECT * FROM users").fetchall()}
conn.commit()
conn.close()


def sanitize_messages(msgs: list[str]):
    ret = []
    for msg in msgs:
        msg = msg.lower()
        if "http" in msg: # ignore websites
            continue

        if re.search(r'\d{18}', msg): # replace user ids with their names
            msg = re.sub(r'\b(' + '|'.join(map(re.escape, user_map.keys())) + r')\b', lambda m: user_map[m.group(0)], msg)

        stripped = re.sub(r'[^a-zA-z\s1-9]+', '', msg) # strip output
        if len(stripped) == 0:
            continue
        
        temp = ["<sos>"]
        temp += stripped.split()
        temp.append("<eos>")

        ret.append(temp)

    return ret

def get_user_msgs(username: str = 'whetan', number: int = 10):
    '''
    return a list of stripped user messages represented as a list of words
    '''
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    ethan_id = c.execute(f"SELECT id FROM users WHERE name='{username}'").fetchone()
    ethan_id = ethan_id[0]

    ethan_msgs = c.execute(f"SELECT text FROM messages WHERE sender_id={ethan_id} ORDER BY timestamp LIMIT {number}").fetchall()
    conn.commit()
    conn.close()

    return sanitize_messages([i[0] for i in ethan_msgs])

def get_user_msgs_unlimited(username: str = 'whetan'):
    '''
    return a list of stripped ethan messages represented as a list of words
    '''
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    ethan_id = c.execute(f"SELECT id FROM users WHERE name='{username}'").fetchone()
    ethan_id = ethan_id[0]

    ethan_msgs = c.execute(f"SELECT text FROM messages WHERE sender_id={ethan_id} ORDER BY timestamp").fetchall()
    conn.commit()
    conn.close()
    
    return sanitize_messages([i[0] for i in ethan_msgs])

if __name__ == "__main__":
    test_msgs = ["Shake down 1979, cool kids never have the time"] * 10

    ethan_msgs = sanitize_messages(test_msgs)
    
    tokenizer = Tokenizer()
    
    # Pad messages
    max_msg_len = max([len(i) for i in ethan_msgs])
    for msg in ethan_msgs:
        if len(msg) < max_msg_len:
            msg += ["<pad>"] * (max_msg_len - len(msg))
    
    tokenizer.fit(ethan_msgs)
    ethan_msgs_rep = [tokenizer.encode(msg) for msg in ethan_msgs]

    with open("ethan_msgs_rep.pkl", "wb") as f:
        pickle.dump(ethan_msgs_rep, f)
    with open("ethan_msgs.pkl", "wb") as f:
        pickle.dump(ethan_msgs, f)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("Done")
