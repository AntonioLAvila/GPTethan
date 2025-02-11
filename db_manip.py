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

    ret = []
    for msg, in ethan_msgs:
        if "http" in msg: # ignore websites
            continue

        if re.search(r'\d{18}', msg): # replace user ids with their names
            msg = re.sub(r'\b(' + '|'.join(map(re.escape, user_map.keys())) + r')\b', lambda m: user_map[m.group(0)], msg)

        stripped = re.sub(r'[^a-zA-z\s]+', '', msg) # strip output
        if len(stripped) == 0:
            continue
        
        temp = ["<sos>"]
        temp += stripped.split()
        temp.append("<eos>")

        ret.append(temp)
    
    return ret

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

    ret = []
    for msg, in ethan_msgs:
        if "http" in msg: # ignore websites
            continue

        if re.search(r'\d{18}', msg): # replace user ids with their names
            msg = re.sub(r'\b(' + '|'.join(map(re.escape, user_map.keys())) + r')\b', lambda m: user_map[m.group(0)], msg)

        stripped = re.sub(r'[^a-zA-z\s]+', '', msg) # strip output
        if len(stripped) == 0:
            continue
        
        temp = ["<sos>"]
        temp += stripped.split()
        temp.append("<eos>")

        ret.append(temp)
    
    return ret

if __name__ == "__main__":
    ethan_msgs = get_user_msgs_unlimited('whetan')

    tokenizer = Tokenizer()

    tokenizer.fit(ethan_msgs)
    
    # Pad messages
    max_msg_len = max([len(i) for i in ethan_msgs])
    for msg in ethan_msgs:
        if len(msg) < max_msg_len:
            length_to_extend = max_msg_len - len(msg)
            msg.pop()
            msg += ["<pad>"] * (length_to_extend)
            msg += ["<eos>"]
    
    ethan_msgs_rep = [tokenizer.encode(msg) for msg in ethan_msgs]

    with open("ethan_msgs_rep.pkl", "wb") as f:
        pickle.dump(ethan_msgs_rep, f)
    # with open("ethan_msgs.pkl", "wb") as f:
    #     pickle.dump(ethan_msgs, f)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("Done")
