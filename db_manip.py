import sqlite3
import re

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
    return a list of stripped ethan messages represented as a list of words
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

        ret.append(stripped.split())
    
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

        ret.append(stripped.split())
    
    return ret


def vocab_mapping(messages: list[str]):
    '''
    return map of word id number to word and vice versa
    '''
    id_to_word = {}
    word_to_id = {}
    i = 0
    for msg in messages:
        for word in msg:
            if word not in id_to_word:
                id_to_word[i] = word
                word_to_id[word] = i
                i += 1
    return id_to_word, word_to_id

