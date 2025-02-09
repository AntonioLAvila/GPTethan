import sqlite3
import re
import torch
import pickle

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
    id_to_word = {0: ' '}
    word_to_id = {' ': 0}
    i = 1
    for msg in messages:
        for word in msg:
            if word not in id_to_word:
                id_to_word[i] = word
                word_to_id[word] = i
                i += 1
    return id_to_word, word_to_id

def msg_to_rep(msg):
    '''
    return a list of word ids
    '''
    # rep = []
    # for word in msg:
    #     word_rep = [0]*vocab_size
    #     word_rep[word_to_id[word]] = 1
    #     rep.append(word_rep)
    # return rep
    return [word_to_id[word] for word in msg]

def rep_to_msg(msg):
    '''
    return list of words from list of ids
    '''
    # return [id_to_word[torch.argmax(encoding)] for encoding in msg]
    return [id_to_word[wid] for wid in msg]

if __name__ == "__main__":
    ethan_msgs = get_user_msgs('whetan', 50)
    max_msg_len = max([len(i) for i in ethan_msgs])
    for msg in ethan_msgs:
        if len(msg) < max_msg_len:
            msg += [' '] * (max_msg_len - len(msg))

    id_to_word, word_to_id = vocab_mapping(ethan_msgs)

    ethan_msgs_rep = [msg_to_rep(msg) for msg in ethan_msgs]

    with open("ethan_msgs_rep.pkl", "wb") as f:
        pickle.dump(ethan_msgs_rep, f)
    with open("ethan_msgs.pkl", "wb") as f:
        pickle.dump(ethan_msgs, f)
    with open("id_to_word.pkl", "wb") as f:
        pickle.dump(id_to_word, f)
    with open("word_to_id.pkl", "wb") as f:
        pickle.dump(word_to_id, f)

    print("Done")
