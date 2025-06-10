import sqlite3
from util import MessageData
from collections import defaultdict
from constants import DB

def show_database_info():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables in the database:")
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        print(f"\nTable: {table_name}")

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            # col is a tuple: (cid, name, type, notnull, dflt_value, pk)
            print(f"  - {col[1]} ({col[2]})")

    conn.close()


def get_message_data(msg_id):
    # return a dict of relevant message info
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    sender_id, channel_id, text, timestamp = c.execute(
        "SELECT sender_id, channel_id, text, timestamp FROM messages WHERE message_id=?;",
        (msg_id,)
    ).fetchone()
    conn.close()
    return MessageData(sender_id, channel_id, text, timestamp)


def find_root_message(msg_id, reply_map, user_id):
    while get_message_data(msg_id).sender_id == user_id:
        msg_id = reply_map[msg_id]
    return msg_id


def concatenate_replies(user_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    reply_table = c.execute("SELECT message_id, replied_to_id FROM message_replied_to;").fetchall()
    all_msgs = set()
    reply_map = {}
    child_map = defaultdict(list)
    for reply, replied_to in reply_table:
        reply_map[reply] = replied_to
        child_map[replied_to].append(reply)
        all_msgs.add(reply)
        all_msgs.add(replied_to)

    prompts_and_repsonses = []

    for msg_id in all_msgs:
        # skip if not user of interest
        if get_message_data(msg_id).sender_id != user_id:
            continue

        # find root message
        root_msg = find_root_message(msg_id, reply_map, user_id)
        prompt = get_message_data(root_msg).text

        # traverse down and concatenate
        visited = set()
        
            

    
    

if __name__ == "__main__":
    pass