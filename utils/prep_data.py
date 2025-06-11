import sqlite3
from util import MessageData
from collections import defaultdict
from constants import DB, ethan_id

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


def get_message_data(msg_id: int):
    # return relevant message info
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    sender_id, channel_id, text, timestamp = c.execute(
        "SELECT sender_id, channel_id, text, timestamp FROM messages WHERE message_id=?;",
        (msg_id,)
    ).fetchone()
    conn.close()
    return MessageData(sender_id, channel_id, text, timestamp)


def remove_unreachable():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    reply_table = c.execute("SELECT message_id, replied_to_id FROM message_replied_to;").fetchall()
    conn.close()

    start_len_reply_table = len(reply_table)

    all_replies = set()
    for i in reply_table:
        for j in i:
            all_replies.add(j)
    
    removed_ids = []
    for msg_id in all_replies:
        try:
            get_message_data(msg_id)
        except:
            removed_ids.append(msg_id)
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            c.execute(
                "DELETE FROM message_replied_to WHERE message_id=? OR replied_to_id=?;",
                (msg_id, msg_id)
            )
            conn.commit()
            conn.close()

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    reply_table = c.execute("SELECT message_id, replied_to_id FROM message_replied_to;").fetchall()
    conn.close()

    finish_len_reply_table = len(reply_table)
    
    print(f"{len(removed_ids)}/{len(all_replies)} were removed")
    print(f"reply table length went {start_len_reply_table} -> {finish_len_reply_table}")
    print(f"{removed_ids}")


def find_root_message(msg_id: int, reply_map: dict, user_id: int):
    while get_message_data(msg_id).sender_id == user_id and msg_id in reply_map:
        msg_id = reply_map[msg_id]
    return msg_id


def dfs(root_msg_id: int, reply_tree: dict, user_id: int, visited: set):
    concatenated_replies = []
    s = [[root_msg_id]]
    visited.add(root_msg_id)
    while s:
        curr = s.pop(-1)
        if curr[-1] not in reply_tree: # if leaf concatenate responses
            concatenated_replies.append(concatenate_replies(curr[1:]))
            continue
        for child in reply_tree[curr[-1]]:
            if child in visited:
                continue
            visited.add(child)
            if get_message_data(child).sender_id != user_id: # if we get to a reply thats not pertinent, concatenate responses
                concatenated_replies.append(concatenate_replies(curr[1:]))
                continue
            else: # if the next reply is still from the target user continue
                curr.append(child)
                s.append(curr)
    
    return concatenated_replies


def concatenate_replies(replies: list[int]):
    reply_text = [get_message_data(msg_id).text for msg_id in replies]
    return " ".join(reply_text)


def parse_replies(user_id: int):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    reply_table = c.execute("SELECT message_id, replied_to_id FROM message_replied_to;").fetchall()
    conn.close()

    all_msgs = set()
    reply_map = {}
    reply_tree = defaultdict(list)
    for reply, replied_to in reply_table:
        reply_map[reply] = replied_to
        reply_tree[replied_to].append(reply)
        all_msgs.add(reply)
        all_msgs.add(replied_to)

    prompts_and_repsonses = []
    visited = set()
    for msg_id in all_msgs:
        # skip if not user of interest
        if get_message_data(msg_id).sender_id != user_id:
            continue

        # find root message
        root_msg_id = find_root_message(msg_id, reply_map, user_id)
        prompt = get_message_data(root_msg_id).text

        # traverse down and concatenate
        responses = dfs(root_msg_id, reply_tree, user_id, visited)

        prompts_and_repsonses.append((prompt, responses))

    return prompts_and_repsonses, reply_map
    

if __name__ == "__main__":
    data, _ = parse_replies(ethan_id)
    print(data)