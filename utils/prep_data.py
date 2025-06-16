import sqlite3
from util import MessageData
from collections import defaultdict
from constants import DB, ethan_id, staleness

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


def get_message_before(message_id, staleness):
    # return the message_id of the message that was sent immediately
    # before the given message in the same channel
    # return None if no such message exists or last message was stale
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # Get the timestamp and channel_id of the given message
    target_timestamp, channel_id = c.execute(
        "SELECT timestamp, channel_id FROM messages WHERE message_id=?",
        (message_id,)
    ).fetchone()
    # Find the most recent message in the same channel with an earlier timestamp
    res = c.execute(
        "SELECT message_id, timestamp FROM messages WHERE channel_id=? AND timestamp<? ORDER BY timestamp DESC LIMIT 1",
        (channel_id, target_timestamp)
    ).fetchone()
    conn.close()

    if res is None:
        return None
    
    prev_message_id, prev_timestamp = res

    if target_timestamp - prev_timestamp > staleness:
        return None

    return prev_message_id


def get_message_after(message_id, staleness):
    # return the message_id of the message that was sent immediately
    # after the given message in the same channel
    # return None if no such message exists or given message was stale
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # Get the timestamp and channel_id of the given message
    target_timestamp, channel_id = c.execute(
        "SELECT timestamp, channel_id FROM messages WHERE message_id=?",
        (message_id,)
    ).fetchone()
    # Find the most recent message in the same channel with a later timestamp
    res = c.execute(
        "SELECT message_id, timestamp FROM messages WHERE channel_id=? AND timestamp>? ORDER BY timestamp ASC LIMIT 1",
        (channel_id, target_timestamp)
    ).fetchone()
    conn.close()

    if res is None:
        return None
    
    next_msg_id, next_timestamp = res

    if next_timestamp - target_timestamp > staleness:
        return None

    return next_msg_id


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


def find_root_reply_msg(msg_id: int, reply_map: dict):
    # geiven a message thats guaranteed to come from user
    orig_sender = get_message_data(msg_id).sender_id
    while get_message_data(msg_id).sender_id == orig_sender and msg_id in reply_map:
        msg_id = reply_map[msg_id]
    return msg_id


def dfs(root_msg_id: int, reply_tree: dict, user_id: int):
    concatenated_replies = []
    s = [[root_msg_id]]
    visited = set()
    visited.add(root_msg_id)
    while s:
        curr = s.pop()
        last_id = curr[-1]
        if last_id not in reply_tree: # if leaf concatenate responses
            concatenated_replies.append(concatenate(curr[1:]))
            continue
        for child in reply_tree[last_id]:
            if child in visited:
                continue
            visited.add(child)
            if get_message_data(child).sender_id != user_id: # if we get to a reply thats not pertinent, concatenate responses
                concatenated_replies.append(concatenate(curr[1:]))
                continue
            else: # if the next reply is still from the target user continue
                s.append(curr + [child])
    
    return concatenated_replies


def concatenate(replies: list[int]):
    # concatenate a list of strings
    reply_text = [get_message_data(msg_id).text for msg_id in replies]
    return " ".join(reply_text)


def parse_replies(user_id: int):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    reply_table = c.execute("SELECT message_id, replied_to_id FROM message_replied_to;").fetchall()
    conn.close()

    all_replies = set()
    reply_map = {}
    reply_tree = defaultdict(list)
    for reply, replied_to in reply_table:
        reply_map[reply] = replied_to
        reply_tree[replied_to].append(reply)
        all_replies.add(reply)

    prompts_and_responses = []
    for msg_id in all_replies:
        # skip if not user of interest
        if get_message_data(msg_id).sender_id != user_id:
            continue

        # find root message
        root_msg_id = find_root_reply_msg(msg_id, reply_map)
        prompt = get_message_data(root_msg_id).text
        if not prompt.strip(): # skip if empty
            continue

        # traverse down and concatenate
        responses = dfs(root_msg_id, reply_tree, user_id)
        responses_filtered = [] # too lazy to do it right only gonna be like max 3 branches
        for r in responses:     # O(n^2) my ass, the visited set is already fucked up
            if r.strip():
                responses_filtered.append(r)
        if len(responses_filtered) < 1:
            continue

        prompts_and_responses.append((prompt, responses_filtered))

    return prompts_and_responses, reply_map


def find_root_msg(msg_id: int):
    # Get initial message
    orig_sender = get_message_data(msg_id).sender_id
    curr = msg_id
    while True:
        prev_msg_id = get_message_before(curr, staleness)
        if prev_msg_id is None:
            return None
        prev_msg = get_message_data(prev_msg_id)
        if prev_msg.sender_id != orig_sender:
            return prev_msg_id
        else:
            curr = prev_msg_id
    

def aggregate_down(root_msg_id: int, user_id: int, visited: set):
    messages = []
    while True:
        next_msg = get_message_after(root_msg_id, staleness)
        if next_msg is None:
            return None
        visited.add(next_msg)
        if not get_message_data(next_msg).text.strip():
            return None
        if get_message_data(next_msg).sender_id != user_id:
            break
        messages.append(get_message_data(next_msg).text)
        root_msg_id = next_msg
    return messages

def parse_database(user_id: int):
    prompts_and_responses, reply_map = parse_replies(user_id)
    visited = set(reply_map.keys())

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    ethan_msgs = c.execute("SELECT message_id FROM messages WHERE sender_id=?;", (ethan_id,)).fetchall()
    conn.close()
    
    for msg_id_t in ethan_msgs:
        msg_id = msg_id_t[0]
        # skip if we processed as reply or seen it
        if msg_id in visited:
            continue
        visited.add(msg_id)
        
        get_message_data(msg_id)

        # find prompt
        root_msg_id = find_root_msg(msg_id)
        if root_msg_id is None:
            continue
        prompt = get_message_data(root_msg_id).text

        # concatenate response
        messages = aggregate_down(root_msg_id, user_id, visited)
        if messages is None or len(messages) < 1:
            continue
        response = ''
        for msg in messages:
            if msg.strip():
                response += msg
        if not response.strip():
            continue
        
        prompts_and_responses.append((prompt, [response]))

    return prompts_and_responses
    

if __name__ == "__main__":
    data = parse_database(ethan_id)
    with open('data.txt', 'w') as f:
        for prompt, responses in data:
            for response in responses:
                f.write(prompt + "\n")
                f.write(response + "\n")
                f.write("\n")
