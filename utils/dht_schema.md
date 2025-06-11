Table: metadata
Columns:
  - key (TEXT)
  - value (TEXT)

Table: users
Columns:
  - id (INTEGER)
  - name (TEXT)
  - display_name (TEXT)
  - avatar_url (TEXT)
  - discriminator (TEXT)

Table: servers
Columns:
  - id (INTEGER)
  - name (TEXT)
  - type (TEXT)
  - icon_hash (TEXT)

Table: channels
Columns:
  - id (INTEGER)
  - server (INTEGER)
  - name (TEXT)
  - parent_id (INTEGER)
  - position (INTEGER)
  - topic (TEXT)
  - nsfw (INTEGER)

Table: messages
Columns:
  - message_id (INTEGER)
  - sender_id (INTEGER)
  - channel_id (INTEGER)
  - text (TEXT)
  - timestamp (INTEGER)

Table: attachments
Columns:
  - attachment_id (INTEGER)
  - name (TEXT)
  - type (TEXT)
  - normalized_url (TEXT)
  - download_url (TEXT)
  - size (INTEGER)
  - width (INTEGER)
  - height (INTEGER)

Table: message_embeds
Columns:
  - message_id (INTEGER)
  - json (TEXT)

Table: message_reactions
Columns:
  - message_id (INTEGER)
  - emoji_id (INTEGER)
  - emoji_name (TEXT)
  - emoji_flags (INTEGER)
  - count (INTEGER)

Table: message_edit_timestamps
Columns:
  - message_id (INTEGER)
  - edit_timestamp (INTEGER)

Table: message_replied_to
Columns:
  - message_id (INTEGER)
  - replied_to_id (INTEGER)

Table: message_attachments
Columns:
  - message_id (INTEGER)
  - attachment_id (INTEGER)