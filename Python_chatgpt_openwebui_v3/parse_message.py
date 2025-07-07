import re

def normalize_quotes(s):
    return s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")


def parse_message(message):
    message = normalize_quotes(message)
    match = re.search(r'for\s+file_id="?([\w\-\.]+)"?\s+(.*)', message, re.IGNORECASE)
    if match:
        file_id = match.group(1)
        question = match.group(2).strip()
        print(f"[parse_message] file_id: {file_id}, question: {question}")
        return file_id, question
    else:
        print(f"[parse_message] No file_id found. Using entire message as question: '{message.strip()}'")
        return None, message.strip()
