import re
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient

import os
from dotenv import load_dotenv

MESSAGE_KEY = 'content'
USER_KEY = 'nickname'
TIME_KEY = 'createdAt'
ROOM_KEY = 'room'
TOPIC_KEY = 'topic'

def calc_time_from_ampm(time_str):
    parsed_time_str = re.match(r'(오전|오후) (\d{1,2}):(\d{1,2})', time_str)

    if (not parsed_time_str):
        raise Exception('Invalid time format')

    ampm, _hour, _min = parsed_time_str.groups()

    hour = int(_hour)
    min = int(_min)
    if (ampm == '오후' and _hour != '12'):
        hour = int(_hour) + 12
    if (ampm == '오전' and _hour == '12'):
        hour = 0
    
    return datetime.time(datetime(1, 1, 1, hour, min, 0))

def generate_mock_data_from_raw_file(raw_file, room_id):
    chats = []

    user = ''
    message = ''
    time = ''

    while (text := raw_file.readline()):
        text = text.strip()

        # Find date
        if (text[:15] == '---------------'):
            date_string = re.search(r'(\d{4})년 (\d{1,2})월 (\d{1,2})일', text)
            date_digits = [int(date_char) for date_char in date_string.groups()]
            current_date = datetime.date(datetime(*date_digits))
            continue

        # Find new user, message, time
        chat_string = re.match(r'\[(.*?)\] \[(.*?)\] (.*)', text)
        if (chat_string):
            # Append the most recent message first
            if (user and message and time):
                current_time = datetime.combine(current_date, time)

                chats.append({
                    MESSAGE_KEY: message,
                    USER_KEY: user,
                    TIME_KEY: current_time,
                    ROOM_KEY: room_id,
                    TOPIC_KEY: ''
                })
                
            # Set new user to recent one
            user, time_str, message = chat_string.groups()
            time = calc_time_from_ampm(time_str)
        
        else:
            # System message (나갔습니다, 들어왔습니다) or Continuing message
            if (text[-9:] == '님이 나갔습니다.' or text[-10:] == '님이 들어왔습니다.'):
                continue

            message += ' ' + text
    
    # Append the most recent message first
    if (user and message and time):
        current_time = datetime.combine(current_date, time)
        
        chats.append({
            MESSAGE_KEY: message,
            USER_KEY: user,
            TIME_KEY: current_time,
            ROOM_KEY: room_id,
            TOPIC_KEY: ''
        })
        

    return chats



def connect_db(mongodb_uri):
    if not mongodb_uri or not mongodb_uri.startswith(('mongodb://', 'mongodb+srv://')):
        raise ValueError("Invalid MONGODB_URI. It must start with 'mongodb://' or 'mongodb+srv://'")

    client = MongoClient(mongodb_uri)

    return client

def insert_chat_into_db(db_client, collection_name, chats):
    db = db_client.get_database()
    collection = db[collection_name]

    collection.insert_many(chats)


if __name__ == '__main__':
    # .env 파일 로드
    load_dotenv()
    train_file = './crop_chat.txt'

    mongodb_uri = os.getenv('MONGODB_URI')
    room_id = ObjectId()

    with open(train_file, 'r', encoding='UTF-8') as f:
        chats = generate_mock_data_from_raw_file(f, room_id=room_id)

    client = connect_db(mongodb_uri)
    insert_chat_into_db(db_client=client, collection_name='chats-sub', chats=chats)

    print('Insertion finished')
    print(f'Room ID: {room_id}')
