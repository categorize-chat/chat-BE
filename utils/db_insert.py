import re
import uuid
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient

import os
from dotenv import load_dotenv

MESSAGE_KEY = 'content'
USER_KEY = 'user'  # nickname에서 user ObjectId로 변경
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

def find_or_create_user_by_nickname(db_client, nickname):
    """nickname으로 사용자를 찾거나 생성합니다."""
    db = db_client.get_database()
    users_collection = db['users']
    
    # 기존 사용자 찾기
    user = users_collection.find_one({'nickname': nickname})
    
    if not user:
        # 임의의 문자열로 userId와 이메일 생성
        random_id = str(uuid.uuid4())[:8]
        temp_email = f"user_{random_id}@temp.example.com"
        
        new_user = {
            'nickname': nickname,
            'email': temp_email,
            'userId': random_id,  # 고유한 userId 설정
            'isVerified': False,
            'isBanned': False,
            'subscriptions': [],
            'readCounts': {},
            'profileUrl': None,
            'lastProfileUpdateTime': None
        }
        result = users_collection.insert_one(new_user)
        user = {'_id': result.inserted_id, **new_user}
        print(f"Created new user: {nickname} with email: {temp_email} and userId: {random_id}")
    
    return user['_id']

def generate_mock_data_from_raw_file(raw_file, room_id, db_client):
    chats = []
    user_nickname = ''
    message = ''
    time = ''
    current_date = None
    line_number = 0

    for text in raw_file:
        line_number += 1
        text = text.strip()

        # Find date
        if text[:15] == '---------------':
            date_string = re.search(r'(\d{4})년 (\d{1,2})월 (\d{1,2})일', text)
            if date_string:
                date_digits = [int(date_char) for date_char in date_string.groups()]
                current_date = datetime.date(datetime(*date_digits))
                print(f"Found date: {current_date} at line {line_number}")
            else:
                print(f"Warning: Expected date format not found at line {line_number}: {text}")
            continue

        # Find new user, message, time
        chat_string = re.match(r'\[(.*?)\] \[(.*?)\] (.*)', text)
        if chat_string:
            # Append the most recent message first
            if user_nickname and message and time and current_date:
                user_id = find_or_create_user_by_nickname(db_client, user_nickname)
                current_time = datetime.combine(current_date, time)
                chats.append({
                    MESSAGE_KEY: message,
                    USER_KEY: user_id,
                    TIME_KEY: current_time,
                    ROOM_KEY: room_id,
                    TOPIC_KEY: -1  # 기본값을 -1로 설정 (스키마 기본값과 일치)
                })
            
            # Set new user to recent one
            user_nickname, time_str, message = chat_string.groups()
            try:
                time = calc_time_from_ampm(time_str)
            except Exception as e:
                print(f"Warning: Invalid time format at line {line_number}: {time_str}")
                time = None
        else:
            # System message (나갔습니다, 들어왔습니다) or Continuing message
            if text[-9:] == '님이 나갔습니다.' or text[-10:] == '님이 들어왔습니다.':
                continue
            message += ' ' + text

    # Append the most recent message
    if user_nickname and message and time and current_date:
        user_id = find_or_create_user_by_nickname(db_client, user_nickname)
        current_time = datetime.combine(current_date, time)
        chats.append({
            MESSAGE_KEY: message,
            USER_KEY: user_id,
            TIME_KEY: current_time,
            ROOM_KEY: room_id,
            TOPIC_KEY: -1
        })

    if not chats:
        print("Warning: No valid chat messages were found in the file.")
    else:
        print(f"Successfully processed {len(chats)} chat messages.")

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
    train_file = './chats/2024grill.txt'

    mongodb_uri = os.getenv('MONGODB_URI')
    room_id = ObjectId('68a81b25e4b11fde53f335b4')

    client = connect_db(mongodb_uri)
    
    with open(train_file, 'r', encoding='UTF-8') as f:
        chats = generate_mock_data_from_raw_file(f, room_id=room_id, db_client=client)

    insert_chat_into_db(db_client=client, collection_name='chats', chats=chats)

    print('Insertion finished')
    print(f'Room ID: {room_id}')
