from datetime import datetime
import os
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Define constants ----------------------------------
load_dotenv()

MONGO_URI = os.environ.get('MONGO_URI')
mongoClient = MongoClient(MONGO_URI)
database = mongoClient.user_pool
collection_user_pool = database.users

# Functions -----------------------------------------

def add_user(user_name, user_pw) -> bool:
    try:
        collection_user_pool.insert_one({
            'user_name': user_name,
            'user_password': user_pw,
            'created_at': datetime.now()
        })
        return True
    except DuplicateKeyError:
        return False
    

def check_user(user_name, user_pw) -> bool:
    user = collection_user_pool.find_one({
        'user_name': user_name,
        'user_password': user_pw
    })
    return user


def delete_user(user_name) -> bool:
    collection_user_pool.delete_one({'user_name': user_name})
    return True


def list_users() -> list:
    users = collection_user_pool.find()
    return users

