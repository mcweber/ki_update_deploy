# ---------------------------------------------------
# Version: 13.10.2024
# Author: M. Weber
# ---------------------------------------------------
# 25.07.2024 switched all to gpt 4o mini and added llama 3.1
# 20.08.2024 Corrected write_summary function
# 29.09.2024 switched tuples to lists in all functions
# 30.09.2024 added last_days parameter to text_search_artikel
# 06.10.2024 added llama prompt generation function
# 13.10.2024 return keywords as list
# ---------------------------------------------------

from datetime import datetime, timedelta
import os
from bson import ObjectId
from dotenv import load_dotenv
import re

import requests
import imaplib
import email
from bs4 import BeautifulSoup

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import openai
# from groq import Groq
import ollama

import torch
from transformers import AutoTokenizer, AutoModel

# Init MongoDB Client and Collections ---------------------
load_dotenv()
mongoClient = MongoClient(os.environ.get('MONGO_URI_PRIVAT'))
database = mongoClient.ki_update_db
collection_mail_pool = database.mail_pool
collection_artikel_pool = database.artikel_pool

openaiClient = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY_PRIVAT'))
# groqClient = Groq(api_key=os.environ['GROQ_API_KEY_PRIVAT'])

# Load pre-trained model and tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define Ollama check function ----------------------------
def ollama_active(llm_name: str) -> bool:
    try:
        if ollama.show(llm_name) != {}:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# Define String functions ---------------------------

def remove_encoded_words(text: str, pattern: str = "UNICODE") -> str:
    if pattern.upper() == "UNICODE":
        pattern = r' =\?[^?]+\?[^?]+\?[^?]+\?='
    return re.sub(pattern, '', text)

def convert_date_string(date_string: str) -> datetime:
    date_format = "%a, %d %b %Y %H:%M:%S %z"
    return datetime.strptime(date_string, date_format)

# Define Database functions ----------------------------------

def fetch_emails(sender: str) -> list:
    IMAP_SERVER = 'imap.web.de'
    USERNAME = 'maweber@web.de'
    PASSWORD = 'hvn6gtv!EPZ8xmy0nrm'
    # Login to the IMAP server
    imap = imaplib.IMAP4_SSL(IMAP_SERVER)
    imap.login(USERNAME, PASSWORD)
    imap.select('INBOX')
    # Get a list of all email IDs
    _, data = imap.search(None, f'(FROM "{sender}")')
    decoded_string = data[0].decode('utf-8')
    email_ids = decoded_string.split()
    email_list = []
    # Fetch and parse each email
    for email_id in email_ids:
        # Fetch email data html encoded
        _, email_data = imap.fetch(email_id, "(RFC822)")
        message = email.message_from_bytes(email_data[0][1])
        subject_text = remove_encoded_words(message.get('Subject'))
        subject_text = remove_encoded_words(subject_text, pattern = "\r\n")
        date_text = message.get('Date')
        message_text = ""
        for part in message.walk():
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                payload = payload.decode()
            message_text += str(payload)
        # Add subject_text and date_text and message_text to list
        email_list.append([subject_text, date_text, message_text])
    imap.close()
    imap.logout()
    return email_list

def add_new_emails(email_list: list) -> [int, int]:
    duplicate_error_count = 0
    new_emails_count = 0
    for email in email_list:
        email_dict = {"title": email[0], "date": convert_date_string(email[1]), "body": email[2], "processed": False}
        try:
            collection_mail_pool.insert_one(email_dict)
            new_emails_count += 1
        except DuplicateKeyError:
            duplicate_error_count += 1
    return new_emails_count, duplicate_error_count

def fetch_tldr_urls(record: list) -> [str, list]:
    record_date = record.get('date')
    record_body = ''.join(record.get('body'))
    index = record_body.find("<!DOCTYPE html>")
    if index != -1:
        record_body = record_body[:index]
    # collect urls
    record_urls = re.findall(r'(https?://\S+)', record_body)
    record_urls = list(set(record_urls))
    #delete all urls that contain "tldrai"
    record_urls = [url for url in record_urls if "refer.tldr" not in url]
    record_urls = [url for url in record_urls if "tldrnewsletter" not in url]
    record_urls = [url for url in record_urls if "tldr.tech" not in url]
    #delete string "?utm_source=...""
    record_urls = [re.sub(r'\?utm_source=.*', '', url) for url in record_urls]
    return record_date, record_urls

def add_urls_to_db(source: str, date: datetime, urls: list = []) -> [int, int]:
    duplicate_error_count = 0
    new_count = 0
    for url in urls:
        record_dict = {"title": "", "date": date, "url": url, "summary": "", "summary_embeddings": {}, "source": source}
        try:
            collection_artikel_pool.insert_one(record_dict)
            new_count += 1
        except DuplicateKeyError:
            duplicate_error_count += 1
    return new_count, duplicate_error_count

def generate_summary_title(llm: str, url: str) -> [str, str]:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "Error: {e}", "Error: {e}"
    if soup.body:
        text = soup.get_text(separator="\n", strip=True)
        text = text[:2000]
        task = f"Extract the abstract from the following URL: {text}. Don't start with 'Abstract:'. Don't include Title or Author. No comments, only the main text."
        summary = ask_llm(llm=llm, question=task, system_prompt="")
        task = f"Generate one blog title for the following abstract: {summary}. The answer should only be one sentence long and just contain the text of the title. No comments, only the title text."
        title = ask_llm(llm=llm, question=task, system_prompt="")
    else:
        title = "empty"
        summary = "empty"
    return title, summary

def generate_keywords(llm: str, text: str = "", max_keywords: int = 5) -> list:
    if text == "":
        return "empty"
    system_prompt = """
                    You are an experienced editor, spezialized in tech and AI related topics..
                    You are an expert in generating keywords that help describe and cluster news articles.
                    """
    task = f"""
            Based on the following text: "{text}".
            Generate a maximum of {max_keywords} keywords.
            The answer must only consist of the keywords,
            with the following format: "keyword1, keyword2, keyword3, ..."
            """
    keywords_str = ask_llm(llm=llm, question=task, system_prompt=system_prompt)
    keywords_list = [keyword.strip() for keyword in keywords_str.split(',') if keyword.strip()]
    return keywords_list

def create_embeddings(text: str) -> []:
    # inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", clean_up_tokenization_spaces=True)
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings_list = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings_list

def generate_llama_prompt(prompt: list = []) -> str:
    prompt_string = ""
    for text in prompt:
        if text['role'] == "user":
            prompt_string += f"<|start_header_id|>user<|end_header_id|>{text['content']}"
        elif text['role'] == "assistant":
            prompt_string += f"{text['content']}"
        elif text['role'] == "system":
            prompt_string += f"{text['content']}"
    prompt_string += "<|eot_id|>"
    return prompt_string

def ask_llm(llm: str, question: str, history: list = [], system_prompt: str = "", results : str = "") -> str:
    # generate prompt -----------------------------------
    prompt = [{"role": "system", "content": system_prompt}]
    if results != "":
        prompt.append({"role": "assistant", "content": f"Here is some relevant information: {results}"})
        prompt.append({"role": "user", "content": f"Based on the given information, {question}"})
    else:
        prompt.append({"role": "user", "content": f"{question}"})
    # call LLM -------------------------------------------
    if llm == "GPT 4o":
        response = openaiClient.chat.completions.create(
        model="gpt-4o", temperature=0.2, messages=prompt)
        output = response.choices[0].message.content
    elif llm == "GPT 4o mini":
        response = openaiClient.chat.completions.create(model="gpt-4o-mini", temperature=0.2, messages=prompt)
        output = response.choices[0].message.content
    elif llm == "LLAMA 3.X":
        response = ollama.chat(model="llama3.2", messages=[{'role': 'user', 'content': generate_llama_prompt(prompt)}])
        output = response['message']['content']
    else:
        output = "Error: No valid LLM specified."
    return output

def text_search_emails(search_text: str = "") -> [list, int]:
    if search_text != "":
        query = {"$text": {"$search": search_text }}
    else:
        query = {}
    cursor = collection_mail_pool.find(query).sort([("date", -1)])
    count = collection_mail_pool.count_documents(query)
    return cursor, count

def list_keywords() -> list:
    pipeline = [
    {
    '$unwind': '$keywords'
    },
    {
    '$group': {
        '_id': '$keywords', 
        'count': {'$sum': 1}
        }
    },
    {
    '$sort': {'count': -1}
    },
    {
    '$project': {
        '_id': 0, 
        'keyword': '$_id', 
        'count': 1
        }
    }
    ]
    cursor = collection_artikel_pool.aggregate(pipeline)
    return list(cursor)

def text_search_artikel(search_text : str = "*", gen_schlagworte: bool = False, score: float = 0.0, limit: int = 10, last_days: int = 0) -> [list, str]:
    query = {
        "index": "volltext_gewichtet",
        "text": {
            "query": query_input,
            # "path": {"wildcard": "*"}
            "path": ["title", "summary", "keywords"]
            # "path": "title"
            }
        }
    fields = {
        "_id": 1,
        "title": 1,
        "date": 1,
        "url": 1,
        "summary": 1,
        "source": 1,
        "keywords": 1,
        "score": {"$meta": "searchScore"},
        }
    pipeline = [
        {"$search": query},
        {"$project": fields},
        {"$match": {"score": {"$gte": score}}},
        {"$sort": {"date": -1}},
        {"$limit": limit},
        ]
    if last_days > 0:
        days_ago = datetime.now() - timedelta(days=last_days)
        pipeline.insert(2, {"$match": {"date": {"$gte": days_ago}}})
    cursor = collection_artikel_pool.aggregate(pipeline)
    return list(cursor), query_input

def keyword_search_artikel(search_text : str = "", gen_schlagworte: bool = False, score: float = 0.0, limit: int = 10, last_days: int = 0) -> [list, str]:
    query_input = search_text
    query = {
        "index": "volltext_gewichtet",
        "text": {
            "query": query_input,
            # "path": ["title", "summary", "keywords"]
            "path": "keywords"
            }
        }
    fields = {
        "_id": 1,
        "title": 1,
        "date": 1,
        "url": 1,
        "summary": 1,
        "source": 1,
        "keywords": 1,
        "score": {"$meta": "searchScore"},
        }
    pipeline = [
        {"$search": query},
        {"$project": fields},
        {"$sort": {"date": -1}},
        {"$limit": limit},
        ]
    if last_days > 0:
        days_ago = datetime.now() - timedelta(days=last_days)
        pipeline.insert(2, {"$match": {"date": {"$gte": days_ago}}})
    cursor = collection_artikel_pool.aggregate(pipeline)
    return list(cursor), query_input

def text_search_artikel(search_text : str = "*", gen_schlagworte: bool = False, score: float = 0.0, limit: int = 10, last_days: int = 0) -> [list, str]:
    query_input = "No query_input."
    if search_text == "":
        return [], query_input
    if search_text == "*":
        schlagworte = "*"
        score = 0.0
        query = {
            "index": "volltext_gewichtet",
            "exists": {"path": "title"},
            }
    else:
        query_input = generate_keywords(llm="GPT 4o mini", text=search_text) if gen_schlagworte else search_text
        query = {
            "index": "volltext_gewichtet",
            "text": {
                "query": query_input,
                # "path": {"wildcard": "*"}
                "path": ["title", "summary", "keywords"]
                # "path": "title"
                }
            }
    fields = {
        "_id": 1,
        "title": 1,
        "date": 1,
        "url": 1,
        "summary": 1,
        "source": 1,
        "keywords": 1,
        "score": {"$meta": "searchScore"},
        }
    pipeline = [
        {"$search": query},
        {"$project": fields},
        {"$match": {"score": {"$gte": score}}},
        {"$sort": {"date": -1}},
        {"$limit": limit},
        ]
    if last_days > 0:
        days_ago = datetime.now() - timedelta(days=last_days)
        pipeline.insert(2, {"$match": {"date": {"$gte": days_ago}}})
    cursor = collection_artikel_pool.aggregate(pipeline)
    return list(cursor), query_input

def vector_search_artikel(query_string: str, limit: int = 10) -> list:
    embeddings = create_embeddings(query_string)
    pipeline = [
        {"$vectorSearch": {
            "index": "summary_vector_index",
            "path": "summary_embeddings",
            "queryVector": embeddings,
            "numCandidates": int(limit * 10),
            "limit": limit,
            }
        },
        {"$project": {
            "_id": 1,
            "title": 1,
            "date": 1,
            "url": 1,
            "summary": 1,
            "source": 1,
            "keywords": 1,
            "score": {"$meta": "vectorSearchScore"}
            }
        }
        ]
    result = collection_artikel_pool.aggregate(pipeline)
    return result

def get_article(id: str) -> dict:
    return collection_artikel_pool.find_one({"_id": ObjectId(id)})

def print_results(cursor: list, limit: int = 10) -> None:
    if not cursor:
        print("Keine Artikel gefunden.")
    for i in cursor[:limit]:
        print(f"[{str(i['date'])[:10]}] {i['title'][:70]}")
