# ---------------------------------------------------
# Version: 25.07.2024
# Author: M. Weber
# ---------------------------------------------------
# 09.06.2024 Updated code with chatdvv module.
# 11.06.2024 Updated sort parameter in text_search_artikel function.
# 11.06.2024 Add latest list on home screen.
# 29.06.2024 Added current date to System Prompt
# 03.07.2024 modified generate_abstracts function 
# 06.07.2024 switched create summary to GROQ
# 25.07.2024 switched all to gpt 4o mini and added llama 3.1
# ---------------------------------------------------

from datetime import datetime
import os
from bson import ObjectId
from dotenv import load_dotenv
import re

import requests
import imaplib
import email
from bs4 import BeautifulSoup

# from validators import url as valid_url

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import openai
from groq import Groq
# import ollama

import torch
from transformers import AutoTokenizer, AutoModel

# Init MongoDB Client and Collections ---------------------
load_dotenv()
mongoClient = MongoClient(os.environ.get('MONGO_URI_PRIVAT'))
database = mongoClient.ki_update_db
collection_mail_pool = database.mail_pool
collection_artikel_pool = database.artikel_pool

openaiClient = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY_DVV'))
groqClient = Groq(api_key=os.environ['GROQ_API_KEY_PRIVAT'])

# Load pre-trained model and tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

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
    # get a list of all email IDs
    _, data = imap.search(None, f'(FROM "{sender}")')
    decoded_string = data[0].decode('utf-8')
    email_ids = decoded_string.split()
    # initiate list to store email text data
    email_list = []
    # fetch and parse each email
    for email_id in email_ids:
        # fetch email data html encoded
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
        #add subject_text and date_text and message_text to list
        email_list.append([subject_text, date_text, message_text])
    # logout and close the connection when you're done
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

def fetch_tldr_urls(record: tuple) -> [str, list]:
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
        # convert list to json
        record_dict = {"title": "", "date": date, "url": url, "summary": "", "summary_embeddings": {}, "source": source}
        try:
            collection_artikel_pool.insert_one(record_dict)
            new_count += 1
        except DuplicateKeyError:
            duplicate_error_count += 1
    return new_count, duplicate_error_count

def generate_abstracts(max_iterations: int = 20) -> None:
    cursor = collection_artikel_pool.find({'summary': ""}).limit(max_iterations)
    iteration = 0
    for record in cursor:
        iteration += 1
        record_url = record.get('url')
        print(record_url[:50])
        start_time = datetime.now()
        title, summary = write_summary(record_url)
        end_time = datetime.now()
        duration = end_time - start_time
        print(title[:50])
        print(f"#{iteration} Duration: {round(duration.total_seconds(), 2)}")
        print("-"*50)
        collection_artikel_pool.update_one({"_id": record.get('_id')}, {"$set": {"title": title, "summary": summary}})
    cursor.close()

def write_summary(url: str) -> [str, str]:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A"}
    try:
        response = requests.get(url, headers=headers)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "error", "error"
    soup = BeautifulSoup(response.text, 'html.parser')
    if soup.body:
        text = soup.get_text()
        text = text[:900]
        question = f"Extract the abstract from the following URL: {text}. Don't start with 'Abstract:'. Don't include Title or Author. No comments, only the main text."
        summary = ask_llm(llm="openai_mini", question=question, history=[], systemPrompt="", results="")
        question = f"Generate one blog title for the following abstract: {summary}. The answer should only be one sentence long and just contain the text of the title. No comments, only the title text."
        title = ask_llm(llm="openai_mini", question=question, history=[], systemPrompt="", results="")
    else:
        title = "empty"
        summary = "empty"
    return title, summary

def generate_embeddings(max_iterations: int = 10) -> int:
    query = {"summary_embeddings": {}, "summary": {"$ne": ""}}
    cursor = collection_artikel_pool.find(query)
    count = collection_artikel_pool.count_documents(query)
    print(f"Count: {count}")
    iteration = 0
    for record in cursor:
        iteration += 1
        if (max_iterations > 0) and (iteration > max_iterations):
            break
        summary_text = record.get('summary')
        if summary_text is None:
            summary_text = "Keine Zusammenfassung vorhanden."
        embeddings = create_embeddings([summary_text])
        collection_artikel_pool.update_one({"_id": record.get('_id')}, {"$set": {"summary_embeddings": embeddings}})
    return iteration

def create_embeddings(text: str) -> str:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings_list = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings_list

def ask_llm(llm: str, question: str, history: list = [], systemPrompt: str = "", results : str = "") -> str:
    prompt = [{"role": "system", "content": systemPrompt}]
    if results != "":
        prompt.append({"role": "assistant", "content": f"Here is some relevant information: {results}"})
        prompt.append({"role": "user", "content": f"Based on the given information, {question}"})
    else:
        prompt.append({"role": "user", "content": f"{question}"})
    if llm == "GPT 4o":
        response = openaiClient.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=prompt
        )
        output = response.choices[0].message.content
    elif llm == "GPT 4o mini":
        response = openaiClient.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=prompt
        )
        output = response.choices[0].message.content
    elif llm == "LLAMA 3.1 70b":
        response = ollama.chat(
            model="llama-3.1-70b-versatile", 
            messages=[{"role": "user", "content": f"Based on the given information, {question}"}]
            )
        output = response['message']['content']
    else:
        output = "Error: No valid LLM specified."
    return output

def text_search_emails(search_text: str = "") -> [tuple, int]:
    if search_text != "":
        query = {"$text": {"$search": search_text }}
    else:
        query = {}
    cursor = collection_mail_pool.find(query).sort([("date", -1)])
    count = collection_mail_pool.count_documents(query)
    return cursor, count

def text_search_artikel(search_text: str = "", sort_parameter: bool = True) -> [tuple, int]:
    if search_text != "":
        query = {"$text": {"$search": search_text }}
    else:
        query = {}
    fields = {"_id": 1, "title": 1, "date": 1, "url": 1, "summary": 1}
    sort = [("date", -1)] if sort_parameter else []
    cursor = collection_artikel_pool.find(query, fields).sort(sort)
    count = collection_artikel_pool.count_documents(query)
    return cursor, count

def vector_search_artikel(query_string: str, limit: int = 10) -> tuple:
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
            "score": {"$meta": "vectorSearchScore"}
            }
        }
        ]   
    result = collection_artikel_pool.aggregate(pipeline)
    return result

def get_article(id: str) -> dict:
    return collection_artikel_pool.find_one({"_id": ObjectId(id)})

def print_results(cursor: tuple, limit: int = 10) -> None:
    if not cursor:
        print("Keine Artikel gefunden.")
    for i in cursor[:limit]:
        print(f"[{str(i['date'])[:10]}] {i['title'][:70]}")
        # print("-"*80)