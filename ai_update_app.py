# ---------------------------------------------------
# Version: 09.07.2024
# Author: M. Weber
# ---------------------------------------------------
# 11.06.2024 Added latest articles on home screen
# 29.06.2024 Added current date to system prompt
# 29.06.2024 Added button "Last 7 days" to search form
# 29.06.2024 Created function write_result()
# 03.07.2024 added UpdateDB; generate_result() replaces write_result()
# 06.07.2024 switched create summary to GROQ
# 09.07.2024 Bug fixes, added function show_latest_articles() and button "Latest articles"
# ---------------------------------------------------

import os
import time
from datetime import datetime
import streamlit as st
import ai_update_module as myapi
import user_management as um

# Define Constants -----------------------------------------------------
SEARCH_TYPES = {"vector": "Vector search", "llm": "LLM search",
                "rag": "RAG search", "fulltext": "Fulltext search"}
TODAY = datetime.now().strftime('%d.%m.%Y')

# Functions ------------------------------------------------------------

@st.experimental_dialog("Login User")
def login_user_dialog() -> None:
    st.write(f"Status: {st.session_state.userStatus}")
    user_name = st.text_input("User")
    user_pw = st.text_input("Passwort", type="password")
    if st.button("Login"):
        if user_name and user_pw:
            if um.check_user(user_name, user_pw):
                st.session_state.userStatus = 'True'
                st.rerun()
            else:
                st.error("User not found.")
        else:
            st.error("Please fill in all fields.")

@st.experimental_dialog("Add User")
def add_user_dialog() -> None:
    user_name = st.text_input("User")
    user_pw = st.text_input("Passwort", type="password")
    if st.button("Add User"):
        if user_name and user_pw:
            um.check_user(user_name, user_pw)
            st.success("User added.")
        else:
            st.error("Please fill in all fields.")

@st.experimental_dialog("UpdateDB")
def update_db_dialog() -> None:
    st.write(f"Anzahl Mails: {myapi.collection_mail_pool.count_documents({})} Anzahl Artikel: {myapi.collection_artikel_pool.count_documents({})}")
    st.write(f"Anzahl Artikel ohne Summary: {myapi.collection_artikel_pool.count_documents({'summary': ''})}")
    st.write(f"Anzahl Artikel ohne Embeddings: {myapi.collection_artikel_pool.count_documents({'summary_embeddings': {}})}")
    if st.button("Import Mails"):
        input_list = myapi.fetch_emails("tldr")
        neu_count, double_count = myapi.add_new_emails(input_list)
        st.success(f"{neu_count} Mails in Datenbank gespeichert [{double_count} Doubletten].")
    if st.button("Extract URLs"):
        neu_count = 0
        double_count = 0
        cursor, count = myapi.text_search_emails("")
        for record in cursor:
            if record.get("processed") == True:
                continue
            st.write(f"[{record.get('date')}] {record.get('title')[:50]}")
            datum, urls = myapi.fetch_tldr_urls(record)
            neu_count, double_count = myapi.add_urls_to_db("tldr", datum, urls)
            myapi.collection_mail_pool.update_one({"_id": record.get('_id')}, {"$set": {"processed": True}})
        st.success(f"{neu_count} URLs in Datenbank gespeichert [{double_count} Doubletten].")
    if st.button("Generate Abstracts"):
        count = 0
        for i in range(10):
            # st.write(f"Generating abstracts {i}/10")
            myapi.generate_abstracts(100)
            i += 1
            count += 1
        st.success(f"{count-10} Abstracts generated.")
    if st.button("Create Embeddings"):
        count = myapi.generate_embeddings(max_iterations=0)
        st.success(f"{count} Embeddings created.")

@st.experimental_dialog("Show Article")
def show_article_dialog(article_id: str) -> None:
    article = myapi.get_article(article_id)
    st.write(f"Title: {article['title']}")
    st.write(f"Date: {article['date']}")
    st.divider()
    st.write(f"Summary: {article['summary']}")
    st.divider()
    st.write(f"URL: {article['url']}")
    time.sleep(3)
    # st.write(f"Content: {article['content']}")

def generate_result(result: dict, url: bool = True, summary: bool = True) -> str:
    combined_result = f"[{str(result['date'])[:10]}] {result['title'][:70]}..."
    if summary:
        combined_result += f"\n\n{result['summary']}\n\n"
    if url:
        combined_result += f" [{result['url']}]"
    return combined_result

def show_latest_articles(max_items: int = 10):
    st.caption("Latest articles:")
    results, count = myapi.text_search_artikel("")
    for result in results[:max_items]:
        st.write(generate_result(result=result, url=True, summary=False))
        # st.write(f"[{str(result['date'])[:10]}] {result['title'][:70]}... ({result['url']})", unsafe_allow_html=True)

# Main -----------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title='AI Insight', initial_sidebar_state="collapsed")

    # Initialize Session State -----------------------------------------
    if 'userStatus' not in st.session_state:
        st.session_state.userStatus = True
        st.session_state.searchStatus = False
        st.session_state.searchPref = "rag"
        st.session_state.llmStatus = "openai"
        st.session_state.results = ""
        st.session_state.history = []
        st.session_state.searchType = "rag" # llm, vector, rag, fulltext
        st.session_state.systemPrompt = f"""
            You are a helpful assistant for tech news.
            Today is {TODAY}.
            Your task is to provide news in the field of artificial intelligence.
            When your answer refers to a specific article, please provide the URL.
            """
    # Define Main Page ------------------------------------------------
    if not st.session_state.userStatus:
        login_user_dialog()
    st.title("AI Insight")
    st.caption("Version 21.07.2024 Status: POC")
    # Define Sidebar ---------------------------------------------------
    with st.sidebar:
        switch_searchType = st.radio(label="Choose Search Type", options=("rag", "llm", "vector", "fulltext"), index=0)
        if switch_searchType != st.session_state.searchType:
            st.session_state.searchType = switch_searchType
            st.rerun()
        switch_llm = st.radio(label="Switch to LLM", options=("groq", "openai"), index=0)
        if switch_llm != st.session_state.llmStatus:
            st.session_state.llmStatus = switch_llm
            st.rerun()
        switch_SystemPrompt = st.text_area("System-Prompt", st.session_state.systemPrompt, height=500)
        if switch_SystemPrompt != st.session_state.systemPrompt:
            st.session_state.systemPrompt = switch_SystemPrompt
            st.rerun()
        if st.button("UpdateDB"):
                update_db_dialog()
        if st.button("Logout"):
            st.session_state.userStatus = False
            st.session_state.searchStatus = False
    # Define Search Form ----------------------------------------------
    with st.form(key="searchForm"):
        question = st.text_area(SEARCH_TYPES[st.session_state.searchType])
        if st.session_state.searchType in ["rag", "llm"]:
            button_caption = "Ask a question"
        else:
            button_caption = "Search"
        col = st.columns([0.4, 0.3, 0.3])
        with col[0]:
            if st.form_submit_button(button_caption):
                st.session_state.searchStatus = True
        with col[1]:
            if st.form_submit_button("7-day summary"):
                question = """
                    What is new from last week? Give a comprehensive summary
                    of all the new developments. Create section for each topic.
                    At the end of the summary, provide a list of the articles used with the urls.
                    """
                st.session_state.searchStatus = True
        with col[2]:
            if st.form_submit_button("Latest articles"):
                st.session_state.searchStatus = False
    # Show latest articles ---------------------------------------------
    if not st.session_state.searchStatus:
        show_latest_articles(max_items=10)
    # Define Search & Search Results -------------------------------------------
    if st.session_state.userStatus and st.session_state.searchStatus:
        # Fulltext Search ---------------------------------------------------
        if st.session_state.searchType == "fulltext":
            results, count = myapi.text_search_artikel(question)
            st.caption(f'Search for: "{question}". {count} articles found.')
            for result in results[:10]:
                st.write(generate_result(result=result, url=True, summary=True))
        # LLM Search -------------------------------------------
        elif st.session_state.searchType == "llm":
            summary = myapi.ask_llm(llm=st.session_state.llmStatus, question=question,
                                    history=st.session_state.history,
                                    systemPrompt=st.session_state.systemPrompt, results="")
            st.write(summary)
        # RAG Search ---------------------------------------------------
        elif st.session_state.searchType == "rag":
            results = myapi.vector_search_artikel(question, 10)
            with st.expander("DB Search Results"):
                results_string = ""
                for result in results:
                    st.write(generate_result(result=result, url=False, summary=False))
                    results_string += f"Date: {str(result['date'])}\nURL: {result['url']}\n Summary: {result['summary']}\n\n"
            summary = myapi.ask_llm(llm=st.session_state.llmStatus, question=question,
                                    history=st.session_state.history,
                                    systemPrompt=st.session_state.systemPrompt, results=results_string)
            st.write(summary)
        # Vector Search ---------------------------------------------------
        elif st.session_state.searchType == "vector":
            results = myapi.vector_search_artikel(question, 10)
            st.session_state.searchStatus = False
            for result in results:
                st.write(generate_result(result=result, url=True, summary=True))
        st.session_state.searchStatus = False

if __name__ == "__main__":
    main()
