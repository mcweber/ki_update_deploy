# ---------------------------------------------------
VERSION = "17.10.2024"
# Author: M. Weber
# ---------------------------------------------------
# 28.09.2024 switch to llama 3.2
# 06.10.2024 bug fixing and code cleaning
# 08.10.2024 Export Prompt function
# 17.10.2024 bug fixes
# ---------------------------------------------------

# import os
# import time
from datetime import datetime
import streamlit as st
import ai_update_module as myapi

# Define Constants -----------------------------------------------------
SEARCH_TYPES = {"rag": "RAG search", "vector": "Vector search", "fulltext": "Fulltext search", "keywords": "Keyword search"}
TODAY = datetime.now().strftime('%d.%m.%Y')

# Functions ------------------------------------------------------------
@st.dialog("UpdateDB")
def update_db_dialog() -> None:
    st.write(f"Anzahl Mails: {myapi.collection_mail_pool.count_documents({})} Anzahl Artikel: {myapi.collection_artikel_pool.count_documents({})}")
    st.write(f"Anzahl Artikel ohne Summary: {myapi.collection_artikel_pool.count_documents({'summary': ''})}")
    st.write(f"Anzahl Artikel ohne Embeddings: {myapi.collection_artikel_pool.count_documents({'summary_embeddings': {}})}")
    st.write(f"Anzahl Artikel ohne Keywords: {myapi.collection_artikel_pool.count_documents({'keywords': ''})}")
    if st.button("Import Mails & Extract URLs"):
        # import mails --------------------------------------------------
        input_list = myapi.fetch_emails("tldr")
        neu_count, double_count = myapi.add_new_emails(input_list)
        st.success(f"{neu_count} Mails in Datenbank gespeichert [{double_count} Doubletten].")
        # extract urls ---------------------------------------------------
        neu_count = 0
        double_count = 0
        cursor, count = myapi.text_search_emails("")
        for record in cursor:
            if record.get("processed") == True:
                continue
            datum, urls = myapi.fetch_tldr_urls(record)
            count, dcount = myapi.add_urls_to_db("tldr", datum, urls)
            neu_count += count
            double_count += dcount
            myapi.collection_mail_pool.update_one({"_id": record.get('_id')}, {"$set": {"processed": True}})
        st.success(f"{neu_count} URLs in Datenbank gespeichert [{double_count} Doubletten].")
    if st.button("Generate title, abstracts, keywords & embeddings"):
        iteration = 0
        print("Generating title, abstracts, keywords & embeddings")
        cursor = myapi.collection_artikel_pool.find({'summary': ""})
        if cursor:
            iteration = 0
            results_list = list(cursor)
            for record in results_list:
                id = record.get('_id')
                url = record.get('url')
                title, summary = myapi.generate_summary_title(llm=st.session_state.llm_status, url=url)
                sum_embeddings = myapi.create_embeddings(record.get('summary'))
                keywords = myapi.generate_keywords(llm=st.session_state.llm_status, text=summary)
                keyw_embeddings = myapi.create_embeddings(keywords)
                print(f"[{iteration}] Title: {title}")
                myapi.collection_artikel_pool.update_one(
                    {"_id": id}, 
                    {"$set": {
                        "title": title,
                        "summary": summary,
                        "summary_embeddings": sum_embeddings,
                        "keywords": keywords,
                        "keywords_embeddings": keyw_embeddings
                        }
                        }
                    )
                iteration += 1
        else:
             st.error("No articles without summary found.")
        cursor.close()

@st.dialog("Show Article")
def show_article_dialog(article_id: str) -> None:
    article = myapi.get_article(article_id)
    st.write(f"Title: {article['title']}")
    st.write(f"Date: {article['date']}")
    st.divider()
    st.write(f"Summary: {article['summary']}")
    st.divider()
    st.write(f"URL: {article['url']}")
    # time.sleep(3)
    # st.write(f"Content: {article['content']}")

@st.dialog("Export")
def export_dialog() -> None:
    exp_system_prompt = st.checkbox(f"System-Prompt [{st.session_state.system_prompt[:20]}...]", value=True)
    exp_user_prompt = st.checkbox(f"User-Prompt  [{st.session_state.prompt[:20]}...]", value=True)
    exp_results = st.checkbox(f"Results  [{st.session_state.results[:20]}...]", value=True)
    exp_response = st.checkbox(f"Response  [{st.session_state.response[:20]}...]", value=True)
    export_string = ""
    if exp_system_prompt:
        export_string += st.session_state.system_prompt + "\n\n"
    if exp_user_prompt:
        export_string += st.session_state.prompt + "\n\n"
    if exp_results:
        export_string += st.session_state.results + "\n\n"
    if exp_response:
        export_string += st.session_state.response + "\n\n"
    st.download_button(label="Start Download", data=export_string, file_name=f"export_{datetime.now().strftime('%Y-%m-%d')}.txt")

def generate_result_str(result: dict, url: bool = True, summary: bool = True) -> str:
    combined_result = f"\n\nDate: {str(result['date'])[:10]}\n\nTitle:{result['title']}"
    if url:
        combined_result += f"\n\nURL: {result['url']}"
    if result['keywords']:
        combined_result += f"\n\nKeywords: {result['keywords']}" 
    if summary:
        combined_result += f"\n\nSummary: {result['summary']}"
    return combined_result

def search_show_articles(query: str, max_items: int = 10):
    if query == "":
        st.error("Please enter a text for the search.")
        return
    results, query_input = myapi.text_search_artikel(search_text=query, limit=max_items, gen_schlagworte=True if count_words(query) > 3 else False)
    for result in results:
        st.write(generate_result_str(result=result, url=True, summary=False))
        st.write("---------------------------------------------------")

def keyword_search_show_articles(query: str, max_items: int = 10):
    if query == "":
        st.error("Please enter a keyword for the keyword search.")
        return
    results, query_input = myapi.keyword_search_artikel(search_text=query, limit=max_items)
    for result in results:
        st.write(generate_result_str(result=result, url=True, summary=False))
        st.write("---------------------------------------------------")

def remove_recurring_spaces(input_string: str) -> str:
    words = input_string.split()
    return ' '.join(words)

def count_words(input_string: str) -> int:
    words = input_string.split()
    return len(words)

# Main -----------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title='AI Update', initial_sidebar_state="expanded", layout="wide")
    # Initialize Session State -----------------------------------------
    if 'init' not in st.session_state:
        st.session_state.init = True
        st.session_state.search_status = False
        st.session_state.search_pref = "rag"
        st.session_state.llm_status = "GPT 4o mini"
        st.session_state.history = []
        st.session_state.search_type = "rag"
        # --------------------------------------------------------------
        st.session_state.system_prompt = remove_recurring_spaces(f"""
            You are a helpful assistant for tech news. Today is {TODAY}.
            Your task is to provide news in the field of artificial intelligence.
            When your answer refers to a specific article, please provide the URL.
            If the question consists only of two words, please provide a comprehensive dossier on the given topic.
            """)
        st.session_state.prompt = ""
        st.session_state.results = ""
        st.session_state.response = ""
        # --------------------------------------------------------------
        
    # Define Main Page ------------------------------------------------
    st.title("AI Insight")
    st.caption(f"Version {VERSION} Status: POC/{st.session_state.llm_status}")

    # Define Sidebar ---------------------------------------------------
    with st.sidebar:
        switch_search_type = st.radio(label="Choose Search Type", options=("rag", "vector", "fulltext", "keywords"), index=0)
        if switch_search_type != st.session_state.search_type:
            st.session_state.search_type = switch_search_type
            st.rerun()
        switch_llm = st.radio(label="Switch to LLM", options=("GPT 4o mini", "GPT 4o", "LLAMA 3.X"), index=0)
        if switch_llm != st.session_state.llm_status:
            st.session_state.llm_status = switch_llm
            st.rerun()
        switch_system_prompt = st.text_area("System-Prompt", st.session_state.system_prompt, height=200)
        if switch_system_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = switch_system_prompt
            st.rerun()
        if st.button("Export"):
            export_dialog()
        if st.button("UpdateDB"):
            update_db_dialog()
        if st.button("Logout"):
            st.session_state.user_status = False
            st.session_state.search_status = False

    # Define Search Form ----------------------------------------------
    with st.form(key="searchForm"):
        question = st.text_area(SEARCH_TYPES[st.session_state.search_type])
        if st.session_state.search_type in ["rag", "llm"]:
            button_caption = "Ask a question"
        else:
            button_caption = "Search"
        col = st.columns([0.25, 0.25, 0.25, 0.25])
        with col[0]:
            if st.form_submit_button(button_caption):
                st.session_state.search_status = True
        with col[1]:
            if st.form_submit_button("5-day summary"):
                st.session_state.search_type = "5day"
                st.session_state.search_status = True
        with col[2]:
            if st.form_submit_button("Latest articles"):
                st.session_state.search_type = "latest"
                st.session_state.search_status = True
        with col[3]:
            if st.form_submit_button("Show Keywords"):
                st.session_state.search_type = "keyword_rank"
                st.session_state.search_status = True

    # Define Search & Search Results -------------------------------------------
    if st.session_state.search_status:

        # RAG Search ---------------------------------------------------
        if st.session_state.search_type == "rag":
            results, query_input = myapi.text_search_artikel(search_text=question, limit=20, gen_schlagworte=True if count_words(question) > 3 else False)
            results_string = ""
            st.caption(f"Schlagworte: {query_input}")
            for result in results:
                results_string += generate_result_str(result=result, url=True, summary=True)
                results_string += "\n\n-----------------------------------"
            with st.expander("DB Search Results"):
                st.write(results_string)
            summary = myapi.ask_llm(llm=st.session_state.llm_status,
                                    question=question,
                                    history=st.session_state.history,
                                    system_prompt=st.session_state.system_prompt,
                                    results=results_string
                                    )
            st.write(summary)
            st.session_state.prompt = question
            st.session_state.results = results_string
            st.session_state.response = summary
            # st.session_state.search_pref = "rag"
            st.session_state.search_status = False

        # Fulltext Search -------------------------------------------------
        elif st.session_state.search_type == "fulltext":
            search_show_articles(query=question, max_items=10)
            st.session_state.search_pref = "rag"
            st.session_state.search_status = False

        # Vector Search ---------------------------------------------------
        elif st.session_state.search_type == "vector":
            results = myapi.vector_search_artikel(question, 10)
            st.session_state.search_status = False
            for result in results:
                st.write(generate_result_str(result=result, url=True, summary=True))
            st.session_state.search_status = False

        # Latest Articles -------------------------------------------------
        elif st.session_state.search_type == "latest":
            search_show_articles(query="*", max_items=10)
            st.session_state.search_status = False

        # Keyword search --------------------------------------------------
        elif st.session_state.search_type == "keywords":
            keyword_search_show_articles(query=question, max_items=10)
            st.session_state.search_status = False
        
        # Keywords Ranking-------------------------------------------------
        elif st.session_state.search_type == "keyword_rank":
            keywords_list = myapi.list_keywords()
            for keyword in keywords_list[:100]:
                st.write(f"{keyword['count']} {keyword['keyword']}")
            st.session_state.search_status = False

        # 3-day Summary ---------------------------------------------------
        elif st.session_state.search_type == "5day":
            st.write("5-day-summary")
            question = "Give a comprehensive summary of all the new developments by grouping into sections and summarize per section."
            results, schlagworte = myapi.text_search_artikel(search_text="*", limit=50, last_days=5)
            with st.expander("Latest Articles:"):
                results_string = ""
                for result in results:
                    st.write(generate_result_str(result=result, url=True, summary=False))
                    results_string += generate_result_str(result=result, url=True, summary=True)
                    results_string += "\n\n-----------------------------------"
            summary = myapi.ask_llm(llm=st.session_state.llm_status,
                                    question=question,
                                    history=st.session_state.history,
                                    system_prompt=st.session_state.system_prompt,
                                    results=results_string)
            st.write(summary)
            st.session_state.prompt = question
            st.session_state.results = results_string
            st.session_state.response = summary
            st.session_state.search_status = False

if __name__ == "__main__":
    main()
