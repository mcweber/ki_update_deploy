# ---------------------------------------------------
# Version: 02.06.2024
# Author: M. Weber
# ---------------------------------------------------

import os
import streamlit as st
import ai_update_module as myapi
import user_management as um

SEARCH_TYPES = {"vector": "Vector search", "llm": "LLM search",
                "rag": "RAG search", "fulltext": "Fulltext search"}

# Functions -------------------------------------------------------------

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


# Main -----------------------------------------------------------------

def main() -> None:
    st.title("AI Update")
    st.write("Version 0.1.2 02.06.2024")

    # Initialize Session State -----------------------------------------
    if 'userStatus' not in st.session_state:
        st.session_state.userStatus = True
        st.session_state.searchStatus = False
        st.session_state.searchPref = "rag"
        st.session_state.llmStatus = "openai"
        st.session_state.systemPrompt = "You are a helpful assistant for tech news."
        st.session_state.results = ""
        st.session_state.history = []
        st.session_state.searchType = "rag" # llm, vector, rag, fulltext
        
    if not st.session_state.userStatus:
        login_user_dialog()

    # Define Sidebar ---------------------------------------------------
    with st.sidebar:
        
        switch_searchType = st.radio(label="Choose Search Type", options=("rag", "llm", "vector", "fulltext"), index=0)
        if switch_searchType != st.session_state.searchType:
            st.session_state.searchType = switch_searchType
            st.experimental_rerun()

        switch_llm = st.radio(label="Switch to LLM", options=("groq", "openai"), index=0)
        if switch_llm != st.session_state.llmStatus:
            st.session_state.llmStatus = switch_llm
            st.experimental_rerun()

        switch_SystemPrompt = st.text_area("System-Prompt", st.session_state.systemPrompt)
        if switch_SystemPrompt != st.session_state.systemPrompt:
            st.session_state.systemPrompt = switch_SystemPrompt
            st.experimental_rerun()

        if st.button("Logout"):
            st.session_state.userStatus = False
            st.session_state.searchStatus = False

    # Define Search Form ----------------------------------------------
    with st.form(key="searchForm"):
        question = st.text_input(SEARCH_TYPES[st.session_state.searchType])
        if st.session_state.searchType in ["rag", "llm"]:
            button_caption = "Ask a question"
        else:
            button_caption = "Search"
        if st.form_submit_button(button_caption):
            st.session_state.searchStatus = True

    # Define Search & Search Results -------------------------------------------
    if st.session_state.userStatus and st.session_state.searchStatus:

        if st.session_state.searchType == "fulltext":

            results, count = myapi.text_search_artikel(question)

            st.caption(f'Search for: "{question}". {count} articles found.')
            st.session_state.searchStatus = False

            for result in results[:10]:
                st.write(f"[{str(result['date'])[:10]}] {result['title']}")
                st.write(result['summary'])
                st.write(f"URL: {result['url']}")
                st.divider()
        
        elif st.session_state.searchType == "llm":

            summary = myapi.ask_llm(llm=st.session_state.llmStatus, question=question, 
                                    history=st.session_state.history, 
                                    systemPrompt=st.session_state.systemPrompt, results="")
            st.write(summary)
            st.session_state.searchStatus = False

        elif st.session_state.searchType == "rag":

            results = myapi.vector_search_artikel(question, 10)
            
            with st.expander("DB Search Results"):
                results_string = ""
                for result in results:
                    st.write(f"[{str(result['date'])[:10]}] {result['title'][:70] + '...'}")
                    results_string += f"Date: {str(result['date'])}\nSummary: {result['summary']}\n\n"

            summary = myapi.ask_llm(llm=st.session_state.llmStatus, question=question, 
                                    history=st.session_state.history, 
                                    systemPrompt=st.session_state.systemPrompt, results=results_string)
            st.write(summary)
            st.session_state.searchStatus = False

        elif st.session_state.searchType == "vector":
            
            results = myapi.vector_search_artikel(question, 10)
            st.session_state.searchStatus = False

            for result in results:
                st.write(f"[{str(result['date'])[:10]}] {result['title']}")
                st.write(result['summary'])
                st.write(f"URL: {result['url']}")
                st.divider()
            
            st.session_state.searchStatus = False
   
   
if __name__ == "__main__":
    main()