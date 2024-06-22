# ---------------------------------------------------
# Version: 21.06.2024
# Author: M. Weber
# ---------------------------------------------------
# 05.06.2024 added searchFilter in st.session_state and sidebar
# 07.06.2024 implemented rag with fulltext search
# 09.06.2024 activated user management. Statistiken implemented.
# 15.06.2024 added filter for textsearch
# 16.06.2024 switched rag to vector search
# 16.06.2024 added Markbereiche as search filter
# 16.06.2023 added user role. sidebar only for admin
# 21.06.2024 added switch between fulltext and vector search for rag
# ---------------------------------------------------

import os
import streamlit as st
import chatdvv_module as myapi
import user_management

SEARCH_TYPES = ("rag", "llm", "vektor", "volltext")
MARKTBEREICHE = ("Alle", "Logistik", "Maritim", "Rail", "ÖPNV")
PUB_LOG = ("THB", "DVZ", "DVZT", "THBT", "DVZMG", "DVZM", "DVZ-Brief")
PUB_MAR = ("THB", "THBT", "SHF", "SHIOF", "SPI", "NSH")
PUB_RAIL =("EI", "SD", "BM", "BAMA")
PUB_OEPNV = ("RABUS", "NAHV", "NANA", "DNV")

# Functions -------------------------------------------------------------

@st.experimental_dialog("Login User")
def login_user_dialog() -> None:
    with st.form(key="loginForm"):
        st.write(f"Status: {st.session_state.userStatus}")
        user_name = st.text_input("Benutzer")
        user_pw = st.text_input("Passwort", type="password")
        if st.form_submit_button("Login"):
            if user_name and user_pw:
                active_user = user_management.check_user(user_name, user_pw)
                if active_user:
                    st.session_state.userName = active_user["username"]
                    st.session_state.userRole = active_user["rolle"]
                    st.session_state.userStatus = 'True'
                    st.rerun()
                else:
                    st.error("User not found.")
            else:
                st.error("Please fill in all fields.")


@st.experimental_dialog("Statistiken")
def statistiken_dialog() -> None:
    st.write(f"Anzahl Artikel: {myapi.collection.count_documents({})}")
    st.write(f"Anzahl Artikel ohne Abstract: {myapi.collection.count_documents({'ki_abstract': ''})}")
    st.write(f"Anzahl Artikel ohne Embeddings: {myapi.collection.count_documents({'embeddings': {}})}")
#     st.write("-"*50)
#     st.write(myapi.group_by_field())
#     st.write(myapi.list_fields())
    if st.button("Close"):
        st.rerun()


# Main -----------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title='DVV Insight', initial_sidebar_state="collapsed")
    
    # Initialize Session State -----------------------------------------
    if 'userStatus' not in st.session_state:
        st.session_state.feldListe: list = list(myapi.group_by_field().keys())
        st.session_state.history: list = []
        st.session_state.llmStatus: str = myapi.LLMS[0]
        st.session_state.marktbereich: str = "Alle"
        st.session_state.marktbereichIndex: int = 0
        st.session_state.rag_index: str = "fulltext"
        st.session_state.results: str = ""
        st.session_state.searchFilter: list = st.session_state.feldListe
        st.session_state.searchPref: str = "Artikel"
        st.session_state.searchResultsLimit:int  = 50
        st.session_state.searchStatus: bool = False
        st.session_state.searchType: str = "rag"
        st.session_state.searchTypeIndex: int  = SEARCH_TYPES.index(st.session_state.searchType)
        st.session_state.systemPrompt: str = myapi.get_systemprompt()
        st.session_state.userName: str = ""
        st.session_state.userRole: str = ""
        st.session_state.userStatus: bool = False
        
        
    if st.session_state.userStatus == False:
        login_user_dialog()
    st.header("DVV Insight")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Version 0.2.1 - 22.06.2024")
    with col2:
        if st.session_state.userStatus:
            st.write(f"Eingeloggt als: {st.session_state.userName}")
        else:
            st.write("Nicht eingeloggt.")

    # Define Sidebar ---------------------------------------------------
    if st.session_state.userRole == "admin":
        with st.sidebar:
            switch_searchFilter = st.multiselect(label="Choose Publications", options=st.session_state.feldListe, default=st.session_state.searchFilter)
            if switch_searchFilter != st.session_state.searchFilter:
                st.session_state.searchFilter = switch_searchFilter
                st.experimental_rerun()
            if st.button("Reset Filter"):
                st.session_state.searchFilter = st.session_state.feldListe
                st.session_state.marktbereich = "Alle"
                st.session_state.marktbereichIndex = 0
                st.experimental_rerun()
            switch_search_results = st.slider("Search Results", 1, 100, st.session_state.searchResultsLimit)
            if switch_search_results != st.session_state.searchResultsLimit:
                st.session_state.searchResultsLimit = switch_search_results
                st.experimental_rerun()
            switch_llm = st.radio(label="Switch LLM", options=myapi.LLMS, index=0)
            if switch_llm != st.session_state.llmStatus:
                st.session_state.llmStatus = switch_llm
                st.experimental_rerun()
            switch_rag_index = st.radio(label="Switch RAG-index", options=("fulltext", "vector"), index=0)
            if switch_rag_index != st.session_state.rag_index:
                st.session_state.rag_index = switch_rag_index
                st.experimental_rerun()
            switch_SystemPrompt = st.text_area("System-Prompt", st.session_state.systemPrompt)
            if switch_SystemPrompt != st.session_state.systemPrompt:
                st.session_state.systemPrompt = switch_SystemPrompt
                myapi.update_systemprompt(switch_SystemPrompt)
                st.experimental_rerun()
            if st.button("Statistiken"):
                statistiken_dialog()
            if st.button("Logout"):
                st.session_state.userStatus = False
                st.session_state.searchStatus = False
                st.session_state.userName = ""
                st.experimental_rerun()

    # Define Search Type ------------------------------------------------
    switch_searchType = st.radio(label="Auswahl Suchtyp", options=SEARCH_TYPES, index=st.session_state.searchTypeIndex, horizontal=True)
    if switch_searchType != st.session_state.searchType:
        st.session_state.searchType = switch_searchType
        st.session_state.searchTypeIndex = SEARCH_TYPES.index(switch_searchType)
        st.experimental_rerun()
    
    # Define Search Filter ----------------------------------------------
    switch_marktbereich = st.radio(label="Auswahl Marktbereich", options=MARKTBEREICHE, index=st.session_state.marktbereichIndex, horizontal=True)
    if switch_marktbereich != st.session_state.marktbereich:
        if switch_marktbereich == "Logistik":
            st.session_state.searchFilter = PUB_LOG
        elif switch_marktbereich == "Maritim":
            st.session_state.searchFilter = PUB_MAR
        elif switch_marktbereich == "Rail":
            st.session_state.searchFilter = PUB_RAIL
        elif switch_marktbereich == "ÖPNV":
            st.session_state.searchFilter = PUB_OEPNV
        elif switch_marktbereich == "Alle":
            st.session_state.searchFilter = st.session_state.feldListe
        st.session_state.marktbereich = switch_marktbereich
        st.session_state.marktbereichIndex = MARKTBEREICHE.index(switch_marktbereich)
        st.experimental_rerun()

    # Define Search Form ----------------------------------------------
    with st.form(key="searchForm"):
        question = st.text_input(f"{st.session_state.searchType} [{st.session_state.rag_index}]")
        if st.session_state.searchType in ["rag", "llm"]:
            button_caption = "Fragen"
        else:
            button_caption = "Suchen"
        if st.form_submit_button(button_caption) and question != "":
            st.session_state.searchStatus = True
        
    # Define Search & Search Results -------------------------------------------
    if st.session_state.userStatus and st.session_state.searchStatus:
        st.warning(f'Suche in: {st.session_state.searchFilter}')
        # Fulltext Search -------------------------------------------------
        if st.session_state.searchType == "volltext":
            results, results_count = myapi.text_search(
                search_text=question, 
                filter=st.session_state.searchFilter, 
                limit=st.session_state.searchResultsLimit
                )
            counter = 1
            for result in results:
                # st.write(f"[{result['datum']}] {result['titel']}")
                st.write(f"[{result['quelle_id']}, {result['nummer']}/{result['jahrgang']}] {result['titel']}")
                st.write(result['text'][:500] + " ...")
                st.divider()
                counter += 1
                if counter > st.session_state.searchResultsLimit:
                    break
        # Vector Search --------------------------------------------------        
        elif st.session_state.searchType == "vektor":
            results = myapi.vector_search(
                query_string=question, 
                limit=st.session_state.searchResultsLimit
                )
            for result in results:
                # st.write(f"[{result['datum']}] {result['titel']}")
                st.write(f"[{result['quelle_id']}, {result['nummer']}/{result['jahrgang']}] {result['titel']}")
                st.write(result['text'])
                st.divider()
        # LLM Search -----------------------------------------------------
        elif st.session_state.searchType == "llm":
            summary = myapi.ask_llm(
                llm=st.session_state.llmStatus,
                temperature=0.2,
                question=question,
                history=[],
                systemPrompt=st.session_state.systemPrompt,
                results_str=""
                )
            st.write(summary)
        # RAG Search -----------------------------------------------------
        elif st.session_state.searchType == "rag":
            if st.session_state.rag_index == "vector":
                results = myapi.vector_search(
                query_string=question, 
                # filter=st.session_state.searchFilter, 
                limit=st.session_state.searchResultsLimit
                )
            else:
                results, results_count = myapi.text_search(
                    search_text=question, 
                    filter=st.session_state.searchFilter, 
                    limit=st.session_state.searchResultsLimit
                    )
            with st.expander("DB Suchergebnisse"):
                results_str = ""
                for result in results:
                    st.write(f"[{result['quelle_id']}, {result['nummer']}/{result['jahrgang']}] {result['titel']}")
                    results_str += f"Datum: {result['datum']}\nTitel: {result['titel']}\nText: {result['text']}\n\n"
            summary = myapi.ask_llm(
                llm=st.session_state.llmStatus,
                temperature=0.2,
                question=question,
                history=[],
                systemPrompt=st.session_state.systemPrompt,
                results_str=results_str
                )
            st.write(summary)
        st.session_state.searchStatus = False


if __name__ == "__main__":
    main()
