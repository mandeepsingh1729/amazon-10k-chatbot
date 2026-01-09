import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import AmazonChatBot

def main():
    """Streamlit web interface"""
    st.set_page_config(
        page_title="Amazon 10-K ChatBot",
        
        layout="wide"
    )
    
    st.title(" Amazon 10-K Assistant")
    st.markdown("""
    Ask questions about **Amazon's 2022 Annual Report (10-K)**.
    The chatbot uses RAG (Retrieval-Augmented Generation) to find answers in the document.
    """)
    
    with st.sidebar:
        st.header("ℹ About")
        st.markdown("""
        - **Document**: Amazon 2022 10-K Report
        - **Technology**: ChromaDB + Sentence Transformers + GPT-3.5
        - **Source**: Public SEC filing
        """)
        
        st.header("⚙️ Settings")
        num_chunks = st.slider("Number of chunks to retrieve", 1, 10, 3)
        
        if st.button("Clear Chat History", type="secondary"):
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.clear_history()
                st.success("History cleared!")
    
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = AmazonChatBot()
            st.success("ChatBot initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.info("Make sure you've run the setup scripts first.")
            return
    
    st.header(" Chat")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What was Amazon's revenue in 2022?",
        key="question_input"
    )
    
    if question:
        with st.spinner("Searching document and generating answer..."):
            try:
                result = st.session_state.chatbot.answer(question, k=num_chunks)
                
                st.success("Answer found!")
                st.markdown("### Answer:")
                st.write(result['answer'])
                
                if result['sources']:
                    st.markdown(f"**Sources:** Pages {', '.join(result['sources'])}")
                
                with st.expander(" View retrieved context"):
                    search_results = st.session_state.chatbot.search(question, k=num_chunks)
                    
                    for i, (doc, meta) in enumerate(zip(
                        search_results['documents'][0],
                        search_results['metadatas'][0]
                    )):
                        st.markdown(f"**Context {i+1}** (Page {meta.get('page', '?')}):")
                        st.text(doc[:300] + "...")
                        st.divider()
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.header(" Chat History")
    
    if st.session_state.chatbot.chat_history:
        for i, (q, a) in enumerate(reversed(st.session_state.chatbot.chat_history[-10:])):
            with st.expander(f"Q: {q[:50]}..." if len(q) > 50 else f"Q: {q}"):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")
    else:
        st.info("No chat history yet. Ask a question to get started!")

if __name__ == "__main__":
    main()