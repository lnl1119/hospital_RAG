import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

openai_api_key = st.secrets["OPENAI_API_KEY"]

def RAG(input_question):
    llm_model = ChatOpenAI(
        openai_api_key = openai_api_key,
        model="gpt-4o mini",
    )
    persist_directory = "/Users/leeliang/Desktop/hospital_RAG/db"

    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding
    )
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 1}
    )  # search_kwargs={"k": 3,"score_threshold": 0.6}

    #################################
    prompt_template = """You are an professional encouraging doctor who helps patients.
    Answer the questions using the facts provided. Use the following pieces of context to answer the users question.
    If you don't know the answer, just say "I don't know", don't try to make up answers.
    You will answer in mandarin only.
    {summaries} 
    """
    # Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
    #
    
    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    result = chain.invoke(input_question)
    

    return result["answer"]


assistant_logo = 'https://cdn.pixabay.com/photo/2021/11/09/05/44/doctor-6780685_1280.png'
st.set_page_config(
    page_title="Animals in Research",
    page_icon=assistant_logo
)
st.title("台北市立聯合醫院仁愛院區影像醫學科 一般X光攝影檢查機器人")


if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": """歡迎來到台北市立聯合醫院仁愛院區影像醫學科進行一般X光攝影檢查。
                                                您可以詢問我關於檢查的流程，以及檢查的注意事項。"""}]

for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=assistant_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if query := st.chat_input("Ask me about animals in research"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = RAG(query)
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": response})