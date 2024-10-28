import streamlit as st
from ai_response_agent import (
    add_translated_msg,
    stream_ai_ar_response,
    stream_ai_response,
)
from prompts import INIT_PROMPT
from pre_data.info_retrieval import InfoRetrieval

from function_call import FuncCall
from ai_tools import *


st.title("Ai Sales Chatbot")


# st.session_state.lang = 'en'
# Add language selection dropdown
# st.session_state.lang = st.radio("Select Language", ['en', 'ar'], index=0)

# Initialize the language in session state if not already set
if "lang" not in st.session_state:
    st.session_state.lang = None

# Create a placeholder to hold the language selection buttons
lang_selection = st.empty()

# Show language selection buttons if no language is selected yet
if st.session_state.lang is None:
    with lang_selection.container():
        col1, col2 = st.columns(2)
        with col1:
            if st.button("English"):
                st.session_state.lang = 'en'
                lang_selection.empty()  # Clear the button container
        with col2:
            if st.button("Arabic"):
                st.session_state.lang = 'ar'
                lang_selection.empty()  # Clear the button container

if st.session_state.lang is not None:

    is_ar_lang = st.session_state.lang == 'ar'

    print('-'*40)

    print(" - Checking if messages are in session state...")

    if "messages" not in st.session_state:
        print(" -- Messages not found in session state. Initializing...")
        st.session_state.messages = [
            {
                "role": 'system',
                "content": INIT_PROMPT
            }
        ]
        print(" - Initialization complete.\n")

    if "translated_messages" not in st.session_state and is_ar_lang:
        st.session_state.translated_messages = []

    print(" - Checking if info retrieval model is in session state...")
    if "infoRetrieval" not in st.session_state:

        with st.spinner(text='Loading...'):
            print(" -- Loading info retrieval model...")
            st.session_state.infoRetrieval = InfoRetrieval()
            st.session_state.infoRetrieval.load_chroma_collection()
            print(" - Loading info retrieval model... Done\n")

    print(" - Checking if function call is in session state...")
    if "funcall" not in st.session_state:

        with st.spinner(text='Loading...'):
            print(" -- Loading funcall...")
            st.session_state.funcall = FuncCall()
            print(" - Loading funcall... Done\n")

    print(" - Streaming messages...")

    looping_msgs = st.session_state.translated_messages \
        if is_ar_lang \
        else st.session_state.messages

    for msg in looping_msgs:

        if msg['role'] == 'system':
            continue

        with st.chat_message(
            msg['role'],
            avatar="banque_logo.png"
            if msg["role"] == 'assistant' else "user",
        ):
            msg_content = \
                f"""{msg['time']}
                \r{msg['content']}
                """

            if is_ar_lang:
                msg_content = get_arabic_markdown(msg_content)

            st.markdown(msg_content, unsafe_allow_html=is_ar_lang)

    origin_prompt = None

    print(" - Checking if user input is available...")

    if prompt := st.chat_input("Say something!"):

        if is_ar_lang:
            origin_prompt = prompt
            prompt = translate(prompt, 'en')

        st.session_state.user_prompt = prompt

        with st.chat_message("user"):
            time_ = add_msg('user', prompt)

            if is_ar_lang:
                st.write(get_arabic_markdown(time_), unsafe_allow_html=True)

                origin_prompt = get_arabic_markdown(origin_prompt)
                add_translated_msg('user', origin_prompt or "", time_)
                st.markdown(origin_prompt, unsafe_allow_html=True)
            else:
                st.write(time_)
                st.markdown(prompt)

        with st.spinner(text='Thinking...'):

            print(" - Generating response...")

            res_func = st.session_state.funcall.query_raven(
                prompt,
                st.session_state.messages
            )

            print(f' -- call function = {res_func}')
            response = eval(res_func)
            print(" - Response generated.\n")

        with st.chat_message("assistant", avatar="banque_logo.png"):

            print(" - Streaming response...")
            if is_ar_lang:
                stream_ai_ar_response(response)
            else:
                stream_ai_response(response)

            print(" - Streamed response done.\n")
        print('-'*40)
