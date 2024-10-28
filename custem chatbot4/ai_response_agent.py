from groq import Groq
import streamlit as st
import time

from gen_tools import get_arabic_markdown, get_current_time
from gen_tools import translate

client = Groq(
    api_key="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"
)


# model_name = "llama-3.1-70b-versatile"
# model_name = "llama3-70b-8192"
# model_name = "llama3-8b-8192"
model_name = "llama-3.1-8b-instant"
# model_name = "mixtral-8x7b-32768"


def add_msg(role: str, content: str):
    time_ = get_current_time()
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "time": time_,
        }
    )
    return time_


def add_translated_msg(role: str, content: str, time_: str):
    st.session_state.translated_messages.append(
        {
            "role": role,
            "content": content,
            "time": time_,
        }
    )


def get_messages():
    return [
        {"role": m['role'], "content": m['content']}
        for m in st.session_state.messages
    ]


def get_response(
    prompt: str,
    role: str = 'user',
    rag_req: bool = False
):

    if rag_req:
        rag_response(prompt)

    chat_completion = client.chat.completions.create(
        messages=get_messages(),  # type: ignore
        model=model_name,
        temperature=0.1,
    )

    response = chat_completion.choices[0].message.content or "No response"

    time_ = add_msg('assistant', response)

    if st.session_state.lang == 'ar':
        response = translate(response, 'ar')
        add_translated_msg("assistant", response, time_)

    response = \
        f"""{time_}
        \r{response}
        """

    return response


def stream_ai_response(response: str, delay=0.001):

    placeholder = st.empty()
    content = ""

    for word in response:
        content += word
        placeholder.markdown(content)
        time.sleep(delay)


def stream_ai_ar_response(response: str, delay=0.001):

    response = get_arabic_markdown(response)
    placeholder = st.empty()
    content = ""

    for word in response:
        content += word
        placeholder.markdown(content, unsafe_allow_html=True)
        time.sleep(delay)


def rag_response(user_prompt: str):

    res_docs = st.session_state.infoRetrieval.query(user_prompt)

    res_docs_text = ''
    for i, doc in enumerate(res_docs):
        res_docs_text += f'DOC({i}):\n{doc} \n\n'

    system_prompt = f"""
    Based on the following documents:
    {res_docs_text} 
    
    NOTE: Dont Mention the DOC Number, this is just an example of separated DOC.
    you can answer the following question if user asks you: 
    Question: "{user_prompt}". 
    """
    print(system_prompt)
    add_msg('system',  system_prompt)


def recommended_types_response(recommended_types: list[str]):

    rec_types = ', '.join(recommended_types)
    system_prompt = f"""
    After Making Some Operations 
    we found this products types are the most product types recommended to the user based on his transaction history in our bank:
    product types: {rec_types}
    
    if user asks you to recommend products based on these type:
    make him to specify which type you want to recommend and its description.
    but show the results of the recommended / recommend types to the user.
    """
    add_msg('system',  system_prompt)
