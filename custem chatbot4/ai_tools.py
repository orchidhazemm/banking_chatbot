from ai_response_agent import (
    get_response,
    recommended_types_response,
    add_msg
)
import streamlit as st
from custom_rec_sys import CustomRecSys
from branch_locations.nearest_branches import get_nearby_banque_branches
from gen_tools import *


def is_question_answered(user_prompt: str) -> bool:

    clean_user_prompt = clean_text(user_prompt)
    for msg in st.session_state.messages[:-1]:

        clean_big_msg = clean_text(msg['content'])
        if advanced_string_search(clean_user_prompt, clean_big_msg):
            return True

    return False


user_id = 558398


def get_user_id(): return user_id


def recommend_product_based_on_historical_behavior(user_id: int):
    """
    Recommend a product based on user historical behavior.

    Args:
        user_id (int): The ID of the user for whom to recommend a product.

    Returns:
        The response to the user's prompt, based on the recommended product.
    """

    if is_question_answered(user_prompt=st.session_state.user_prompt):
        print("The question has been answered before")
        return get_response(st.session_state.user_prompt, rag_req=False)

    if "rec_sys" not in st.session_state:
        print(" -- rec_sys not found in session state. Initializing...")
        st.session_state.rec_sys = CustomRecSys("qty")
        print(" - Initialization complete.\n")

    rec_prod_types = st.session_state.rec_sys.recommend_products_types(
        get_user_id()
    )

    print(rec_prod_types)
    recommended_types_response(rec_prod_types)
    return get_response(st.session_state.user_prompt, rag_req=False)


def fetch_and_filter_data_from_chroma_collection(user_prompt: str | None = None):

    if is_question_answered(user_prompt=st.session_state.user_prompt):

        print("The question has been answered before")
        return get_response(st.session_state.user_prompt, rag_req=False)

    return get_response(st.session_state.user_prompt, rag_req=True)


def give_normal_response(user_prompt: str | None = None):
    return get_response(st.session_state.user_prompt, rag_req=False)


def get_the_nearest_banque_branches(**kwargs):

    # if is_question_answered(user_prompt=st.session_state.user_prompt):

    #     print("The question has been answered before")
    #     return get_response(st.session_state.user_prompt, rag_req=False)

    nearest_branches = get_nearby_banque_branches()

    branch_prompt = \
        f"""
        Take this banque misr branches that are top 3 nearest if the user asks for the them:
        {nearest_branches}
        """

    add_msg('system', branch_prompt)

    return get_response(
        st.session_state.user_prompt,
        rag_req=False
    )
