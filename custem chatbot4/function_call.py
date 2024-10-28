from prompts import FUNC_CALL_PROMPT


class FuncCall:
    def __init__(self):

        self.API_URL = "http://nexusraven.nexusflow.ai"
        self.headers = {
            "Content-Type": "application/json"
        }

        self.prompt = FUNC_CALL_PROMPT

    def __query(self, payload):
        """
        Sends a payload to a TGI endpoint.
        """
        import requests
        response = requests.post(
            self.API_URL, headers=self.headers, json=payload)
        return response.json()

    def query_raven(self, user_query, past_conversation):
        """
        This function sends a request to the TGI endpoint to get Raven's function call.
        This will not generate Raven's justification and reasoning for the call, to save on latency.
        """

        msgs_len = len(past_conversation)

        conversation = ''
        if msgs_len == 0:
            conversation = 'No Past Conversation.'
        elif msgs_len == 1:
            conversation = f'Role: {past_conversation[0]["role"]} \n message: {past_conversation[0]["content"]}'
        else:
            from_msg_num = -min(3, msgs_len)
            conversation = [
                f'Role: {msg["role"]} \nmessage: \n"{msg["content"]}"\n'
                for msg in past_conversation[from_msg_num:] if msg['role'] != 'user']
            conversation = '\n'.join(conversation)

        inputs_ = self.prompt.format(
            query=user_query,
            conversation=conversation
        )

        output = self.__query(
            {
                "inputs": inputs_,
                "parameters":
                {
                    "temperature": 0.01,
                    "stop": ["<bot_end>"],
                    "do_sample": False,
                    "max_new_tokens": 2000,
                    "return_full_text": False
                }
            }
        )
        call = output[0]["generated_text"].replace("Call:", "").strip()
        return call


# QUESTION = "hi there"
# give_response_to_user(user_prompt='hi there')


# QUESTION = "what products are loans with interest rates of 10%?"
# filter_products(user_input_description='loans with interest rates of 10%')
# QUESTION = "recommend to me based on my historical behavior."
# recommend_product(user_id=get_user_id())
# QUESTION = "what of this products good for my business?"
# QUESTION = "hi there"

# funcall = FuncCall()
# raven_call = funcall.query_raven(QUESTION)

# print(f"Raven's Call: {raven_call}")
