import streamlit as st

from langchain_cohere import ChatCohere
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import time

load_dotenv()


# helper functions


def do_stream(text: str):
    for x in text.split(' '):
        yield x

st.set_page_config(page_title="911 for Ananya")




    # setup genai variables


if 'chain' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.store=[]

    st.session_state.QA_PROMPT = PromptTemplate.from_template('''
                                                            You are an assistant trained to answer ananya's question. Converse with her in a short precise way.
                                                            You task is to encourage ananya. Reply should not exceed 100 words
                                                            Previous conversation history: {history}
                                                            Some facts about Ananya:
                                                                1. She is beautiful
                                                                2. She is a advocate in making
                                                                3. She likes horror movies and sunflower
                                                                4. She likes travelling
                                                                5. She likes cooking
                                                            Facts about Sayan:
                                                                1. He likes guitar, movies and exploring cities
                                                                2. He likes cooking. His one of the favorites foods are pasta, biriyani, likes italian, indian, chinese cuisine
                                                                3. He likes travelling
                                                                4. He likes Ananya
                                                            Question: {input}
                                                            ''')
    st.session_state.llm = ChatCohere(temperature=0, streaming=True)
    st.session_state.output_parser = StrOutputParser()
    st.session_state.chain = ( {'input': lambda x: x['input'], 'history': lambda x: x['history']} | st.session_state.QA_PROMPT | st.session_state.llm | st.session_state.output_parser )

st.header("911 for Ananya")
    # Display chat messages from history on app rerun
msg_type_mapper = {'ai': 'assistant', 'human': 'user'}

# default text from bot
st.chat_message('assistant').write(f"Hi Ananya, how can I help you today ?")

print(st.session_state.memory.chat_memory.messages)
for message in st.session_state.memory.chat_memory.messages:
    
    st.chat_message(msg_type_mapper[message.type]).write(message.content)
    



# React to user input
if prompt := st.chat_input("What is up?"):
   
    with st.chat_message("user"):
        st.markdown(prompt)

    response = st.session_state.chain.invoke({'input': prompt, 'history': st.session_state.memory.chat_memory.messages})

    st.session_state.memory.save_context({'inputs':prompt}, {'outputs':response})

    with st.chat_message('assistant'):

        response_placeholder = st.empty()
        full_response = ''

        for chunk in do_stream(response):

            full_response += ( chunk + ' ')
            response_placeholder.markdown( full_response)
            time.sleep(0.05)


        

