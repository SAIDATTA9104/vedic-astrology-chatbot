import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import hashlib
import yaml

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

st.title('🔮 Vedic Astrology AI Chatbot - Powered by Parashara & Gemini')

# Simple auth for demo
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

username = st.sidebar.text_input('Username', 'demo')
password = st.sidebar.text_input('Password', type='password')

if st.sidebar.button('Login'):
    if username and password:  # Add real check
        st.session_state.logged_in = True
        st.success('Logged in as ' + username)

if st.session_state.logged_in:
    st.sidebar.success('Welcome!')

    uploaded = st.file_uploader('Upload Parashara Report (PDF)', type=['pdf'])
    if uploaded:
        if st.button('Save Chart'):
            os.makedirs('charts', exist_ok=True)
            path = f'charts/{uploaded.name}'
            with open(path, 'wb') as f:
                f.write(uploaded.getbuffer())
            st.session_state.chart_path = path
            st.success('Chart saved!')

    if 'chart_path' in st.session_state and st.session_state.chart_path:
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg['role']).write(msg['content'])

        prompt = st.chat_input('Ask astrology question...')
        if prompt:
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.chat_message('user').write(prompt)

            with st.chat_message('assistant'):
                with st.spinner('Analyzing...'):
                    try:
                        file = client.files.upload(file=st.session_state.chart_path)
                        response = client.models.generate_content(
                            model='gemini-1.5-pro',
                            contents=[f'You are a professional Vedic astrologer. Base ALL answers on the uploaded Parashara report ONLY. No hallucinations.\n\nQuestion: {prompt}', file]
                        )
                        answer = response.text
                        st.write(answer)
                        st.session_state.messages.append({'role': 'assistant', 'content': answer})
                    except Exception as e:
                        st.error(str(e))
else:
    st.warning('Please login to continue.')
