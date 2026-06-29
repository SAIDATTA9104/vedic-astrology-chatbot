import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
import hashlib

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

st.title('🔮 Vedic Astrology AI Chatbot - Powered by Parashara & Gemini')

# Simple auth for demo
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

username = st.sidebar.text_input('Username', 'demo')
if st.sidebar.button('Login'):
    st.session_state.logged_in = True
    st.success('Logged in as ' + username)

if st.session_state.logged_in:
    st.sidebar.success('Welcome!')

    uploaded = st.file_uploader('Upload Parashara Report (PDF)', type=['pdf', 'png', 'jpg', 'jpeg'])
    chart_name = st.text_input('Chart Name', 'ClientChart1')

    if uploaded and st.button('Process & Save Chart'):
        os.makedirs('charts', exist_ok=True)
        file_path = f'charts/{username}_{chart_name}.pdf'
        with open(file_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        st.session_state.current_chart = file_path
        st.success('Chart saved!')

    if 'current_chart' in st.session_state and st.session_state.current_chart:
        if 'messages' not in st.session_state:
            st.session_state.messages = [{'role': 'assistant', 'content': 'Upload complete. Ask me anything about the chart!'}]

        for msg in st.session_state.messages:
            st.chat_message(msg['role']).write(msg['content'])

        prompt = st.chat_input('Ask about dasha, career, remedies...')
        if prompt:
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.chat_message('user').write(prompt)

            with st.chat_message('assistant'):
                with st.spinner('Analyzing with Gemini...'):
                    try:
                        file_obj = client.files.upload(file=st.session_state.current_chart)
                        response = client.models.generate_content(
                            model='gemini-1.5-pro',
                            contents=[
                                f'''You are a professional Vedic astrologer. Base **every** answer strictly on the uploaded Parashara Light report. 
Cite planets, houses, dashas, yogas specifically. No hallucinations or generic advice.

Question: {prompt}''',
                                file_obj
                            ]
                        )
                        answer = response.text
                        st.write(answer)
                        st.session_state.messages.append({'role': 'assistant', 'content': answer})
                    except Exception as e:
                        st.error(f'Gemini Error: {str(e)}')
    else:
        st.info('Please upload a Parashara report to begin.')
else:
    st.warning('Login to access the chatbot.')