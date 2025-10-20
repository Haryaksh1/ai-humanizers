# app.py
import streamlit as st
# try to import your humanizers. Adjust names if you used different function names.
from humanizer_aggressive import humanize as humanize_aggressive
from humanizer_balanced import humanize as humanize_balanced

st.set_page_config(page_title="AI Text Humanizers", layout="centered")

st.title("AI Text Humanizers")
st.write("Choose a humanizer, paste your text, and get the humanized output. (Your code remains private.)")

st.sidebar.header("Choose a Humanizer")
choice = st.sidebar.radio(
    "Which humanizer would you like?", 
    ("Balanced — safer & more grammatical", "Aggressive — stronger rephrasing")
)

if choice.startswith("Balanced"):
    st.sidebar.write("Balanced: preserves meaning, prioritizes grammar and readability.")
else:
    st.sidebar.write("Aggressive: stronger rewrite, more creative — may alter phrasing more.")

st.header(choice)
input_text = st.text_area("Paste or type the text to humanize:", height=250)

if st.button("Humanize"):
    if not input_text.strip():
        st.warning("Please enter some text to humanize.")
    else:
        with st.spinner("Humanizing..."):
            # call the correct function
            if choice.startswith("Balanced"):
                output = humanize_balanced(input_text)
            else:
                output = humanize_aggressive(input_text)
        st.subheader("Humanized text")
        st.write(output)
