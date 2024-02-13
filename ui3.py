import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")

# Define the query function
def query(prompt):
    headers = {
        "customer-id": f"{VECTARA_CUSTOMER_ID}",
        "x-api-key": f"{VECTARA_API_KEY}",
    }

    body = {
        "query": [
            {
                "query": prompt,
                "start": 0,
                "numResults": 10,
                "corpusKey": [
                    {
                        "customerId": VECTARA_CUSTOMER_ID,
                        "corpusId": 1,
                        "semantics": "DEFAULT",
                        "lexicalInterpolationConfig": {
                            "lambda": 0
                        }
                    }
                ],
                "summary": [
                    {
                        "summarizerPromptName": "vectara-summary-ext-v1.2.0",
                        "maxSummarizedResults": 5,
                        "responseLang": "en"
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            "https://api.vectara.io/v1/query",
            json=body,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        parsed_response = response.json()
        return parsed_response["responseSet"][0], False
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred in Vectara: {e}")
        return "An error occurred while generating a response to your question.", True

# Define the query_hhme_model function
def query_hhme_model(generated_response, references):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    body = [{"text": generated_response, "text_pair": reference} for reference in references]

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/vectara/hallucination_evaluation_model",
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        parsed_response = response.json()
        return [obj[0]["score"] for obj in parsed_response]
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return []

# App title and markdown
st.title("Parent Quest AI")
st.markdown("🚀 Your Guide to Parenting")

# Initialize or get the conversation messages from session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing messages
for msg in st.session_state["messages"]:
    st.text_area("Conversation", value=msg["content"], height=300, disabled=True)

# Input for new message with key 'query_input'
prompt = st.text_input("Ask a question", key='query_input', on_change=None)

# Function to handle the submission
def handle_submit():
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    answer, errored = query(prompt)
    if errored:
        st.session_state["messages"].append({"role": "assistant", "content": answer})
    else:
        summary = answer["summary"][0]["text"]
        raw_references = [reference["text"] for reference in answer["response"]]
        hhme_scores = query_hhme_model(summary, raw_references)

        references = []
        if hhme_scores:
            for i, (reference, score) in enumerate(zip(answer["response"], hhme_scores)):
                text = f"Reference {i+1} - {score:.2%} chance of being factually consistent with the generated response: {reference['text']}"
                references.append(text)
        else:
            for i, reference in enumerate(answer["response"]):
                text = f"Reference {i+1}: {reference['text']}"
                references.append(text)

        references_joined = "\n\n".join(references)
        full_response = f"{summary}\n\nReferences:\n\n{references_joined}"
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
    
    # Clear the input box after submission
    # st.session_state.query_input = ""
    
    # Rerun the app to refresh the state and display messages
    st.experimental_rerun()

# Check if 'Submit' button is pressed
if st.button("Submit"):
    handle_submit();

