import os
import requests
import streamlit as st

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")

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
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code.
        parsed_response = response.json()
        return parsed_response["responseSet"][0], False
    except requests.exceptions.RequestException as e:
        # Log error, could be a connection error, timeout, etc.
        st.error(f"An error occurred in Vectara: {e}")
        return "An error occurred while generating a response to your question.", True

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

st.title("Parent Quest AI")
st.markdown("🚀 Your Guide to Parenting")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.text(msg["content"])

prompt = st.text_input("Ask a question")
if st.button("Submit"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer, errored = query(prompt)
    if errored:
        st.session_state.messages.append({"role": "assistant", "content": answer})
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
            references = [f"Reference {i+1}: {reference['text']}" for i, reference in enumerate(answer["response"])]

        references_joined = "\n\n".join(references)
        full_response = f"{summary}\n\nReferences:\n\n{references_joined}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
