import os
import requests
import streamlit as st

HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
VECTARA_CUSTOMER_ID = os.environ.get("VECTARA_CUSTOMER_ID")
VECTARA_API_KEY = os.environ.get("VECTARA_API_KEY")

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

    response = requests.post(
        "https://api.vectara.io/v1/query",
        json=body,
        headers=headers,
        timeout=30,
    )

    if response.status_code != 200:
        print(response.status_code)
        print(response.json())

        return "An error occurred while generating a response to your question.", True
    
    parsed_response = response.json()
    return parsed_response["responseSet"][0], False

def query_hhme_model(generated_response, references):
    body = []
    for reference in references:
        body.append({
            "text": generated_response,
            "text_pair": reference
        })

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    response = requests.post(
        "https://api-inference.huggingface.co/models/vectara/hallucination_evaluation_model",
        headers=headers,
        json=body,
    )
    
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
        return []

    results = []
    parsed_response = response.json()
    for obj in parsed_response:
        results.append(obj[0]["score"])

    return results


st.title("Parent Quest AI")
st.markdown("🚀 Your Guide to Parenting")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.text(msg["content"])

if prompt := st.text_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer, errored = query(prompt)
    if errored:
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        summary = answer["summary"][0]["text"] 

        raw_references = []
        for i, reference in enumerate(answer["response"]):
            raw_references.append(reference["text"])
        hhme_scores = query_hhme_model(summary, raw_references)

        references = []
        if len(hhme_scores) == 0:
            for i, reference in enumerate(answer["response"]):
                text = f"Reference {i+1}: {reference['text']}"
                references.append(text)
                raw_references.append(reference["text"])
        else:
            for i, reference in enumerate(answer["response"]):
                text = f"Reference {i+1} - {hhme_scores[i]:.2%} chance of being factually consistent with the generated response: {reference['text']}"
                references.append(text)
                raw_references.append(reference["text"])

        references_joined = "\n\n".join(references)
        full_response = f"{summary}\n\nReferences:\n\n{references_joined}"

        st.session_state.messages.append({ "role": "assistant", "content": full_response })
