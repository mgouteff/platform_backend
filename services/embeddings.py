# services/embeddings.py

import os
import re
from openai import OpenAI

# Load API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
else:
    print("✅ DEBUG: OpenAI API Key starts with:", OPENAI_API_KEY[:8])

query_expansion_instructions = """
    **Purpose:**
    The model’s job is **not** to answer the question directly. Instead, it transforms the user’s input into a **search query optimized for your Czech labor law knowledge base**.
    ### **Behavior Rules**
    1. **Input:**

       * A user question in any language.
       * Example: `"Hi! what are the norms for work of pregnant women?"`

    2. **Output:**
       * A concise **Czech query** suitable for semantic search in your knowledge base.
       * Must use terminology likely present in Czech labor law texts.
       * Should avoid full sentences or explanations — just keywords or short phrases are preferred.
    3. **Requirements:**
       * Always produce **Czech text**, regardless of the input language.
       * Preserve the **legal meaning** of the question.
       * Avoid adding irrelevant information.
       * Avoid polite greetings, commentary, or apologies — output only the query.
    ---
    ### **Formatting**
    * **Plain text only** — no JSON, markdown, or extra symbols.
    * Example input → output:
    | User Question                                        | Expanded Query (Czech)                                |
    | ---------------------------------------------------- | ----------------------------------------------------- |
    | “Hi! what are the norms for work of pregnant women?” | “práce těhotné ženy ochrana zdraví pracovní podmínky” |
    | “How many vacation days does an employee get?”       | “dovolená pracovník počet dní zákoník práce”          |
    ---
    ### **Optional Notes for GPT**
    * If the user question contains multiple sub-questions, merge them into a **single concise query**.
    * Use **official legal terminology** wherever possible.
    * Focus only on **Czech labor law context** — ignore other countries’ laws.
    """
# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def expand_user_query(text):
     return client.responses.create(
                    model="gpt-4o-mini",
                    input=[{"role": "user", "content": text}],
                    instructions=query_expansion_instructions,
                    stream=False
                ).output_text

# Function to get embeddings for a given text
def get_embedding(text: str):
    """
    Generate an embedding vector for the given input text using OpenAI embeddings API.
    """
    expanded_q = expand_user_query(text)
    print(f"Expanded user query: {expanded_q}")
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=expanded_q
    )
    return response.data[0].embedding

def response_instructions(knowledge_base):
    return f"""
        ### 📌 Chatbot Instructions

        **Purpose:**
        The chatbot’s purpose is to read provided source paragraphs and answer user questions based strictly on these sources.

        ---

        ### **Response Rules**

        1. The chatbot must only use the information from the provided sources to answer questions.
        2. If the answer cannot be found in the sources, the chatbot should clearly say so.
        3. The chatbot should respond in users language
        4. In the last line of your answer include comma separated list of ids (further referenced as "uuid list") of sources that you have used to produce response. Example [9830219d-78bb-491b-9af0-7826e34878d2,886492ad-502a-443d-aef7-7559826f1309]
        5. Include all sources you reference in answer in uuid list
        6. If no sources were relevant and none of them was used then do not write line with uuids

        ---

        ### **Sources**

        ```json
        {knowledge_base}
        ```
    """

def get_ai_response(knowledge_base, question):
     return client.responses.create(
                    model="gpt-4.1",
                    input=[{"role": "user", "content": question}],
                    instructions=response_instructions(knowledge_base),
                    stream=False
                )

def extract_source_ids_from_res(res: str):
    lines = [line.strip() for line in res.strip().splitlines() if line.strip()]
    if not lines:
        return []

    last_line = lines[-1]

    uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    
    uuids = re.findall(uuid_pattern, last_line)
    return uuids

def remove_uuid_line(message: str) -> str:
    lines = message.splitlines()
    if not lines:
        return message

    lines = lines[:-1]
    return "\n".join(lines)
