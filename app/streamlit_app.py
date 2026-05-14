import streamlit as st
import json
import numpy as np
import faiss
import re

from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Adaptive ML Tutor",
    layout="wide"
)

st.title("🤖 Adaptive Conversational ML Tutor")

st.markdown("""
This tutor demonstrates:

- Conversational Memory
- Question Rewriting
- Semantic Retrieval
- Confusion Detection
- Adaptive Teaching Styles
""")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource

def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

@st.cache_data

def load_knowledge_base():

    kb_path = r"C:\Users\Moetez\OneDrive\ml_tutor_project\data\ml_knowledge_base\ml_concepts.json"

    with open(kb_path, "r", encoding="utf-8") as f:

        kb = json.load(f)

    return kb


knowledge_base = load_knowledge_base()


# =========================================================
# CREATE FAISS INDEX
# =========================================================

documents = [
    item["content"]
    for item in knowledge_base
]


doc_embeddings = embedding_model.encode(documents)


dimension = doc_embeddings.shape[1]


index = faiss.IndexFlatL2(dimension)


index.add(np.array(doc_embeddings))


# =========================================================
# KNOWLEDGE RETRIEVAL
# =========================================================

def retrieve_knowledge(query, top_k=2):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(knowledge_base[idx])

    return results


# =========================================================
# TOPIC EXTRACTION
# =========================================================

def extract_topic(conversation_history, current_question):

    topics = [
        "machine learning",
        "overfitting",
        "gradient descent",
        "linear regression",
        "decision trees",
        "regularization",
        "neural networks",
        "classification",
        "regression"
    ]

    # ------------------------------------------------
    # PRIORITY 1:
    # detect topic from current question
    # ------------------------------------------------

    current_q = current_question.lower()

    for topic in topics:

        if topic in current_q:

            return topic

    # ------------------------------------------------
    # PRIORITY 2:
    # fallback to recent history
    # ------------------------------------------------

    recent_history = " ".join(
        conversation_history[-3:]
    ).lower()

    for topic in topics:

        if topic in recent_history:

            return topic

    return "machine learning"


# =========================================================
# QUESTION REWRITING
# =========================================================

def rewrite_question(
    conversation_history,
    question
):

    topic = extract_topic(
        conversation_history,
        question
    )

    q = question.lower()

    if "why" in q:

        return f"Why is {topic} important in machine learning?"

    elif "how" in q and "prevent" in q:

        return f"How can we prevent issues related to {topic}?"

    elif "how" in q:

        return f"How does {topic} work in machine learning?"

    elif "explain" in q:

        return f"Explain {topic} in simple terms."

    elif "difference" in q:

        return f"What is the difference between concepts related to {topic}?"

    else:

        return question


# =========================================================
# CONFUSION DETECTION
# =========================================================

def detect_confusion(question):

    q = question.lower()

    patterns = [
        r"don't understand",
        r"confused",
        r"not clear",
        r"explain again",
        r"simplify",
        r"what do you mean",
        r"hard to understand",
        r"still don't get it",
        r"can you explain simply"
    ]

    for pattern in patterns:

        if re.search(pattern, q):

            return True

    return False


# =========================================================
# TEACHING STYLES
# =========================================================

teaching_styles = {

    "machine learning": {

        "beginner":
        "Machine learning is a field of AI where systems learn patterns from data to make predictions or decisions.",

        "analogy":
        "Machine learning is like teaching a computer using examples instead of explicitly programming every rule.",

        "concise":
        "Machine learning enables systems to learn from data.",

        "detailed":
        "Machine learning is a branch of artificial intelligence that allows models to learn patterns and improve performance using data."
    },

    "overfitting": {

        "beginner":
        "Overfitting happens when a machine learning model memorizes training data instead of learning general patterns.",

        "analogy":
        "Imagine memorizing exam answers without understanding the subject. You succeed on known questions but fail new ones. That is similar to overfitting.",

        "concise":
        "Overfitting reduces model generalization on unseen data.",

        "detailed":
        "Overfitting occurs when a model learns noise and specific details from training data, causing poor performance on new unseen examples."
    },

    "gradient descent": {

        "beginner":
        "Gradient descent is a method used to improve machine learning models by gradually reducing prediction errors.",

        "analogy":
        "Imagine walking downhill while blindfolded. You take small steps downward until you reach the lowest point. That is similar to gradient descent finding the minimum error.",

        "concise":
        "Gradient descent minimizes model error iteratively.",

        "detailed":
        "Gradient descent is an optimization algorithm that updates model parameters step by step to minimize a loss function."
    },

    "linear regression": {

        "beginner":
        "Linear regression predicts continuous numerical values using relationships between variables.",

        "analogy":
        "Imagine drawing the best straight line through scattered points on a graph. That is similar to linear regression.",

        "concise":
        "Linear regression models linear relationships between variables.",

        "detailed":
        "Linear regression estimates relationships between input variables and continuous target values using a best-fit linear equation."
    },

    "decision trees": {

        "beginner":
        "Decision trees make predictions by splitting data into branches using rules.",

        "analogy":
        "Imagine playing a yes/no guessing game where each question splits possibilities into smaller groups.",

        "concise":
        "Decision trees recursively split data based on features.",

        "detailed":
        "Decision trees are hierarchical machine learning models that recursively partition data based on feature conditions."
    },

    "neural networks": {

        "beginner":
        "Neural networks are machine learning models inspired by the human brain.",

        "analogy":
        "Imagine many tiny decision-makers working together to solve a problem. That is similar to neural networks.",

        "concise":
        "Neural networks learn complex patterns using interconnected layers.",

        "detailed":
        "Neural networks are layered computational models that learn hierarchical representations from data."
    }
}


# =========================================================
# MEMORY RETRIEVAL
# =========================================================

def retrieve_relevant_memory(
    conversation_history,
    query,
    top_k=3
):

    if len(conversation_history) == 0:

        return []

    embeddings = embedding_model.encode(
        conversation_history
    )

    memory_index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    memory_index.add(np.array(embeddings))

    query_embedding = embedding_model.encode([query])

    distances, indices = memory_index.search(
        np.array(query_embedding),
        min(top_k, len(conversation_history))
    )

    relevant_memory = []

    for idx in indices[0]:

        relevant_memory.append(
            conversation_history[idx]
        )

    return relevant_memory


# =========================================================
# MAIN TUTOR PIPELINE
# =========================================================

def conversational_ml_tutor(
    conversation_history,
    current_question
):

    # -----------------------------------------
    # Rewrite Question
    # -----------------------------------------

    rewritten_question = rewrite_question(
        conversation_history,
        current_question
    )

    # -----------------------------------------
    # Retrieve Relevant Memory
    # -----------------------------------------

    relevant_memory = retrieve_relevant_memory(
        conversation_history,
        rewritten_question
    )

    # -----------------------------------------
    # Retrieve Knowledge
    # -----------------------------------------

    retrieved_docs = retrieve_knowledge(
        rewritten_question
    )

    # -----------------------------------------
    # Detect Confusion
    # -----------------------------------------

    confused = detect_confusion(
        current_question
    )

    # -----------------------------------------
    # Topic Detection
    # -----------------------------------------

    topic = extract_topic(
        conversation_history,
        current_question
    )

    # -----------------------------------------
    # Importance Questions
    # -----------------------------------------

    is_importance_question = (
        "important" in current_question.lower()
        or "why" in current_question.lower()
    )

    importance_responses = {

        "overfitting":
        "Overfitting is important because it affects how well machine learning models perform on unseen data.",

        "gradient descent":
        "Gradient descent is important because it helps machine learning models learn efficiently by minimizing prediction errors.",

        "machine learning":
        "Machine learning is important because it enables intelligent systems to learn from data and automate complex tasks."
    }

    # -----------------------------------------
    # Select Teaching Mode
    # -----------------------------------------

    if confused:

        mode = "analogy"

    elif "briefly" in current_question.lower():

        mode = "concise"

    elif "detail" in current_question.lower():

        mode = "detailed"

    else:

        mode = "beginner"

    # -----------------------------------------
    # Generate Response
    # -----------------------------------------

    if is_importance_question and topic in importance_responses:

        response = importance_responses[topic]

    elif topic in teaching_styles:

        response = teaching_styles[topic][mode]

    else:

        response = retrieved_docs[0]["content"]

    return {

        "rewritten_question": rewritten_question,

        "relevant_memory": relevant_memory,

        "retrieved_knowledge": retrieved_docs,

        "confusion_detected": confused,

        "teaching_mode": mode,

        "response": response
    }


# =========================================================
# SESSION STATE
# =========================================================

if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = []


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input(
    "Ask a machine learning question..."
)


# =========================================================
# PROCESS USER INPUT
# =========================================================

if user_input:

    # Store user message
    st.session_state.chat_history.append({

        "role": "user",

        "content": user_input
    })

    st.session_state.conversation_history.append(
        user_input
    )

    # Run tutor pipeline
    result = conversational_ml_tutor(
        st.session_state.conversation_history,
        user_input
    )

    tutor_response = result["response"]

    # Store assistant response
    st.session_state.chat_history.append({

        "role": "assistant",

        "content": tutor_response
    })

    # Display user message
    with st.chat_message("user"):

        st.write(user_input)

    # Display assistant response
    with st.chat_message("assistant"):

        st.write(tutor_response)

        # Tutor reasoning panel
        with st.expander("🧠 Tutor Reasoning"):

            st.markdown("### Rewritten Question")

            st.write(
                result["rewritten_question"]
            )

            st.markdown("### Teaching Mode")

            st.write(
                result["teaching_mode"]
            )

            st.markdown("### Confusion Detected")

            st.write(
                result["confusion_detected"]
            )

            st.markdown("### Relevant Memory")

            for memory in result["relevant_memory"]:

                st.write("-", memory)

            st.markdown("### Retrieved Knowledge")

            for doc in result["retrieved_knowledge"]:

                st.write("-", doc["content"])


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📘 Demo Ideas")

    st.markdown("""
### Context Awareness
- What is overfitting?
- Why is it bad?
- How can we prevent it?

### Confusion Detection
- Explain gradient descent.
- I still don't understand.

### Adaptive Teaching
- Explain briefly.
- Explain in detail.
""")

    st.header("⚙️ Features")

    st.markdown("""
✅ Conversational Memory  
✅ Question Rewriting  
✅ Semantic Retrieval  
✅ Confusion Detection  
✅ Adaptive Teaching  
✅ Dialogue Coherence  
""")
