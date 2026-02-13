import os
import asyncio
import streamlit as st

from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent


# --- CONFIG ---
MCP_SERVER_URL = "https://ynov-projet-mcp-c6a26a44f877.herokuapp.com/sse"



st.set_page_config(page_title="Agent MCP MLflow", layout="wide")
st.title("Interface MCP — MLflow Cats vs Dogs")

col1, col2 = st.columns([1, 2])

# ---------------- UI ACTIONS ----------------
with col1:
    st.subheader("Actions")

    action = st.selectbox(
        "Choisir une action",
        ["Chat Libre", "Train CatsDogs", "Prédire image", "Lister Repos Github"]
    )

    user_input = ""

    if action == "Train CatsDogs":
        user_input = "Entraîne un modèle cats dogs"

    elif action == "Prédire image":
        image_path = st.text_input("Chemin image serveur (ex: /tmp/cat.jpg)")
        if image_path:
            user_input = f"Utilise l'outil predict_cat_dog avec image {image_path}"

    elif action == "Lister Repos Github":
        user_input = "Liste mes repos publics Github"

    else:
        user_input = st.text_input("Votre demande")


async def run_agent(query: str) -> str:
    if not MCP_SERVER_URL:
        return "Erreur: MCP_SERVER_URL non défini."

    try:
        async with sse_client(url=MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await load_mcp_tools(session)

                llm = ChatOllama(
                    model="qwen2.5:0.5b",
                    base_url="http://51.44.163.235:11434",
                    temperature=0,
                )

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "Tu es un assistant utile. Utilise les outils quand c'est pertinent."),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ]
                )

                agent = create_tool_calling_agent(llm, tools, prompt)

                executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                )

                result = await executor.ainvoke({"input": query})
                return result.get("output", "Pas de réponse.")

    except Exception as e:
        return f"Erreur: {type(e).__name__}: {str(e)}"


with col2:
    st.subheader("Résultat")

    if st.button("Exécuter"):
        if user_input:
            with st.spinner("Agent en cours..."):
                response = asyncio.run(run_agent(user_input))
                st.success(response)
        else:
            st.warning("Veuillez entrer une demande.")