import httpx
import streamlit as st

from rest.service import Chat, client


def main():
    # 1. Configuration de la page
    st.set_page_config(page_title="JuniaGPT", page_icon="🚀")
    st.title("JuniaGPT")

    # 2. Sidebar pour les paramètres
    temperature_mapping = {"Accurate": 0.0, "Balanced": 0.7, "Creative": 1.0}
    temperature_choice = st.sidebar.radio(
        label="Model Behavior",
        options=list(temperature_mapping.keys()),
        index=1,
    )
    temperature = temperature_mapping.get(temperature_choice)

    # 3. Initialisation de l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 4. Affichage des messages existants
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. Zone de saisie utilisateur
    if prompt := st.chat_input("What is your question?"):
        # Affichage du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Génération de la réponse (Tout est dans le IF pour éviter les répétitions)
        with st.chat_message("assistant"):
            chat = Chat(
                model="phi3.5",  # Modèle plus petit pour usage local
                temperature=temperature,
                messages=st.session_state.messages,
            )

            try:
                response = client.post(chat=chat)
                if response.status_code == httpx.codes.OK:
                    answer = response.json()["message"]["content"]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    st.error(f"Erreur serveur Ollama : {response.status_code}")
            except Exception as e:
                st.error(f"Impossible de contacter le service : {e}")


if __name__ == "__main__":
    main()
