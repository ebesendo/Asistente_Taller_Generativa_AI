import base64

import streamlit as st
from PIL import Image

from sql_app_functions import *

st.set_page_config(page_title="Asistente Galicia", page_icon="🤖", layout="centered", initial_sidebar_state="auto", menu_items=None)

def init_gpt_memory() -> None:
    st.session_state.gpt_memory = []

def setup_var_sessions() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "gpt_memory" not in st.session_state:
        init_gpt_memory()

def set_logo(path: str) -> None:
    """This function"""
    image = Image.open(path)
    st.image(
        image, 
        use_column_width=False, 
        width=int(image.size[0] * 0.3457),
        output_format='PNG'
    )

def set_background(path: str) -> None:
    """This functions set the app backgorund."""
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def set_bar(path: str) -> None:
    with open(path, "r", encoding='utf-8') as f:
        aclaraciones= f.read() 
    st.sidebar.markdown(aclaraciones)
    if st.sidebar.button("Resetear la memoria de GPT"):
        reset_memory()

def setup() -> None:
    setup_var_sessions()
    set_logo(path = r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\logo_galicia.png")
    set_background(path=r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\Fondo_galicia")
    set_bar(path='static/Aclaraciones.txt') 
def show_headers() -> None:
    st.markdown('<h1 style="color: white; text-align: center;">¡Bienvenido al portal de consultas del Banco Galicia!</h1>', unsafe_allow_html=True)
    html_style= '''<style>div.st-emotion-cache-7sak6c{padding-bottom: 1rem;}</style>'''
    st.markdown(html_style, unsafe_allow_html=True)
    html_style_2= '''<style>div.st-emotion-cache-6qob1r{background-color:#EBCB8B;color:black;}</style>'''
    st.markdown(html_style_2, unsafe_allow_html=True)
    html_style_3 = '''<style>.st-emotion-cache-faucq5 {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 38.4px;
    margin: 0px;
    line-height: 1.6;
    color: white;
    width: auto;
    user-select: none;
    background-color: #C41230;
    border: 1px solid rgba(46, 52, 64, 0.2);
    }
</style>'''
    st.markdown(html_style_3,unsafe_allow_html = True)

def show_history() -> None:
    """This function shows the message history in the chat."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def save_user_prompt(user_prompt: str) -> None:
    """This function save the user promt into gpt_memory and message history"""
    user_dict = {"role": "user", "content": user_prompt}
    # Save in message history
    st.session_state.messages.append(user_dict) # memoria streamlit
    st.session_state.gpt_memory.append(user_dict) # memoria gpt
    # Save in gpt memory
    #cf.expand_gpt_memory(st.session_state.gpt_memory, user_dict)

def get_gpt_response() -> str:
    """
    This function makes a request to the gpt model configured to respond.
    Also, upgrade the model memory.
    """
    gpt_response = get_assistant(st.session_state.gpt_memory)
    assit_dict = {"role": "assistant", "content": gpt_response}
    #st.session_state.gpt_memory.append(assit_dict)
    #st.session_state.messages.append(assit_dict)
    # cf.expand_gpt_memory(
    #     st.session_state.gpt_memory,
    #     {"role": "assistant", "content": gpt_response}
    # )

    return gpt_response

# def get_chat_response() -> str:
    
#     response = get_gpt_response()
#     query_result = 'nada'
#     if "```" in response:
#         query = response.split("```")[1]
#         print(query)
#         query_result = "El resultado de la consulta es query_result={}".format(cf.execute_sql_query(query))
#         cf.expand_gpt_memory(
#             st.session_state.gpt_memory, 
#             {"role": "assistant", "content":query_result}
#         )
#         response = cf.get_sql_response(
#             question=st.session_state.messages[-1]['content'], 
#             answer=query_result
#         )
        
#     return response, query_result

def reset_memory() -> None:
    #st.session_state.messages = []
    init_gpt_memory()


if __name__ == "__main__":
    setup()
    show_headers()
    primera_vez = True
    if primera_vez:
        with st.chat_message(name = "assistant",avatar = "👨‍💼"):
            bienvenida = st.write("Hola, soy Gali, el asistente virtual del Banco Galicia, ¿en qué puedo ayudarle?")
        primera_vez = False
    show_history()

    if prompt := st.chat_input("Escribe una pregunta:"):
        primera_vez = False
        # User pipeline
        save_user_prompt(prompt)
        st.chat_message("user",avatar = "🙍‍♂️").markdown(prompt)
        # consulta a gpt 
        with st.chat_message("assistant",avatar = "👨‍💼"):
            message_placeholder = st.empty() # que hace esto? es un contenedor vacio
            # Chat pipeline
            chat_response = get_gpt_response() 
            if ((prompt == 'cual fue la primer pregunta que te hice?' or prompt == 'te hice alguna pregunta?') and st.session_state.gpt_memory == []): # no esta entrando aca al resetear la memoria
                chat_response = "Respuesta Generada: No, no me hiciste ninguna pregunta anteriormente."
            message_placeholder.markdown(chat_response)
            assit_dict = {"role": "assistant", "content": chat_response}
            st.session_state.gpt_memory.append(assit_dict)
            primera_vez = False

        #st.chat_message("assistant").markdown(query)
        #st.chat_message("assistant").markdown(st.session_state.gpt_memory)
        # Save chat_response in message history
        st.session_state.messages.append({"role": "assistant", "content": chat_response})
        #st.session_state.messages.append({"role": "assistant", "content": query})
        
        diccionario = {}
        diccionario["question"] = prompt
        diccionario["answer"] = chat_response
        examples = [diccionario]
        predictions = chain.run(question = diccionario["question"],transcripcion = transcripcion)
        #graded_outputs = eval_chain.evaluate(examples, predictions, question_key="question", prediction_key="text")
        for i, example in enumerate(examples):
            #evaluación
            predicted_answer = f"Respuesta Predicha: {predictions}"
            st.chat_message("assistant",avatar = "👨‍💼").markdown(predicted_answer)
            assit_dict = {"role": "assistant", "content": predicted_answer}
            st.session_state.gpt_memory.append(assit_dict)
            st.session_state.messages.append({"role": "assistant", "content": predicted_answer})

            # predicted_grade = f"Calificación predicha: {graded_outputspredictions['results']}"
            # st.chat_message("assistant",avatar = "👨‍💼").markdown(predicted_grade)
            # assit_dict = {"role": "assistant", "content": predicted_grade}
            # st.session_state.gpt_memory.append(assit_dict)
            # st.session_state.messages.append({"role": "assistant", "content": predicted_grade})

            #rouge
            # Crear una instancia de la clase Rouge
            rouge = Rouge()
            # Calcular los puntajes ROUGE
            scores = rouge.get_scores(predictions,chat_response)
           
            #precisión y recuperación.
            lista_predecir = predictions
            lista_respuesta = chat_response
            precision, recuperacion = calcular_precision_y_recuperacion(lista_predecir,lista_respuesta)

            #cer
            respuesta_esperada = predictions
            respuesta_asistente = chat_response
            cer = calcular_cer(respuesta_esperada, respuesta_asistente)
            
            #f1_score
            f1_score = calcular_f1_score(precision, recuperacion)
            
            # dataframe
            df = pd.DataFrame({
                'Rouge-1 Recuperación':f"{scores[0]['rouge-1']['r']:.2f}",
                'Rouge-1 Precisión':f"{scores[0]['rouge-1']['p']:.2f}",
                'Rouge-1 Puntuación':f"{scores[0]['rouge-1']['f']:.2f}",
                'Rouge-2 Recuperación':f"{scores[0]['rouge-2']['r']:.2f}",
                'Rouge-2 Precisión':f"{scores[0]['rouge-2']['p']:.2f}",
                'Rouge-2 Puntuación':f"{scores[0]['rouge-2']['f']:.2f}",
                'Rouge-l Recuperación':f"{scores[0]['rouge-l']['r']:.2f}",
                'Rouge-l Precisión':f"{scores[0]['rouge-l']['p']:.2f}",
                'Rouge-l Puntuación':f"{scores[0]['rouge-l']['f']:.2f}",
                # 'Precisión':f"{precision:.2f}",
                # 'Recuperación':f"{recuperacion:.2f}",
                'Indicador Cer':f"{cer:.2f}"
                # "F1 Score":f"{f1_score:.2f}"
            }, index=['Resultados'])
            mostrar_dataframe_profesional(df)
            st.session_state.messages.append({"role": "assistant", "content":df})




            


        
        
    