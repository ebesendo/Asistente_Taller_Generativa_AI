import streamlit as st
import tiktoken
import openai
import json
from typing import List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import QAGenerationChain
from langchain.evaluation.qa import QAEvalChain
from langchain.chat_models import AzureChatOpenAI
from rouge import Rouge 
import pandas as pd
from IPython.display import HTML

with open('credentials.json') as jsonfile:
    CREDENTIALS = json.load(jsonfile)

API_TYPE = CREDENTIALS['OpenAI']['API_TYPE']
API_BASE = CREDENTIALS['OpenAI']['AZURE_OPENAI_ENDPOINT']
API_KEY = CREDENTIALS['OpenAI']['AZURE_OPENAI_API_KEY']
API_VERSION = CREDENTIALS['OpenAI']['API_VERSION']
#EMBEDDINGS_ENGINE = st.secrets['EMBEDDINGS_ENGINE']
CHAT_ENGINE = CREDENTIALS['OpenAI']['DEPLOYMENT_MODEL_NAME']
print(CHAT_ENGINE)
# Configuracion
openai.api_type = API_TYPE
openai.api_version = API_VERSION
openai.api_base = API_BASE
openai.api_key = API_KEY

GPT_TOKENS_MODEL = "gpt-3.5-turbo"

TOKEN_BUDGET = 8000

# Connect to SQLite database
# conn = sqlite3.connect('chatgpt.db', check_same_thread=False)
# cursor = conn.cursor()

openai.api_key = API_KEY


# def get_system_configuration() -> dict:
#     """Configuracion del system."""
#     system = """Eres un asistente de visitadores médicos. Los visitadores médicos querrán hacerte consultas sobre los mismos.
#     La información para responder estas consultas se extrae de la tabla llamada "historicored" que tiene 
#     las columnas llamadas ['ID Medico', 'Nombres Medicos', 'Periodo', 'ID Droga', 'DROGA','Cantidad recetada', 'Obra social', 'Laboratorio']. 
#     Tené en cuenta que cada fila representa una receta de una determinada droga y que la receta puede contener varias unidades de esa droga lo cual 
#     se especifíca en "Cantidad recetada". 
#     Los médicos disponibles en 'Nombres Medicos' son Lisa, Bart y Maggie.
#     Cada droga puede ser fabricada por distintos laboratorios, en "ID Droga" tenemos códigos que representan una droga asociada a un laboratorio. 
#     Una misma droga puede tener distintos ID's según de qué laboratorio provienen. 
#     Donde Periodo está en formato date time, por ejemplo: '2021-03-01 00:00:00'. 'DROGA' tiene los nombres de las drogas que recetaron los médicos. 
#     "ID Droga" y "ID Medico" son códigos que se utilizan para designar a cada droga y cada medico. 
#     Cada receta se realiza mediante una determinada "Obra social" y "Laboratorio".
#     Cada Obra social está representada con un codigo en la base de datos "historicored"
#     Pueden hacerte consultas como: 
#     - Qué recetó Bart en marzo de 2021? En este caso la query involucra 'Nombres Medicos', 'DROGA' y 'Periodo'.
#     - Cuál es el médico que más SERTRALINE recetó? En este caso los parámetros son 'Nombres Medicos', 'DROGA' y 'Cantidad recetada'. 
#     - Cuánto LORAZEPAM recetó Lisa en enero 2021? En este caso 'Nombres Medicos', 'DROGA', 'Periodo' y 'Cantidad recetada' sumada.
#     Estas preguntas iran apareciendo durante la conversación, cuando aparezcan transformá la consulta que el usuario tiene en lenguaje natural a una query de sql. 
#     Con la query que proporciones se ejecutará la consulta y se te devolverá la respuesta "query_result" para que se la comentes al usuario. 
#     Asegurate de normalizar a minúscula todas las variables de texto, como por ejemplo LOWER(`Nombres Medicos`)= 'nombre_minúscula', LOWER(`LABORATORIO`) = 'laboratorio_minúsucula' y LOWER(`DROGA`) = 'droga_minúscula', antes de generar la query. 
#     También asegurate de que la query que generas siempre esté entre ```.
#     Una vez que te otorgue  "El resultado de su consulta es query_result" generá una respuesta usando ese resultado. Es decir no vuelvas a decirle al usuario cuál fue la query que realizaste, solo el resultado. 
#     Si la función te devuelve un error o un nan es posible que el visitador medico (usuario) haya escrito algo mal, asegurate de decirle que intente nuevamente revisando posibles errores de ortografía.
#     Sólo podes responder a las consultas de los usuarios con información de la tabla "historicored". Si no sabés algo podés generar la consulta de sql para la base de datos o bien decir que no sabes. 
#     Ejemplos de consultas y querys esperadas:
#     Consulta: Qué médico recetó más drogas en marzo de 2021?
#     Query:
#     ```
#     SELECT `Nombres Medicos`, SUM(`Cantidad recetada`) AS `Cantidad de drogas recetadas`
#     FROM `historicored`
#     WHERE LOWER(`Periodo`) LIKE '2021-03%'
#     GROUP BY `Nombres Medicos`
#     ORDER BY `Cantidad de drogas recetadas` DESC
#     LIMIT 1;
#     ```
#     """
#     return {"role": "system", "content": system}

chat = AzureChatOpenAI(temperature = 0,deployment_name = "chat")

with open('audioprueba.txt') as f:
    transcripcion = f.read()

def get_system_configuration()-> dict:
    """Configuracion del system support."""
    system = """
    Eres Gali, un asistente de empleados de la empresa Galicia Seguros, amable y respetuoso. En la misma hay un call center que recibe 
    llamadas de clientes que necesitan soporte para distintas tareas, como resolver consultas, darse de baja, etc.
    Tu tarea es, dada la transcripción de una de las llamadas al call center, responder consultas sobre la 
    conversación entre el cliente y el empleado de Galicia y el motivos de su llamada.
    Sólo podés responder en base a esa conversación. Si no encontrás la respuesta en la transcripción proporcionada respondé que no sabes.
    La transcripción de la llamada es:
    {}
    """.format(transcripcion)
    return {"role": "system", "content": system}



def get_assistant(
        mensajes: List[dict], 
        system_configuration: dict = get_system_configuration(),
        temperature: int = 0
    ) -> str:
    messages = [system_configuration] + mensajes
    print(messages)
    try:
        completion = openai.ChatCompletion.create(
            engine= CHAT_ENGINE,
            messages= messages,
            temperature = temperature,)
        return completion.choices[0].message["content"]
    except Exception as e:
        print(f'error: {e}')
        return 'No se pudo conectar con gpt'

# def get_sql_response(question, answer):
#     return get_assistant(
#         mensajes=[{
#             "role": "user", 
#             "content": f'Pregunta: {question}, Respuesta: {answer}'
#         }],
#         system_configuration=system_configuration_support(),
#         temperature=0.2
#     )


# def get_table_columns(table_name) -> list:
#     """"Function to get table columns from SQLite database."""
#     cursor.execute("PRAGMA table_info({})".format(table_name))
#     columns = cursor.fetchall()
#     #print(columns)
#     return [column[1] for column in columns]

# def execute_sql_query(query):
#     """Function to execute SQL query on SQLite database."""
#     cursor.execute(query)
#     result = cursor.fetchall()
#     return result

def num_tokens(text: str, model: str = GPT_TOKENS_MODEL) -> int:
    """Return the number of tokens in a string."""
    return len(tiktoken.get_encoding('p50k_base').encode(text)) #return len(tiktoken.encoding_for_model(model).encode(text))

def num_tokens_in_gpt_configuration() -> int:
    """Return the number of tokens in a system_configuration of gpt."""
    return num_tokens(get_system_configuration()['content'])

def total_tokens_in_memory(gpt_memory: List[dict]) -> int:
    """Return the number of tokens in a gpt memory structure."""
    return (
        sum(num_tokens(message['content']) for message in gpt_memory)
        + num_tokens_in_gpt_configuration()
    )

def expand_gpt_memory(gpt_memory: List[dict], new_message: dict) ->  List[dict]:
    """
    This function adds new messages to the gpt memory structure, as long as the number of tokens 
    in the new message does not exceed the token limit allowed by the gpt api.
    If so, the oldest messages are deleted until there are enough tokens for the new message.
    """
    total_tokens = (
        total_tokens_in_memory(gpt_memory) 
        + num_tokens(new_message['content'])
    )

    while total_tokens > TOKEN_BUDGET:
        total_tokens -= num_tokens(gpt_memory.pop(0)['content'])

    gpt_memory.append(new_message)

    return gpt_memory

# Métricas
prompt = PromptTemplate(
    template = """
    Eres Gali, un asistente de empleados de la empresa Galicia Seguros, amable y respetuoso. En la misma hay un call center que recibe 
    llamadas de clientes que necesitan soporte para distintas tareas, como resolver consultas, darse de baja, etc.
    Tu tarea es, dada la transcripción de una de las llamadas al call center, responder consultas sobre la 
    conversación entre el cliente y el empleado de Galicia,el motivo de su llamada, el sentimiento del cliente y el operador,motivo y conflicto de la llamada,posible solución al conflicto y tono de la llamada, cliente u operador,etc.
    Consulta:{question}.
    Sólo podés responder en base a esa conversación.
    La transcripción de la llamada es:
    {transcripcion}.
    """
,input_variables=["question","transcripcion"]
)
chain = LLMChain(llm=chat, prompt=prompt)
eval_chain = QAEvalChain.from_llm(chat)


def calcular_precision_y_recuperacion(respuestas_esperadas, respuestas_asistente):
    # Convertir listas a conjuntos para facilitar operaciones de intersección y diferencia
    conjunto_esperado = set(respuestas_esperadas)
    conjunto_asistente = set(respuestas_asistente)

    # Calcular precisión
    verdaderos_positivos = conjunto_asistente.intersection(conjunto_esperado)
    precision = len(verdaderos_positivos) / len(conjunto_asistente) if conjunto_asistente else 0

    # Calcular recuperación
    recuperacion = len(verdaderos_positivos) / len(conjunto_esperado) if conjunto_esperado else 0

    return precision, recuperacion

def calcular_f1_score(precision, recuperacion):
    # Verificar si la precisión y la recuperación son cero para evitar división por cero
    if precision == 0 and recuperacion == 0:
        return 0
    return 2 * (precision * recuperacion) / (precision + recuperacion)

def calcular_cer(respuesta_esperada, respuesta_asistente):
    # Dividir las respuestas en palabras
    palabras_esperadas = respuesta_esperada.split()
    palabras_asistente = respuesta_asistente.split()

    # Contar el número de palabras que no coinciden
    errores = sum(1 for esperada, asistente in zip(palabras_esperadas, palabras_asistente) if esperada != asistente)
    
    # Añadir errores por diferencia de longitud
    diferencia_longitud = abs(len(palabras_esperadas) - len(palabras_asistente))
    errores += diferencia_longitud

    # Calcular la tasa de error
    total_palabras = max(len(palabras_esperadas), len(palabras_asistente))
    return errores / total_palabras if total_palabras > 0 else 0

def mostrar_dataframe_profesional(dataframe):
    style = '''
        <style>
            th {
                background-color: #C41230; 
                color: white;
                font-weight: bold;
                text-align: middle;
            }
            th, td {
                padding: 10px;
                border: 2px solid #181818;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                color:black;
                
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            .st-emotion-cache-w9hboq {
                                        border-bottom: 1px solid rgba(46, 52, 64, 0.1);
                                        border-right: 1px solid rgba(46, 52, 64, 0.1);
                                        vertical-align: middle;
                                        padding: 0.25rem 0.375rem;
                                        font-weight: 400;
                                        color: white;
            }
            .st-emotion-cache-1ird9gk {
                                        background-color: #EBCB8B;
                                        border-bottom: 1px solid rgba(46, 52, 64, 0.1);
                                        border-right: 1px solid rgba(46, 52, 64, 0.1);
                                        vertical-align: middle;
                                        padding: 0.25rem 0.375rem;
                                        font-weight: 400;
                                        text-align-last: center;
            }
        </style>
    '''
    st.markdown(style, unsafe_allow_html=True)
    st.table(dataframe)
                