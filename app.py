import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import time

# Configuración de la página
st.set_page_config(
    page_title="Chat con AWS Bedrock",
    page_icon="🤖",
    layout="wide"
)

# Función para inicializar el cliente de Bedrock
@st.cache_resource
def init_bedrock_client(region_name, aws_access_key_id=None, aws_secret_access_key=None):
    try:
        if aws_access_key_id and aws_secret_access_key:
            bedrock = boto3.client(
                'bedrock-runtime',
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # Usar credenciales por defecto (perfil AWS, IAM role, etc.)
            bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        return bedrock
    except Exception as e:
        st.error(f"Error al inicializar cliente Bedrock: {str(e)}")
        return None

# Función para invocar el modelo
def invoke_bedrock_model(bedrock_client, model_id, prompt, max_tokens=1000, temperature=0.7):
    try:
        # Configuración del cuerpo de la solicitud según el modelo
        if "anthropic" in model_id.lower():
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        elif "ai21" in model_id.lower():
            body = {
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        elif "amazon" in model_id.lower():
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature
                }
            }
        elif "cohere" in model_id.lower():
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        else:
            # Configuración genérica
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

        # Invocar el modelo
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )

        # Procesar la respuesta
        response_body = json.loads(response['body'].read())
        
        # Extraer el texto según el modelo
        if "anthropic" in model_id.lower():
            return response_body.get('content', [{}])[0].get('text', '')
        elif "ai21" in model_id.lower():
            return response_body.get('completions', [{}])[0].get('data', {}).get('text', '')
        elif "amazon" in model_id.lower():
            return response_body.get('results', [{}])[0].get('outputText', '')
        elif "cohere" in model_id.lower():
            return response_body.get('generations', [{}])[0].get('text', '')
        else:
            return str(response_body)

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        st.error(f"Error de AWS: {error_code} - {error_message}")
        return None
    except Exception as e:
        st.error(f"Error al invocar el modelo: {str(e)}")
        return None

# Interfaz principal
def main():
    st.title("🤖 Chat con AWS Bedrock")
    st.markdown("Chatea con modelos de IA usando AWS Bedrock")

    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Configuración de AWS
        st.subheader("Credenciales AWS")
        aws_region = st.selectbox(
            "Región AWS",
            ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"],
            index=0
        )
        
        # Opciones de autenticación
        auth_method = st.radio(
            "Método de autenticación",
            ["Credenciales por defecto", "Credenciales manuales"]
        )
        
        aws_access_key = None
        aws_secret_key = None
        
        if auth_method == "Credenciales manuales":
            aws_access_key = st.text_input("AWS Access Key ID", type="password")
            aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
        
        # Selección del modelo
        st.subheader("Modelo")
        model_options = [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-v2:1",
            "ai21.j2-ultra-v1",
            "ai21.j2-mid-v1",
            "amazon.titan-text-express-v1",
            "cohere.command-text-v14"
        ]
        
        selected_model = st.selectbox("Seleccionar modelo", model_options)
        
        # Parámetros del modelo
        st.subheader("Parámetros")
        max_tokens = st.slider("Máximo de tokens", 100, 4000, 1000)
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.7, 0.1)
        
        # Botón para limpiar chat
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.messages = []
            st.rerun()

    # Inicializar el cliente de Bedrock
    bedrock_client = init_bedrock_client(aws_region, aws_access_key, aws_secret_key)
    
    if not bedrock_client:
        st.error("No se pudo inicializar el cliente de Bedrock. Verifica tus credenciales.")
        return

    # Inicializar el historial de mensajes
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar el historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu mensaje aquí..."):
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = invoke_bedrock_model(
                    bedrock_client, 
                    selected_model, 
                    prompt, 
                    max_tokens, 
                    temperature
                )
                
                if response:
                    st.markdown(response)
                    # Agregar respuesta al historial
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("No se pudo generar una respuesta. Verifica la configuración.")

    # Información adicional
    with st.expander("ℹ️ Información"):
        st.markdown("""
        """)

if __name__ == "__main__":
    main()
