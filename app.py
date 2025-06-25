import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import time
from gtts import gTTS
import io
import base64

# Configuración de la página
st.set_page_config(
    page_title="Chat con AWS Bedrock - Con Voz",
    page_icon="🤖",
    layout="wide"
)

# Función para verificar credenciales AWS
def verify_aws_credentials(region_name, aws_access_key_id=None, aws_secret_access_key=None):
    try:
        if aws_access_key_id and aws_secret_access_key:
            sts_client = boto3.client(
                'sts',
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            sts_client = boto3.client('sts', region_name=region_name)
        
        # Verificar identidad
        response = sts_client.get_caller_identity()
        return True, response.get('Arn', 'Unknown')
    except Exception as e:
        return False, str(e)

# Función para inicializar el cliente de Bedrock
@st.cache_resource
def init_bedrock_client(region_name):
    try:
        # Usar credenciales desde secrets.toml
        access_key = st.secrets["AWS_ACCESS_KEY_ID"]
        secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
        
        # Verificar credenciales
        valid, info = verify_aws_credentials(region_name, access_key, secret_key)
        if not valid:
            st.error(f"Credenciales inválidas: {info}")
            return None
        
        bedrock = boto3.client(
            'bedrock-runtime',
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        
        return bedrock
    except KeyError as e:
        st.error(f"Error: Falta la clave en secrets.toml: {str(e)}")
        return None
    except ClientError as e:
        st.error(f"Error de cliente AWS: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error al inicializar cliente Bedrock: {str(e)}")
        return None

# Función para convertir texto a voz
def text_to_speech(text, lang='es'):
    try:
        # Limitar el texto para evitar archivos muy grandes
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        # Crear el objeto gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Guardar en un buffer de memoria
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"Error al generar audio: {str(e)}")
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
    st.title("🤖 Chat con AWS Bedrock - Con Voz")
    st.markdown("Chatea con modelos de IA usando AWS Bedrock y escucha las respuestas")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Control de voz
        voice_enabled = st.checkbox("🔊 Activar respuestas de voz", value=True)
        
        if voice_enabled:
            voice_lang = st.selectbox(
                "Idioma de voz:",
                options=[
                    ("es", "Español"),
                    ("en", "English"),
                    ("fr", "Français"),
                    ("de", "Deutsch"),
                    ("it", "Italiano"),
                    ("pt", "Português")
                ],
                format_func=lambda x: x[1]
            )[0]
        
        st.divider()
        
        # Botón para limpiar chat
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.messages = []
            st.rerun()

    # Configuración por defecto
    aws_region = "us-east-1"
    selected_model = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3 Sonnet
    max_tokens = 1000
    temperature = 0.7
    use_secrets = True

    # Inicializar el cliente de Bedrock
    bedrock_client = init_bedrock_client(aws_region)
    
    if not bedrock_client:
        st.error("Error de conexión con AWS Bedrock. Verifica tu configuración.")
        return

    # Inicializar el historial de mensajes
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar el historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Si es un mensaje del asistente y tiene audio, mostrarlo
            if message["role"] == "assistant" and "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

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
                    
                    # Preparar el mensaje para agregar al historial
                    message_data = {"role": "assistant", "content": response}
                    
                    # Generar audio si está habilitado
                    if voice_enabled:
                        with st.spinner("Generando audio..."):
                            audio_data = text_to_speech(response, voice_lang)
                            
                            if audio_data:
                                # Mostrar el audio
                                st.audio(audio_data, format="audio/mp3")
                                # Agregar audio al mensaje
                                message_data["audio"] = audio_data
                            else:
                                st.warning("No se pudo generar el audio")
                    
                    # Agregar respuesta al historial
                    st.session_state.messages.append(message_data)
                else:
                    st.error("No se pudo generar una respuesta. Verifica la configuración.")

if __name__ == "__main__":
    main()
