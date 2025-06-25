import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import time
from gtts import gTTS
import io
import base64
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import tempfile
import os

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

# Función para convertir voz a texto
def speech_to_text(audio_data, language='es-ES'):
    try:
        # Crear un archivo temporal para el audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        # Inicializar el reconocedor
        recognizer = sr.Recognizer()
        
        # Cargar el archivo de audio
        with sr.AudioFile(tmp_file_path) as source:
            # Ajustar para ruido ambiente
            recognizer.adjust_for_ambient_noise(source)
            # Grabar el audio
            audio = recognizer.record(source)
        
        # Convertir a texto usando Google Speech Recognition
        text = recognizer.recognize_google(audio, language=language)
        
        # Limpiar el archivo temporal
        os.unlink(tmp_file_path)
        
        return text
    
    except sr.UnknownValueError:
        return "No se pudo entender el audio. Intenta hablar más claro."
    except sr.RequestError as e:
        return f"Error del servicio de reconocimiento: {str(e)}"
    except Exception as e:
        return f"Error al procesar el audio: {str(e)}"
    finally:
        # Asegurar que el archivo temporal se elimine
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

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
        speech_input_enabled = st.checkbox("🎤 Activar entrada por voz", value=True)
        
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
        
        if speech_input_enabled:
            speech_lang = st.selectbox(
                "Idioma de reconocimiento:",
                options=[
                    ("es-ES", "Español"),
                    ("en-US", "English (US)"),
                    ("en-GB", "English (UK)"),
                    ("fr-FR", "Français"),
                    ("de-DE", "Deutsch"),
                    ("it-IT", "Italiano"),
                    ("pt-BR", "Português (Brasil)")
                ],
                format_func=lambda x: x[1]
            )[0]
        
        st.divider()
        
        # Botón para limpiar chat
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.messages = []
            st.session_state.last_audio = None
            st.rerun()
            
        # Botón para resetear grabación
        if speech_input_enabled and st.button("🔄 Resetear Grabación"):
            st.session_state.last_audio = None
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
    
    # Inicializar estado para controlar el procesamiento de audio
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None

    # Mostrar el historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Si es un mensaje del asistente y tiene audio, mostrarlo
            if message["role"] == "assistant" and "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

    # Sección de entrada por voz
    if speech_input_enabled:
        st.markdown("### 🎤 Graba tu pregunta")
        
        # Grabador de audio
        audio_bytes = audio_recorder(
            text="Presiona para grabar",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x",
        )
        
        # Procesar audio grabado solo si es nuevo
        if audio_bytes and audio_bytes != st.session_state.last_audio:
            st.session_state.last_audio = audio_bytes
            st.audio(audio_bytes, format="audio/wav")
            
            # Convertir audio a texto
            with st.spinner("Procesando audio..."):
                transcribed_text = speech_to_text(audio_bytes, speech_lang)
                
                if transcribed_text and not transcribed_text.startswith("No se pudo") and not transcribed_text.startswith("Error"):
                    st.success(f"🎯 Texto reconocido: *{transcribed_text}*")
                    
                    # Procesar como si fuera un mensaje de texto
                    process_user_message(transcribed_text, bedrock_client, selected_model, max_tokens, temperature, voice_enabled, voice_lang if voice_enabled else None)
                else:
                    st.error(transcribed_text)
        
        st.divider()

    # Input del usuario por texto
    if prompt := st.chat_input("Escribe tu mensaje aquí..."):
        process_user_message(prompt, bedrock_client, selected_model, max_tokens, temperature, voice_enabled, voice_lang if voice_enabled else None)
        st.rerun()  # Solo para mensajes de texto

# Función auxiliar para procesar mensajes del usuario
def process_user_message(prompt, bedrock_client, selected_model, max_tokens, temperature, voice_enabled, voice_lang):
    # Verificar que no sea un mensaje duplicado
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.messages[-1]["content"] == prompt:
        return
        
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generar respuesta del asistente
    response = invoke_bedrock_model(
        bedrock_client, 
        selected_model, 
        prompt, 
        max_tokens, 
        temperature
    )
    
    if response:
        # Preparar el mensaje para agregar al historial
        message_data = {"role": "assistant", "content": response}
        
        # Generar audio si está habilitado
        if voice_enabled and voice_lang:
            audio_data = text_to_speech(response, voice_lang)
            if audio_data:
                message_data["audio"] = audio_data
        
        # Agregar respuesta al historial
        st.session_state.messages.append(message_data)
    else:
        # Agregar mensaje de error al historial
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Lo siento, no pude generar una respuesta. Por favor, verifica la configuración."
        })

if __name__ == "__main__":
    main()
