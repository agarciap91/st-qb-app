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
def init_bedrock_client(region_name, use_secrets=False, aws_access_key_id=None, aws_secret_access_key=None):
    try:
        # Verificar credenciales primero
        if use_secrets:
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
        elif aws_access_key_id and aws_secret_access_key:
            # Verificar credenciales
            valid, info = verify_aws_credentials(region_name, aws_access_key_id, aws_secret_access_key)
            if not valid:
                st.error(f"Credenciales inválidas: {info}")
                return None
                
            bedrock = boto3.client(
                'bedrock-runtime',
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # Usar credenciales por defecto (perfil AWS, IAM role, etc.)
            valid, info = verify_aws_credentials(region_name)
            if not valid:
                st.error(f"Credenciales por defecto inválidas: {info}")
                return None
                
            bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        
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
            ["Secrets.toml", "Credenciales manuales", "Credenciales por defecto"]
        )
        
        aws_access_key = None
        aws_secret_key = None
        use_secrets = False
        
        if auth_method == "Credenciales manuales":
            aws_access_key = st.text_input("AWS Access Key ID", type="password")
            aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
        elif auth_method == "Secrets.toml":
            use_secrets = True
            # Mostrar información sobre secrets.toml
            st.info("Usando credenciales desde secrets.toml")
            try:
                # Verificar que las claves existen
                if "AWS_ACCESS_KEY_ID" in st.secrets and "AWS_SECRET_ACCESS_KEY" in st.secrets:
                    st.success("✅ Credenciales encontradas en secrets.toml")
                    # Mostrar información de la cuenta (opcional)
                    try:
                        valid, arn_info = verify_aws_credentials(
                            aws_region, 
                            st.secrets["AWS_ACCESS_KEY_ID"], 
                            st.secrets["AWS_SECRET_ACCESS_KEY"]
                        )
                        if valid:
                            st.success(f"✅ Credenciales válidas")
                            # Mostrar solo los últimos 4 caracteres del ARN para privacidad
                            if arn_info and len(arn_info) > 20:
                                masked_arn = arn_info[:20] + "..." + arn_info[-10:]
                                st.caption(f"Cuenta: {masked_arn}")
                        else:
                            st.error(f"❌ Credenciales inválidas: {arn_info}")
                    except Exception as e:
                        st.warning(f"⚠️ No se pudo verificar credenciales: {str(e)}")
                else:
                    st.error("❌ Credenciales no encontradas en secrets.toml")
            except Exception as e:
                st.error(f"❌ Error al acceder a secrets.toml: {str(e)}")
        
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
    bedrock_client = init_bedrock_client(aws_region, use_secrets, aws_access_key, aws_secret_key)
    
    if not bedrock_client:
        st.error("No se pudo inicializar el cliente de Bedrock. Verifica tus credenciales.")
        st.info("""
        **Pasos para solucionar:**
        1. Verifica que tu archivo `.streamlit/secrets.toml` contenga:
           ```
           AWS_ACCESS_KEY_ID = "tu_access_key_aqui"
           AWS_SECRET_ACCESS_KEY = "tu_secret_key_aqui"
           ```
        2. Asegúrate de que las credenciales sean correctas
        3. Verifica que tengas permisos para Bedrock en la región seleccionada
        """)
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
    with st.expander("ℹ️ Información y Troubleshooting"):
        st.markdown("""
        ## Configuración de secrets.toml
        
        Crea un archivo en `.streamlit/secrets.toml` con el siguiente formato:
        ```toml
        AWS_ACCESS_KEY_ID = "AKIA..."
        AWS_SECRET_ACCESS_KEY = "..."
        AWS_DEFAULT_REGION = "us-east-1"
        ```
        
        ## Posibles causas del error InvalidSignatureException:
        
        1. **Credenciales incorrectas**: Verifica que tu Access Key y Secret Key sean correctos
        2. **Región incorrecta**: Asegúrate de usar la región donde tienes acceso a Bedrock
        3. **Permisos insuficientes**: Tu usuario IAM debe tener permisos para `bedrock:InvokeModel`
        4. **Hora del sistema**: Verifica que la hora de tu sistema esté sincronizada
        5. **Caracteres especiales**: Asegúrate de que no haya espacios o caracteres extra en las credenciales
        
        ## Política IAM mínima requerida:
        ```json
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:ListFoundationModels"
                    ],
                    "Resource": "*"
                }
            ]
        }
        ```
        """)

if __name__ == "__main__":
    main()
