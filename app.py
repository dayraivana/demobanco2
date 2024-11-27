import sys
from dotenv import load_dotenv
import pandas as pd
import csv
##Librerías para guardar en ICOS
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError
import ibm_boto3
import json
import random
import pdfplumber
from concurrent.futures import ThreadPoolExecutor
import os
import getpass
from ibm_watsonx_ai.foundation_models import ModelInference
import tempfile

load_dotenv()

api_key = os.getenv("API_KEY")
csv_name="Reclamos.csv"

##credenciales de ICOS
credentials = {
    'IBM_API_KEY_ID': os.getenv("IBM_API_KEY_ID"),
    'IAM_SERVICE_ID': os.getenv("IAM_SERVICE_ID"),
    'ENDPOINT':  os.getenv("ENDPOINT"),
    'IBM_AUTH_ENDPOINT': os.getenv("IBM_AUTH_ENDPOINT"),
    'BUCKET': os.getenv("BUCKET")
    }
    
cos = ibm_boto3.client(service_name='s3',
    ibm_api_key_id=credentials['IBM_API_KEY_ID'],
    ibm_service_instance_id=credentials['IAM_SERVICE_ID'],
    ibm_auth_endpoint=credentials['IBM_AUTH_ENDPOINT'],
    config=Config(signature_version='oauth'),
    endpoint_url=credentials['ENDPOINT'])

##credenciales de watsonx
def get_credentials():
	return {
		"url":"https://us-south.ml.cloud.ibm.com",
		"apikey": api_key
	}

model_id = "mistralai/mixtral-8x7b-instruct-v01"

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1
}

project_id = os.getenv("PROJECT_ID")
space_id = os.getenv("SPACE_ID")


model = ModelInference(
	model_id = model_id,
	params = parameters,
	credentials = get_credentials(),
	project_id = project_id,
	space_id = space_id
	)

# temp_dir = tempfile.gettempdir()
temp_dir = "/dev/shm"

# Función principal
def process_pdf(pdf_name, bucket_name="poc-asbanc"):

    local_pdf = f"{temp_dir}/{pdf_name}"
    local_txt = f"{temp_dir}/{os.path.splitext(pdf_name)[0]}.txt"
    local_csv = f"{temp_dir}/{csv_name}"  # Ruta temporal para trabajar con el archivo
    
    # Descargar el PDF de ICOS
    cos.download_file(bucket_name, pdf_name, local_pdf)
    
    # Extraer texto del PDF
    with pdfplumber.open(local_pdf) as pdf, open(local_txt, 'w', encoding='utf-8') as txt_file:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                txt_file.write(text + '\n')

    # Leer texto extraído
    with open(local_txt, 'r', encoding='utf-8') as txt_file:
        contenido = txt_file.read()

    prompt_input = f"""Recibirás documentos que corresponden a cartas o documentos que responden a reclamos de usuarios por temas relacionados con entidades bancarias. La información siempre estará en español. Lo que debes de hacer es extraer y solamente darme la información como respuesta en formato csv. Los campos a extraer son los siguientes:
1.	Monto (S/ o soles) de algún pago realizado, de alguna devolución, etc. 
2.	Fecha de cuando se envia la carta por parte de la entidad bancaria
3.  Fecha de cuando se presentó el reclamo a la entidad bancaria. 
4.	Numero de solicitud
5.	Actualizacion del reclamo (generar un resumen breve del reclamo y de su estado actual)
6.	Nombre de la entidad bancaria
7.	Nombre del cliente
8.	Producto (tipo de tarjeta de crédito, tipo de cuenta)
La información que brindes debe ser puntual. Todos los campos deben de aparecer en la respuesta que brindes. Esos campos son los indicados a continuación: Monto, Fecha del envio de la carta, Fecha de solicitud, Numero de solicitud, Actualizacion del reclamo, Nombre de la entidad bancaria, Nombre del cliente, Producto. Si no lo encuentra información sobre alguno de los campos decir: "No he podido obtener esta información del documento brindado."

Input: Lima, 16 de noviembre de 2023
Hola, Mariela:
Mi nombre es Erika del BCP y me comunico contigo para dar respuesta a tu solicitud N°C22216963,
referida a la devolución de S/ 500.00 soles correspondiente a una operación no reconocida en tu
Cuenta de Ahorros nro. 191-72573473-0-36.
Analicé tu solicitud nuevamente y lamento informarte que, en esta ocasión el resultado es no
favorable. Considero que es importante ofrecerte una explicación sobre lo sucedido.
Al realizar las verificaciones, identifiqué que en nuestro sistema figura que la operación se realizó
con la tarjeta de Débito e ingreso de la clave de 4 dígitos. Muy probable que, por consecuencia del
robo de tu tarjeta, una tercera persona haya podido acceder a información de esta y realizar
operaciones sin tu consentimiento.
Adjunto a continuación el detalle del movimiento registrado en tu Tarjeta de Débito:
Fecha Hora Detalle Monto Nro. Credimás
Retiro en Cajero BCP - Ubicado en
Agencia Bcp - Las Malvinas 3 - Av.
11/10/2023 17:33 S/ 500.00 4557-88**-****-2736
Argentina 401-407 - Jr. Pacasmayo
416-428 - Lima - Lima - Lima
También confirmé que, en la hora detallada no hubo una comunicación previa de tu parte respecto
al robo, pérdida o extravío de tu tarjeta. Adicionalmente te comento, que el bloqueo de la tarjeta se
realizó el 12 de octubre del 2023 a las 11:11:55 horas; es decir, luego de haberse realizado la
operación no reconocida.
Con respecto a la verificación de los videos de seguridad, lamento comunicarte que no contamos con
el video solicitado por la operación no reconocida. Cabe señalar, que, debido a un tema de capacidad
de almacenamiento en nuestros registros, los videos de seguridad de nuestras oficinas se encuentran
disponibles por un período determinado de tiempo, luego del cual no es factible poder obtenerlos,
por lo cual debo precisar que no es posible atender tu requerimiento de visualización de video.
Por ello, en esta ocasión, no podemos realizar la devolución de las operaciones cuestionadas tal y
como lo solicitaste, debido a que fueron realizadas bajo el proceso de seguridad que exigimos para
autentificar las operaciones, es decir: el uso de tarjeta de débito de nro. 4557-88**-****-2736 y el
ingreso de la clave secreta de 4 dígitos.
No queremos que vuelvas a estar expuesto(a) a un escenario similar, por lo que aprovecho esta
oportunidad para sugerirte algunas recomendaciones a tomar en cuenta en el uso de tu tarjeta:
● En caso hayas sufrido un robo o pérdida, comunícate inmediatamente al 311-9898 (*0)
y bloquea tu tarjeta, también puedes ingresar a nuestra App Banca Móvil y marcar el
ícono de “bloqueo”.
● En caso sufras el robo de tu celular, adicionalmente al bloqueo de la tarjeta con la que
te logueas a tus aplicaciones, bloquea también el chip y el equipo a través de tu
compañía de TELCOM.
Si no te encuentras conforme con nuestra respuesta, puedes solicitarnos una nueva evaluación a través
de nuestra banca por teléfono 311-9898. También puedes acudir a la Defensoría del Cliente Financiero
(consultas: 0-800-1-6777 o 224-1457 y/o https://dcf.pe/), al Instituto Nacional de Defensa de la
Competencia y de la Protección de la Propiedad Intelectual o a la Superintendencia de Banca, Seguros
y AFP´S.
● En caso de contar con el sistema operativo IOS, accede a tu cuenta de iCloud y borra la
información confidencial del teléfono robado/perdido. En el caso de Android, es
posible realizar un borrado remoto, sin embargo, previamente se tiene que configurar
el servicio. Es importante considerar, que mientras más pronto realices el borrado
remoto, es menos probable que el delincuente pueda acceder a tu información.
● Evita considerar tu clave de 6 dígitos como clave de desbloqueo de tu celular.
● Evita guardar tus claves en notas o en aplicaciones en tu celular y no uses la opción
“recordar clave” en tus equipos móviles.
● Que a través de nuestra página web viabcp.com hemos puesto a disposición de
nuestros clientes, consejos de seguridad que ayudarán a evitar que sean víctimas de
posibles fraudes electrónicos y presenciales. Esta información la podrás encontrar en
la opción de “Beneficios- Acerca del BCP – Por tu seguridad – Consejos de Seguridad”.
Espero haber aclarado tus dudas con respecto a tu solicitud.
Saludos cordiales.
Erika Arrostini Otiniano Laura Quispe Calderon
Supervisora Gerente
Banco de Crédito del Perú Banco de Crédito del Perú
Si no te encuentras conforme con nuestra respuesta, puedes solicitarnos una nueva evaluación a través
de nuestra banca por teléfono 311-9898. También puedes acudir a la Defensoría del Cliente Financiero
(consultas: 0-800-1-6777 o 224-1457 y/o https://dcf.pe/), al Instituto Nacional de Defensa de la
Competencia y de la Protección de la Propiedad Intelectual o a la Superintendencia de Banca, Seguros
y AFP´S.

Output: S/ 500.00; 16/11/2023; 11/10/2023; N°C22216963; La solicitud de devolución de S/ 500.00 no ha sido aceptada, ya que la operación fue realizada con la tarjeta de Débito e ingreso de la clave de 4 dígitos. Se recomienda tomar precauciones al utilizar la tarjeta de débito y en caso de robo o pérdida, comunicarse inmediatamente al 311-9898 (*0) para bloquear la tarjeta.; Banco de Crédito del Perú (BCP); Mariela; Cuenta de Ahorros

Input: {contenido}
Output:"""


    respuesta_generada = model.generate_text(prompt=prompt_input, guardrails=False)

    # Guardar resultados en un CSV

    try:
        # Intentar obtener metadatos del archivo en COS
        cos.head_object(Bucket=bucket_name, Key="Reclamos.csv")
        
        # Descargar el archivo existente
        cos.download_file(bucket_name, csv_name, local_csv)
        
        # Abrir el archivo en modo 'append'
        with open(local_csv, 'a', newline='', encoding='utf-8') as salida:
            writer = csv.writer(salida)
            # Escribir los nuevos datos
            datos = respuesta_generada.split(';')  # Asegúrate de estructurar los datos correctamente
            writer.writerow(datos)
        
        # Subir el archivo actualizado de nuevo al bucket
        cos.upload_file(local_csv, bucket_name, Key=csv_name)
        print(f"Archivo actualizado y subido a {csv_name} en el bucket.")
    
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            
            # Crear un nuevo archivo CSV
            local_csv = f"{temp_dir}/{csv_name}"  # Ruta temporal para trabajar con el archivo
            with open(local_csv, 'w', newline='', encoding='utf-8') as salida:
                writer = csv.writer(salida)
                # Escribir encabezados
                writer.writerow(["Monto", "Fecha de envío de carta", "Fecha de solicitud", "Número de solicitud", "Actualización del reclamo", "Nombre de la entidad bancaria", "Nombre del cliente", "Producto"])  # Cambia esto según tus columnas
                # Escribir los datos iniciales
                datos = respuesta_generada.split(';')
                writer.writerow(datos)
            
            # Subir el archivo nuevo al bucket
            cos.upload_file(local_csv, bucket_name, csv_name)
            print(f"Archivo creado y subido como {csv_name} en el bucket.")
        else:
            print(f"Error al verificar el archivo {csv_name}: {e}")
            raise


def main():
    # Extraer evento de entrada proporcionado por Code Engine
    event_data = os.getenv("CE_DATA")
    if not event_data:
        print("No se recibió evento. Asegúrate de que el evento está configurado correctamente.")
        return

    try:
        # Parsear los datos del evento
        event = json.loads(event_data)
        bucket_name = event.get('bucket')  # Extraer nombre del bucket
        object_name = event.get('key')  # Extraer nombre del archivo

        if not bucket_name or not object_name:
            print("El evento no contiene 'bucket_name' o 'object_key'.")
            return

        print(f"Procesando archivo '{object_name}' desde el bucket '{bucket_name}'.")

        # Llamar a la función process_pdf con los datos del evento
        process_pdf(object_name, bucket_name)

    except Exception as e:
        print(f"Error procesando el evento: {e}")



if __name__ == "__main__":
    # Simula un evento configurando manualmente la variable de entorno
#    os.environ["CE_DATA"] = json.dumps({
#        "bucket": "poc-asbanc",   # Nombre de tu bucket
#        "key": "2. Carta Cencosud OK.pdf"  # Nombre de tu archivo PDF
#    })

    # Ejecutar la función principal
    main()
