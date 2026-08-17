"""
Chatbot Conversacional con Detección de Países
Implementa detección de países e inyección de datos de la API local de países
"""

from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
from datetime import datetime
from typing import Optional, Dict, List

# Configurar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Configuración de la API local
API_BASE = "http://localhost:3000"

# Configuración del modelo
MODEL = "gpt-4.1-nano"
SYSTEM_PROMPT = """Eres un asistente conversacional amigable y útil. 
Tu objetivo es responder preguntas generales de cualquier tema.

IMPORTANTE: Si se ha proporcionado información sobre un país en la conversación, 
usa ÚNICAMENTE esa información para responder preguntas sobre ese país. 
No uses tu conocimiento previo del país, solo lo que se ha proporcionado.

Si el usuario pregunta sobre datos de un país y la información no se ha proporcionado, 
indica claramente que esa información no está disponible en los datos que tienes."""


class ClienteAPIsPaises:
    """Cliente para la API local de países"""
    
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
        self.cache_paises = None
        self.cache_individual = {}
    
    def _log(self, mensaje: str, tipo: str = "API"):
        """Función auxiliar para logging con timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"   [{timestamp}] [{tipo}] {mensaje}")
    
    def obtener_lista_paises(self) -> List[Dict]:
        """
        Obtiene la lista completa de países de la API local
        """
        if self.cache_paises is not None:
            self._log("Usando caché de países", "CACHE")
            return self.cache_paises
        
        try:
            self._log(f"GET {self.api_base}/countries", "REQUEST")
            response = requests.get(f"{self.api_base}/countries", timeout=5)
            response.raise_for_status()
            
            paises = response.json()
            self.cache_paises = paises
            self._log(f"✓ Se obtuvieron {len(paises)} países", "SUCCESS")
            return paises
            
        except requests.exceptions.ConnectionError:
            self._log(f"❌ No se pudo conectar a {self.api_base}. ¿Está la API ejecutándose?", "ERROR")
            return []
        except requests.exceptions.RequestException as e:
            self._log(f"❌ Error al obtener lista de países: {e}", "ERROR")
            return []
    
    def obtener_pais_por_nombre(self, nombre_pais: str) -> Optional[Dict]:
        """
        Obtiene información detallada de un país específico de la API local
        """
        if nombre_pais in self.cache_individual:
            self._log(f"Usando caché para país: {nombre_pais}", "CACHE")
            return self.cache_individual[nombre_pais]
        
        try:
            url = f"{self.api_base}/countries/{nombre_pais}"
            self._log(f"GET {url}", "REQUEST")
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            datos = response.json()
            
            # Manejar casos donde la respuesta es un array
            if isinstance(datos, list):
                if len(datos) > 0:
                    pais = datos[0]
                else:
                    self._log(f"❌ Respuesta vacía para {nombre_pais}", "ERROR")
                    return None
            elif isinstance(datos, dict):
                pais = datos
            else:
                self._log(f"❌ Formato de respuesta no esperado: {type(datos)}", "ERROR")
                return None
            
            self.cache_individual[nombre_pais] = pais
            self._log(f"✓ Datos de {nombre_pais} obtenidos correctamente", "SUCCESS")
            return pais
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self._log(f"❌ País '{nombre_pais}' no encontrado en la API", "NOT_FOUND")
            else:
                self._log(f"❌ Error HTTP: {e.response.status_code}", "ERROR")
            return None
        except requests.exceptions.RequestException as e:
            self._log(f"❌ Error al obtener datos del país: {e}", "ERROR")
            return None
    
    def detectar_pais_en_texto(self, texto: str) -> Optional[Dict]:
        """
        Detecta si el texto menciona un país comparando contra la lista de países.
        Retorna los datos del país si se detecta, None en caso contrario.
        """
        texto_lower = texto.lower()
        paises_lista = self.obtener_lista_paises()
        
        if not paises_lista:
            return None
        
        # Buscar coincidencias en la lista de países
        for pais_item in paises_lista:
            # La API retorna strings con el nombre del país
            if isinstance(pais_item, str):
                if pais_item.lower() in texto_lower:
                    self._log(f"País detectado: {pais_item}", "MATCH")
                    # Obtener datos completos del país
                    datos_pais = self.obtener_pais_por_nombre(pais_item)
                    if datos_pais:
                        return datos_pais
                    else:
                        return {"name": pais_item}
            elif isinstance(pais_item, dict) and "name" in pais_item:
                nombre = pais_item.get("name", "")
                if isinstance(nombre, str) and nombre.lower() in texto_lower:
                    self._log(f"País detectado: {nombre}", "MATCH")
                    return pais_item
        
        return None
    
    def formatear_contexto_pais(self, pais: Dict) -> str:
        """
        Formatea la información del país para inyectarla en el prompt del LLM
        """
        try:
            # Extraer información con valores por defecto
            nombre = pais.get("name", "Desconocido")
            nombre_oficial = pais.get("officialName", pais.get("name", "Desconocido"))
            capital = pais.get("capital", "N/A")
            region = pais.get("region", "N/A")
            subregion = pais.get("subregion", "N/A")
            poblacion = pais.get("population", "N/A")
            area = pais.get("areaKm2", "N/A")
            
            # Procesar idiomas (array de strings)
            idiomas = pais.get("languages", [])
            idiomas_str = ", ".join(idiomas) if isinstance(idiomas, list) else str(idiomas)
            if not idiomas_str:
                idiomas_str = "N/A"
            
            # Procesar monedas (array de objetos con {code, name, symbol})
            monedas = pais.get("currencies", [])
            monedas_str = ""
            if isinstance(monedas, list) and len(monedas) > 0:
                monedas_lista = []
                for m in monedas:
                    if isinstance(m, dict):
                        nombre_moneda = m.get("name", m.get("code", ""))
                        codigo_moneda = m.get("code", "")
                        monedas_lista.append(f"{nombre_moneda} ({codigo_moneda})")
                    else:
                        monedas_lista.append(str(m))
                monedas_str = ", ".join(monedas_lista)
            
            if not monedas_str:
                monedas_str = "N/A"
            
            # Construir texto de contexto
            contexto = f"""
═══════════════════════════════════════════════════════════════
DATOS DEL PAÍS (Información actualizada de la API):
═══════════════════════════════════════════════════════════════
• Nombre: {nombre}
• Nombre Oficial: {nombre_oficial}
• Capital: {capital}
• Región: {region}
• Subregión: {subregion}
• Población: {poblacion if isinstance(poblacion, str) else f'{poblacion:,}'} habitantes
• Área: {area} km²
• Idiomas: {idiomas_str}
• Monedas: {monedas_str}
═══════════════════════════════════════════════════════════════
"""
            return contexto
        except Exception as e:
            import traceback
            self._log(f"Error al formatear contexto del país: {e}", "ERROR")
            self._log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return f"Información del país: {pais}"


class ChatbotPaises:
    """Chatbot conversacional con detección de países e inyección de datos"""
    
    def __init__(self):
        self.cliente_api = ClienteAPIsPaises()
        self.historial = []
        self.pais_actual = None
    
    def _log(self, mensaje: str, tipo: str = "INFO"):
        """Función auxiliar para logging con timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{tipo}] {mensaje}")
    
    # FASE 1: DETECCIÓN
    def fase_1_detectar_pais(self, mensaje_usuario: str) -> Optional[Dict]:
        """
        FASE 1 - DETECCIÓN:
        Identifica si el mensaje del usuario menciona un país.
        Retorna los datos del país si se detecta, None en caso contrario.
        """
        self._log("FASE 1: DETECCIÓN - ¿El mensaje menciona un país?", "FASE")
        
        pais = self.cliente_api.detectar_pais_en_texto(mensaje_usuario)
        
        if pais:
            # Obtener el nombre del país (campo "name" es un string)
            nombre_pais = pais.get("name", "Desconocido")
            self._log(f"✓ País detectado: {nombre_pais}", "DETECTED")
            self.pais_actual = pais
            return pais
        
        self._log("✗ No se detectó país", "INFO")
        self.pais_actual = None
        return None
    
    # FASE 2: INYECCIÓN
    def fase_2_inyectar_contexto(self, pais: Dict) -> str:
        """
        FASE 2 - INYECCIÓN:
        Consulta la API de países para obtener datos actualizados
        y prepara el contexto a inyectar en el prompt.
        """
        self._log("FASE 2: INYECCIÓN - Consultando API para obtener datos actualizados", "FASE")
        
        # Obtener el nombre del país (es un string en esta API)
        nombre_pais = pais.get("name", "")
        
        if not nombre_pais:
            self._log("❌ No se puede obtener el nombre del país", "ERROR")
            return ""
        
        # Obtener datos detallados del país
        datos_detallados = self.cliente_api.obtener_pais_por_nombre(nombre_pais)
        
        if datos_detallados:
            contexto = self.cliente_api.formatear_contexto_pais(datos_detallados)
            self._log(f"✓ Contexto de {nombre_pais} inyectado", "INJECTED")
            return contexto
        
        self._log(f"⚠ No se pudieron obtener datos de {nombre_pais}", "WARNING")
        return ""
    
    # FASE 3: RESPUESTA
    def fase_3_generar_respuesta(self, mensaje_usuario: str, contexto_pais: str = "") -> str:
        """
        FASE 3 - RESPUESTA:
        Genera la respuesta final usando OpenAI con el contexto inyectado.
        """
        self._log("FASE 3: RESPUESTA - Generando respuesta con OpenAI", "FASE")
        
        # Preparar el mensaje para el LLM
        contenido_mensaje = mensaje_usuario
        
        if contexto_pais:
            contenido_mensaje = f"{contexto_pais}\nPregunta del usuario: {mensaje_usuario}"
        
        # Construir historial de mensajes
        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Agregar historial previo
        mensajes.extend(self.historial)
        
        # Agregar mensaje actual
        mensajes.append({"role": "user", "content": contenido_mensaje})
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=mensajes,
                temperature=0.7,
                max_tokens=1000
            )
            
            respuesta = response.choices[0].message.content
            self._log("✓ Respuesta generada", "SUCCESS")
            return respuesta
            
        except Exception as e:
            self._log(f"❌ Error al generar respuesta: {e}", "ERROR")
            return "Disculpa, hubo un error al procesar tu pregunta. Por favor, intenta de nuevo."
    
    def procesar_mensaje(self, mensaje_usuario: str) -> str:
        """
        Procesa un mensaje del usuario a través de las 3 fases
        """
        print("\n" + "="*70)
        self._log(f"Usuario: {mensaje_usuario}", "INPUT")
        
        # FASE 1: Detectar si hay país
        pais_detectado = self.fase_1_detectar_pais(mensaje_usuario)
        
        # FASE 2: Inyectar contexto si se detectó país
        contexto_pais = ""
        if pais_detectado:
            contexto_pais = self.fase_2_inyectar_contexto(pais_detectado)
        
        # FASE 3: Generar respuesta
        respuesta = self.fase_3_generar_respuesta(mensaje_usuario, contexto_pais)
        
        # Actualizar historial (sin el contexto inyectado, solo el mensaje original)
        self.historial.append({"role": "user", "content": mensaje_usuario})
        self.historial.append({"role": "assistant", "content": respuesta})
        
        # Limitar historial a últimos 20 mensajes (10 turnos)
        if len(self.historial) > 20:
            self.historial = self.historial[-20:]
        
        # Mostrar respuesta
        print(f"\n🤖 Asistente: {respuesta}")
        print("="*70)
        
        return respuesta


def main():
    """Función principal - Loop del chatbot"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "CHATBOT CONVERSACIONAL CON DETECCIÓN DE PAÍSES" + " "*12 + "║")
    print("║" + " "*15 + "API Local en http://localhost:3000" + " "*19 + "║")
    print("╚" + "="*68 + "╝")
    print("\n📝 Puedes hacer preguntas generales o sobre países específicos.")
    print("🌍 El chatbot detectará automáticamente países y usará datos actualizados.")
    print("❌ Escribe 'salir' para terminar.\n")
    
    chatbot = ChatbotPaises()
    
    while True:
        try:
            # Leer entrada del usuario
            entrada = input("\n👤 Tú: ").strip()
            
            if not entrada:
                print("   ⚠️  Por favor, ingresa un mensaje.")
                continue
            
            if entrada.lower() in ["salir", "exit", "quit"]:
                print("\n👋 ¡Hasta luego! Gracias por usar el chatbot.\n")
                break
            
            # Procesar mensaje
            chatbot.procesar_mensaje(entrada)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chatbot interrumpido. ¡Hasta luego!\n")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print("   Por favor, intenta de nuevo.\n")


if __name__ == "__main__":
    main()