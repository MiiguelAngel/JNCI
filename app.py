import streamlit as st
import openai
import os
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from PIL import Image
import base64
import io
from typing import List, Optional, Dict
from dataclasses import dataclass
import json
from streamlit_option_menu import option_menu

# Configuración
load_dotenv()
# Obtener la clave API de los secrets de Streamlit
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]

@dataclass
class Config:
    """Configuración de la aplicación"""
    MAX_TOKENS: int = 4096
    CHUNK_SIZE: int = 2000
    OCR_MODEL: str = "gpt-4o"
    CORRECTION_MODEL: str = "gpt-3.5-turbo"
    THUMBNAIL_SIZE: tuple = (300, 400)  # Tamaño de las miniaturas

class PDFProcessor:
    """Clase para procesar documentos PDF"""
    
    @staticmethod
    def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
        """Convierte un PDF a una lista de imágenes"""
        return convert_from_bytes(pdf_bytes)
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convierte una imagen a formato base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def create_thumbnail(image: Image.Image, size: tuple) -> Image.Image:
        """Crea una miniatura de la imagen"""
        return image.copy().thumbnail(size, Image.Resampling.LANCZOS)

class OpenAIService:
    """Clase para manejar las interacciones con OpenAI"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract_text_from_image(self, base64_image: str) -> str:
        """Extrae texto de una imagen usando GPT-4 Vision"""
        response = openai.ChatCompletion.create(
            model=self.config.OCR_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un OCR especializado. Extrae el texto de la imagen y devuélvelo exactamente como aparece, sin hacer correcciones. Mantén el formato y la estructura del texto original."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extrae el texto de esta imagen manteniendo el formato original."
                        }
                    ]
                }
            ],
            max_tokens=self.config.MAX_TOKENS
        )
        return response.choices[0].message.content
    
    def correct_text(self, text: str) -> str:
        """Corrige la ortografía del texto usando GPT-4"""
        chunks = [text[i:i+self.config.CHUNK_SIZE] for i in range(0, len(text), self.config.CHUNK_SIZE)]
        corrected_text = ""
        
        for chunk in chunks:
            response = openai.ChatCompletion.create(
                model=self.config.CORRECTION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un corrector ortográfico especializado. Corrige solo errores ortográficos manteniendo el significado y estructura original del texto. No cambies palabras, puntuación ni estructura, incluso si no tiene sentido. Mantén el formato y los saltos de línea."
                    },
                    {
                        "role": "user",
                        "content": f"Corrige la ortografía del siguiente texto manteniendo su estructura y formato:\n\n{chunk}"
                    }
                ],
                max_tokens=self.config.MAX_TOKENS
            )
            corrected_text += response.choices[0].message.content + "\n"
        
        return corrected_text

    def extract_junta_location(self, text: str) -> str:
        """Extrae la ubicación de la Junta Regional del texto"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en identificar la ubicación de Juntas Regionales de Calificación.
                    Extrae el departamento o ciudad donde se realizó la Junta Regional.
                    Devuelve solo el nombre del departamento o ciudad, sin texto adicional."""
                },
                {
                    "role": "user",
                    "content": f"Identifica el departamento o ciudad donde se realizó esta Junta Regional de Calificación:\n\n{text}"
                }
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    def extract_analysis_and_conclusions(self, text: str) -> str:
        """Extrae el análisis y conclusiones de la Junta Regional"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en identificar análisis y conclusiones de Juntas Regionales de Calificación.
                    Busca en el texto:
                    1. Primero, la sección específica llamada "ANÁLISIS Y CONCLUSIONES", que se ubica al final del acta, luego de la sección de "Fundamentos de derecho"
                    2. Luego busca la valoración del calificador y equipo interdisciplinario, esta se encuentra luego de la sección de "Concepto de rehabilitación"
                    3. También incluye la sección de "otros conceptos técnicos" si es relevante
                    
                    Extrae solo el texto de estas secciones concatendado uno debajo del otro, corrigiendo los errores de ortografía, sin modificar el formato original.
                    No incluyas conclusiones de otras entidades, solo las de la Junta Regional."""
                },
                {
                    "role": "user",
                    "content": f"Extrae el análisis y conclusiones de la Junta Regional del siguiente texto:\n\n{text}"
                }
            ],
            max_tokens=self.config.MAX_TOKENS
        )
        return response.choices[0].message.content.strip()

    def extract_medical_concepts(self, text: str) -> str:
        """Extrae los conceptos médicos del texto"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en identificar conceptos médicos en actas de Junta Regional de Calificación.
                    Busca en el texto:
                    1. La sección "CONCEPTOS MÉDICOS"
                    2. La sección "PRUEBAS ESPECÍFICAS"
                    
                    Extrae el texto exactamente como aparece en estas secciones, sin modificarlo.
                    Si no encuentras estas secciones, devuelve un mensaje indicando que no se encontraron conceptos médicos."""
                },
                {
                    "role": "user",
                    "content": f"Extrae los conceptos médicos del siguiente texto:\n\n{text}"
                }
            ],
            max_tokens=self.config.MAX_TOKENS
        )
        return response.choices[0].message.content.strip()

    def extract_recurring_name(self, text: str) -> str:
        """Extrae el nombre de la persona que interpone el recurso"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en identificar el nombre de la persona que interpone un recurso de reposición.
                    Busca en el texto el nombre de la persona que presenta el recurso.
                    Devuelve solo el nombre completo de la persona, sin texto adicional."""
                },
                {
                    "role": "user",
                    "content": f"Identifica el nombre de la persona que interpone el recurso de reposición en el siguiente texto:\n\n{text}"
                }
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    def extract_pcl_info(self, text: str) -> Dict:
        """Extrae toda la información relevante para el dictamen PCL"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en extraer información de dictámenes de PCL de Juntas Regionales de Calificación.
                    Extrae la siguiente información en formato JSON:
                    {
                        "ubicacion": "string",
                        "numero_dictamen": "string",
                        "fecha_dictamen": "string",
                        "diagnosticos": ["string"],
                        "deficiencia_total": "string",
                        "rol_laboral": "string",
                        "pcl_total": "string",
                        "origen": "string",
                        "fecha_estructuracion": "string",
                        "deficiencias_calificadas": [
                            {
                                "nombre": "string",
                                "porcentaje": "string",
                                "fuente": "string (formato: Tabla X.Y)"
                            }
                        ],
                        "analisis_conclusiones": "string",
                        "valoracion_calificador": "string",
                        "otros_conceptos": "string"
                    }
                    Para la fuente de las deficiencias calificadas, extrae específicamente:
                    - La tabla en formato "Tabla X.Y" (ejemplo: "Tabla 13.4")
                    Si algún campo no se encuentra, déjalo como null."""
                },
                {
                    "role": "user",
                    "content": f"Extrae la información del dictamen PCL del siguiente texto:\n\n{text}"
                }
            ],
            max_tokens=self.config.MAX_TOKENS
        )
        return json.loads(response.choices[0].message.content)

    def generate_pcl_template(self, pcl_info: Dict) -> str:
        """Genera la plantilla del dictamen PCL con la información extraída"""
        # Formatear deficiencias calificadas
        deficiencias_table = ""
        if pcl_info.get("deficiencias_calificadas"):
            deficiencias_table = "| Deficiencia | Porcentaje | Capítulo, Numeral, Literal, Tabla |\n"
            deficiencias_table += "|-------------|------------|----------------------------------|\n"
            for deficiencia in pcl_info["deficiencias_calificadas"]:
                deficiencias_table += f"| {deficiencia['nombre']} | {deficiencia['porcentaje']} | {deficiencia['fuente']} |\n"
        else:
            deficiencias_table = "No se pudo extraer con claridad la información completa de las deficiencias calificadas."

        # Formatear diagnósticos
        diagnosticos = "\n".join([f"{i+1}. {d}" for i, d in enumerate(pcl_info.get("diagnosticos", []))])

        # Combinar valoraciones
        valoraciones = f"{pcl_info.get('valoracion_calificador', '')}\n\n{pcl_info.get('otros_conceptos', '')}".strip()

        template = f"""Calificación Junta Regional de Calificación de Invalidez:

                        La Junta Regional de Calificación de Invalidez de {pcl_info.get('ubicacion', '')} mediante dictamen N° {pcl_info.get('numero_dictamen', '')} de fecha {pcl_info.get('fecha_dictamen', '')} establece:

                        DIAGNÓSTICO(S):
                        {diagnosticos}

                        DEFICIENCIAS: {pcl_info.get('deficiencia_total', '')}%
                        ROL LABORAL Y OTROS: {pcl_info.get('rol_laboral', '')}%
                        PCL TOTAL: {pcl_info.get('pcl_total', '')}%

                        ORIGEN: {pcl_info.get('origen', '')}

                        FECHA DE ESTRUCTURACIÓN: {pcl_info.get('fecha_estructuracion', '')}

                        La calificación de PCL emitida se desglosa así:

                        {deficiencias_table}

                        La Junta Regional de Calificación de Invalidez de {pcl_info.get('ubicacion', '')}, fundamenta su dictamen, especialmente, en los siguientes términos:

                        "{pcl_info.get('analisis_conclusiones', '')}

                        {valoraciones}"
                    """

        return template

    def process_recurring_text(self, text: str) -> str:
        """Procesa el texto del recurso de reposición"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en procesar textos de recursos de reposición.
                    Extrae únicamente el texto que fundamenta la motivación de la inconformidad.
                    Sigue estas reglas:
                    1. Elimina pies de página y referencias a leyes citadas textualmente
                    2. Mantén solo el texto que explica los argumentos y razones de la inconformidad
                    3. Aplica corrección ortográfica básica sin cambiar el significado
                    4. Si hay bloques en MAYÚSCULAS, conviértelos a minúsculas siguiendo reglas gramaticales
                    5. Mantén la estructura y formato del texto principal"""
                },
                {
                    "role": "user",
                    "content": f"Extrae solo el texto principal que fundamenta la motivación de inconformidad del siguiente recurso, eliminando pies de página y citas textuales de leyes:\n\n{text}"
                }
            ],
            max_tokens=self.config.MAX_TOKENS
        )
        return response.choices[0].message.content.strip()

    def extract_recurring_entity(self, text: str) -> str:
        """Extrae el nombre o entidad que presenta el recurso"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en identificar quién presenta un recurso de reposición.
                    Busca:
                    1. Si es una persona natural:
                       - Si es hombre: "El señor [Nombre completo]"
                       - Si es mujer: "La señora [Nombre completo]"
                       - Si es apoderado: "El apoderado del señor/señora [Nombre completo]"
                    2. Si es una entidad, identifica el tipo:
                       - Administradora de Riesgos Laborales (NOMBRE)
                       - Entidad Prestadora de Salud (NOMBRE)
                       - Administradora de Fondos Pensionales (NOMBRE)
                    Si no puedes determinar con certeza, devuelve "[Entidad no identificada]"
                    Devuelve solo el texto con el formato especificado."""
                },
                {
                    "role": "user",
                    "content": f"Identifica quién presenta este recurso de reposición:\n\n{text}"
                }
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    def generate_recurring_template(self, entity: str, text: str) -> str:
        """Genera la plantilla para el recurso de reposición"""
        return f"""Motivación de la inconformidad: {entity} manifiesta su inconformidad frente al dictamen con base en:

                "{text}"
                """

    def extract_first_opportunity_info(self, text: str) -> Dict:
        """Extrae toda la información relevante para la Calificación en primera oportunidad"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en extraer información de calificaciones en primera oportunidad.
                    Extrae la siguiente información en formato JSON:
                    {
                        "tipo_entidad": "string (EPS/ARL/AFP)",
                        "nombre_entidad": "string (en mayúsculas)",
                        "diagnosticos": [
                            {
                                "diagnostico": "string",
                                "diagnostico_especifico": "string",
                                "lateralidad": "string",
                                "origen": "string"
                            }
                        ],
                        "deficiencias": [
                            {
                                "nombre": "string",
                                "porcentaje": "string"
                            }
                        ],
                        "deficiencia_total": "string",
                        "rol_laboral": "string",
                        "pcl_total": "string",
                        "origen": "string",
                        "fecha_estructuracion": "string",
                        "conceptos_medicos": [
                            {
                                "especialidad": "string (ej: ortopedia, fisiatría)",
                                "concepto": "string (texto del concepto)",
                                "fecha": "string (fecha de la historia clínica)",
                                "nombre_historia": "string (nombre de la historia clínica)"
                            }
                        ],
                        "pruebas_especificas": [
                            {
                                "tipo": "string (ej: RNM, electromiografía)",
                                "resultado": "string (texto del resultado)",
                                "fecha": "string (fecha de la historia clínica)",
                                "nombre_historia": "string (nombre de la historia clínica)"
                            }
                        ]
                    }
                    Sigue estas reglas:
                    1. Para diagnósticos: combina diagnóstico + diagnóstico específico + lateralidad
                    2. Para deficiencias: extrae nombre y porcentaje total
                    3. Para entidad: identifica tipo (EPS/ARL/AFP) y nombre en mayúsculas
                    4. Para conceptos médicos: 
                       - Extrae la especialidad y el concepto completo
                       - Extrae la fecha de la historia clínica
                       - Extrae el nombre de la historia clínica
                    5. Para pruebas específicas:
                       - Extrae el tipo de prueba y su resultado
                       - Extrae la fecha de la historia clínica
                       - Extrae el nombre de la historia clínica
                    6. Si algún campo no se encuentra, déjalo como null
                    7. Si el diagnóstico contiene abreviaturas, como 'L4-L5', 'C3-C4', o similares, deben aparecer en mayúsculas."""
                },
                {
                    "role": "user",
                    "content": f"Extrae la información de la calificación en primera oportunidad del siguiente texto:\n\n{text}"
                }
            ],
            max_tokens=self.config.MAX_TOKENS
        )
        return json.loads(response.choices[0].message.content)

    def generate_first_opportunity_template(self, info: Dict) -> str:
        """Genera la plantilla para la calificación en primera oportunidad"""
        # Formatear deficiencias
        deficiencias = ", ".join([f"{d['nombre'].lower()} ({d['porcentaje']}%)" for d in info.get("deficiencias", [])])
        
        # Formatear diagnósticos
        diagnosticos = []
        for d in info.get("diagnosticos", []):
            diagnostico = d["diagnostico"].capitalize()
            if d.get("diagnostico_especifico"):
                diagnostico += f" {d['diagnostico_especifico'].lower()}"
            if d.get("lateralidad"):
                diagnostico += f" {d['lateralidad'].lower()}"
            diagnostico += f" como de origen {d['origen'].lower()}"
            diagnosticos.append(diagnostico)
        
        # Formatear conceptos médicos
        conceptos_medicos = []
        if info.get("conceptos_medicos"):
            for c in info["conceptos_medicos"]:
                concepto = f"Concepto de {c['especialidad'].lower()}"
                if c.get("fecha"):
                    concepto += f" del {c['fecha']}"
                concepto += f": {c['concepto']}"
                if c.get("nombre_historia"):
                    concepto += f"\nNombre de la historia clínica: {c['nombre_historia']}"
                conceptos_medicos.append(concepto)
        
        # Formatear pruebas específicas
        pruebas_especificas = []
        if info.get("pruebas_especificas"):
            for p in info["pruebas_especificas"]:
                prueba = f"{p['tipo']}"
                if p.get("fecha"):
                    prueba += f" del {p['fecha']}"
                prueba += f": {p['resultado']}"
                if p.get("nombre_historia"):
                    prueba += f"\nNombre de la historia clínica: {p['nombre_historia']}"
                pruebas_especificas.append(prueba)
        
        # Formatear tipo de entidad
        tipo_entidad = {
            "EPS": "Entidad Prestadora de Salud",
            "ARL": "Administradora de Riesgos Laborales",
            "AFP": "Administradora de Fondos Pensionales"
        }.get(info.get("tipo_entidad", ""), "Entidad")

        template = f"""Calificación en primera oportunidad:

La {tipo_entidad} {info.get('nombre_entidad', '')} le calificó una Pérdida de Capacidad Laboral (PCL) de {info.get('pcl_total', '')}%, de origen {info.get('origen', '').lower()}, con fecha de estructuración {info.get('fecha_estructuracion', '')}. La calificación de PCL emitida se desglosa así: Deficiencia: {info.get('deficiencia_total', '')}%, Rol laboral/ocupacional y otras áreas ocupacionales: {info.get('rol_laboral', '')}%. Las deficiencias calificadas fueron: {deficiencias}. Diagnósticos: {', '.join(diagnosticos)}."""

        # Agregar conceptos médicos si existen
        if conceptos_medicos:
            template += f"\n\nConceptos médicos:\n" + "\n".join(conceptos_medicos)

        # Agregar pruebas específicas si existen
        if pruebas_especificas:
            template += f"\n\nPruebas específicas:\n" + "\n".join(pruebas_especificas)

        return template

    def extract_first_opportunity_origin_info(self, text: str) -> Dict:
        """Extrae la información de determinación de origen en primera oportunidad"""
        response = openai.ChatCompletion.create(
            model=self.config.CORRECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un especialista en extraer información de determinación de origen en primera oportunidad.
                    Extrae la siguiente información en formato JSON:
                    {
                        "tipo_entidad": "string (debe ser exactamente 'EPS', 'ARL' o 'AFP')",
                        "nombre_entidad": "string (en MAYÚSCULAS)",
                        "diagnosticos": [
                            {
                                "nombre": "string (primera letra mayúscula, resto minúsculas)",
                                "lateralidad": "string (solo si aparece textualmente: derecho, izquierdo, bilateral)",
                                "origen": "string (debe ser exactamente 'Enfermedad común' o 'Enfermedad laboral')"
                            }
                        ],
                        "conceptos_medicos": [
                            {
                                "especialidad": "string (ej: ortopedia, fisiatría)",
                                "concepto": "string (texto del concepto)",
                                "fecha": "string (fecha de la historia clínica)",
                                "nombre_historia": "string (nombre de la historia clínica)"
                            }
                        ],
                        "pruebas_especificas": [
                            {
                                "tipo": "string (ej: RNM, electromiografía)",
                                "resultado": "string (texto del resultado)",
                                "fecha": "string (fecha de la historia clínica)",
                                "nombre_historia": "string (nombre de la historia clínica)"
                            }
                        ]
                    }
                    
                    Reglas importantes:
                    1. Para tipo_entidad: Identifica si es EPS, ARL o AFP basado en el texto
                    2. Para nombre_entidad: Extrae SOLO el nombre en MAYÚSCULAS
                    3. Para diagnósticos:
                       - Búscalos en la sección de diagnósticos y en las conclusiones
                       - Primera letra mayúscula, resto en minúsculas
                       - Incluye lateralidad SOLO si aparece textualmente
                       - El origen debe ser exactamente "Enfermedad común" o "Enfermedad laboral"
                    4. Para conceptos médicos:
                       - Extrae la especialidad y el concepto completo
                       - Extrae la fecha de la historia clínica
                       - Extrae el nombre de la historia clínica
                       - Busca en secciones como "CONCEPTOS MÉDICOS" o "CONCEPTO DE ESPECIALISTA"
                    5. Para pruebas específicas:
                       - Extrae el tipo de prueba y su resultado
                       - Extrae la fecha de la historia clínica
                       - Extrae el nombre de la historia clínica
                       - Busca en secciones como "PRUEBAS ESPECÍFICAS" o "EXÁMENES PARACLÍNICOS"
                    6. Si no puedes identificar algún campo con certeza, déjalo como null"""
                },
                {
                    "role": "user",
                    "content": f"Extrae la información de determinación de origen del siguiente texto:\n\n{text}"
                }
            ],
            max_tokens=self.config.MAX_TOKENS
        )
        return json.loads(response.choices[0].message.content)

    def generate_first_opportunity_origin_template(self, info: Dict) -> str:
        """Genera la plantilla para la determinación de origen en primera oportunidad"""
        if not info.get("tipo_entidad") or not info.get("nombre_entidad") or not info.get("diagnosticos"):
            return "No se pudo identificar con claridad la entidad calificadora o los diagnósticos del dictamen"

        # Mapear tipo de entidad a su nombre completo
        tipo_entidad_map = {
            "EPS": "Entidad Prestadora de Salud",
            "ARL": "Administradora de Riesgos Laborales",
            "AFP": "Administradora de Fondos Pensionales"
        }
        
        tipo_entidad = tipo_entidad_map.get(info["tipo_entidad"], "Entidad")
        
        # Agrupar diagnósticos por origen
        diagnosticos_por_origen = {}
        for diagnostico in info["diagnosticos"]:
            origen = diagnostico["origen"]
            if origen not in diagnosticos_por_origen:
                diagnosticos_por_origen[origen] = []
            
            # Construir nombre completo del diagnóstico
            nombre_diagnostico = diagnostico["nombre"]
            if diagnostico.get("lateralidad"):
                nombre_diagnostico += f" {diagnostico['lateralidad']}"
            
            diagnosticos_por_origen[origen].append(nombre_diagnostico)
        
        # Construir la lista de diagnósticos agrupados por origen
        partes_diagnosticos = []
        for origen, diagnosticos in diagnosticos_por_origen.items():
            diagnosticos_str = ", ".join(diagnosticos)
            partes_diagnosticos.append(f"{diagnosticos_str} como de origen {origen.lower()}")
        
        # Formatear conceptos médicos
        conceptos_medicos = []
        if info.get("conceptos_medicos"):
            for c in info["conceptos_medicos"]:
                concepto = f"Concepto de {c['especialidad'].lower()}"
                if c.get("fecha"):
                    concepto += f" del {c['fecha']}"
                concepto += f": {c['concepto']}"
                if c.get("nombre_historia"):
                    concepto += f"\nNombre de la historia clínica: {c['nombre_historia']}"
                conceptos_medicos.append(concepto)
        
        # Formatear pruebas específicas
        pruebas_especificas = []
        if info.get("pruebas_especificas"):
            for p in info["pruebas_especificas"]:
                prueba = f"{p['tipo']}"
                if p.get("fecha"):
                    prueba += f" del {p['fecha']}"
                prueba += f": {p['resultado']}"
                if p.get("nombre_historia"):
                    prueba += f"\nNombre de la historia clínica: {p['nombre_historia']}"
                pruebas_especificas.append(prueba)
        
        # Unir todo en la plantilla final
        template = f"""Calificación en primera oportunidad:

La {tipo_entidad} {info['nombre_entidad']} calificó las patologías: {'; '.join(partes_diagnosticos)}."""

        # Agregar conceptos médicos si existen
        if conceptos_medicos:
            template += f"\n\nConceptos médicos:\n" + "\n".join(conceptos_medicos)

        # Agregar pruebas específicas si existen
        if pruebas_especificas:
            template += f"\n\nPruebas específicas:\n" + "\n".join(pruebas_especificas)
        
        return template

class StreamlitUI:
    """Clase para manejar la interfaz de usuario de Streamlit"""
    
    def __init__(self, pdf_processor: PDFProcessor, openai_service: OpenAIService):
        self.pdf_processor = pdf_processor
        self.openai_service = openai_service
    
    def render(self):
        """Renderiza la interfaz de usuario"""
        try:
            # Configurar la página
            st.set_page_config(
                page_title="Procesamiento de Dictámenes",
                page_icon="📋",
                #layout="wide"
            )

            # Crear menú horizontal superior
            tab_pcl, tab_origen = st.tabs([
                "📊 Dictamen Pérdida de Capacidad Laboral (PCL)",
                "🏥 Dictamen Determinación de Origen"
            ])
            
            with tab_pcl:
                # Menú de opciones usando option_menu para PCL
                tipo_documento = option_menu(
                    menu_title=None,
                    options=["Primera Oportunidad", "Dictamen Junta Regional", "Recurso de Reposición"],
                    icons=["file-earmark-text", "clipboard2-check", "balance-scale"],
                    default_index=0,
                    orientation="horizontal",
                    key="menu_pcl",
                    styles={
                        "container": {"padding": "0px", "background-color": "#fafafa"},
                        "icon": {"color": "black", "font-size": "14px"},
                        "nav-link": {
                            "font-size": "18px",
                            "text-align": "center",
                            "margin": "0px",
                            "--hover-color": "#eee",
                            "color": "black"
                        },
                        "nav-link-selected": {
                            "background-color": "#ff4b6a",
                            "color": "white"
                        },
                    }
                )
                
                # Área principal de procesamiento
                if tipo_documento == "Primera Oportunidad":
                    st.markdown("### 📝 Procesamiento de Calificación en Primera Oportunidad")
                    uploaded_file_po = st.file_uploader("Sube el documento de Calificación en Primera Oportunidad", type=["pdf"], key="first_opportunity")
                    if uploaded_file_po:
                        if st.button("Procesar Documento", type="primary", key="btn_first_opportunity"):
                            with st.spinner("Procesando documento..."):
                                try:
                                    # Procesar el PDF
                                    texto_completo = self._process_pdf(uploaded_file_po, show_images=False)
                                    
                                    # Extraer información
                                    with st.spinner("Extrayendo información..."):
                                        info = self.openai_service.extract_first_opportunity_info(texto_completo)
                                    
                                    # Generar plantilla
                                    template = self.openai_service.generate_first_opportunity_template(info)
                                    
                                    # Mostrar resultado
                                    st.success("¡Plantilla generada exitosamente!")
                                    st.text_area("", template, height=400, key="first_opportunity_result")
                                    
                                    # Opción para copiar
                                    if st.button("Copiar al Portapapeles", key="btn_copy_first_opportunity"):
                                        st.write("Texto copiado al portapapeles")
                                        st.code(template, language=None)
                                        
                                except Exception as e:
                                    st.error("El documento no se pudo procesar correctamente debido a problemas de escaneo o calidad del archivo. Intenta subir una versión más legible.")
                    else:
                        st.info("Sube el documento de Calificación en Primera Oportunidad")
                
                elif tipo_documento == "Dictamen Junta Regional":
                    st.markdown("### 📋 Procesamiento de Dictamen de Junta Regional")
                    uploaded_file_junta = st.file_uploader("Sube el dictamen de Junta Regional de Calificación", type=["pdf"], key="junta_template")
                    if uploaded_file_junta:
                        if st.button("Procesar Dictamen", type="primary", key="btn_acta"):
                            with st.spinner("Procesando dictamen para generar plantilla..."):
                                try:
                                    # Procesar el PDF sin mostrar imágenes
                                    texto_completo = self._process_pdf(uploaded_file_junta, show_images=False)
                                    
                                    # Extraer información PCL
                                    with st.spinner("Extrayendo información del dictamen PCL..."):
                                        pcl_info = self.openai_service.extract_pcl_info(texto_completo)
                                    
                                    # Generar plantilla PCL
                                    template = self.openai_service.generate_pcl_template(pcl_info)
                                    
                                    # Mostrar resultado
                                    st.success("¡Plantilla generada exitosamente!")
                                    st.text_area("", template, height=400, key="acta_result")
                                    
                                    # Opción para copiar
                                    if st.button("Copiar Dictamen", key="btn_copy_acta"):
                                        st.write("Texto copiado al portapapeles")
                                        st.code(template, language=None)
                                        
                                except Exception as e:
                                    st.error("El documento no se pudo procesar correctamente debido a problemas de escaneo o calidad del archivo. Intenta subir una versión más legible.")
                    else:
                        st.info("Sube el Dictamen de Junta Regional de Calificación")
                
                else:  # Recurso de Reposición
                    st.markdown("### ⚖️ Procesamiento de Recurso de Reposición")
                    uploaded_recurring = st.file_uploader("Sube el recurso de reposición", type=["pdf"], key="recurring_template_standalone")
                    if uploaded_recurring:
                        if st.button("Procesar Recurso", type="primary", key="btn_recurring_standalone"):
                            with st.spinner("Procesando recurso de reposición..."):
                                try:
                                    # Procesar el PDF del recurso
                                    texto_recurring = self._process_pdf(uploaded_recurring, show_images=False)
                                    
                                    # Procesar el texto del recurso
                                    texto_procesado = self.openai_service.process_recurring_text(texto_recurring)
                                    
                                    # Extraer entidad que presenta el recurso
                                    entity = self.openai_service.extract_recurring_entity(texto_procesado)
                                    
                                    # Generar plantilla del recurso
                                    recurring_template = self.openai_service.generate_recurring_template(entity, texto_procesado)
                                    
                                    # Mostrar resultado del recurso
                                    st.success("¡Recurso de reposición procesado exitosamente!")
                                    st.text_area("", recurring_template, height=400, key="recurring_result_standalone")
                                    
                                    # Opción para copiar
                                    if st.button("Copiar al Portapapeles", key="btn_copy_recurring_standalone"):
                                        st.write("Texto copiado al portapapeles")
                                        st.code(recurring_template, language=None)
                                        
                                except Exception as e:
                                    st.error("El recurso de reposición no se pudo procesar correctamente debido a problemas de escaneo o calidad del archivo. Intenta subir una versión más legible.")
                    else:
                        st.info("Sube el recurso de reposición")
            
            with tab_origen:
                # Menú de opciones usando option_menu para Origen
                tipo_documento_origen = option_menu(
                    menu_title=None,
                    options=["Primera Oportunidad", "Dictamen Junta Regional", "Recurso de Reposición"],
                    icons=["file-earmark-text", "clipboard2-check", "balance-scale"],
                    default_index=0,
                    orientation="horizontal",
                    key="menu_origen",
                    styles={
                        "container": {"padding": "0px", "background-color": "#fafafa"},
                        "icon": {"color": "black", "font-size": "14px"},
                        "nav-link": {
                            "font-size": "18px",
                            "text-align": "center",
                            "margin": "0px",
                            "--hover-color": "#eee",
                            "color": "black"
                        },
                        "nav-link-selected": {
                            "background-color": "#135029",
                            "color": "white"
                        },
                    }
                )
                
                if tipo_documento_origen == "Primera Oportunidad":
                    st.markdown("### 📝 Determinación de Origen en Primera Oportunidad")
                    uploaded_file_origen = st.file_uploader("Sube el documento de Determinación de Origen", type=["pdf"], key="first_opportunity_origin")
                    if uploaded_file_origen:
                        if st.button("Procesar Documento", type="primary", key="btn_first_opportunity_origin"):
                            with st.spinner("Procesando documento..."):
                                try:
                                    # Procesar el PDF
                                    texto_completo = self._process_pdf(uploaded_file_origen, show_images=False)
                                    
                                    # Extraer información
                                    with st.spinner("Extrayendo información..."):
                                        info = self.openai_service.extract_first_opportunity_origin_info(texto_completo)
                                    
                                    # Generar plantilla
                                    template = self.openai_service.generate_first_opportunity_origin_template(info)
                                    
                                    # Mostrar resultado
                                    st.success("¡Plantilla generada exitosamente!")
                                    st.text_area("", template, height=400, key="first_opportunity_origin_result")
                                    
                                    # Opción para copiar
                                    if st.button("Copiar al Portapapeles", key="btn_copy_first_opportunity_origin"):
                                        st.write("Texto copiado al portapapeles")
                                        st.code(template, language=None)
                                        
                                except Exception as e:
                                    st.error("El documento no se pudo procesar correctamente debido a problemas de escaneo o calidad del archivo. Intenta subir una versión más legible.")
                    else:
                        st.info("Sube el documento de Determinación de Origen")
                
                elif tipo_documento_origen == "Dictamen Junta Regional":
                    st.markdown("### 📋 Dictamen de Junta Regional")
                    st.info("Funcionalidad en desarrollo")
                
                else:  # Recurso de Reposición
                    st.markdown("### ⚖️ Recurso de Reposición")
                    st.info("Funcionalidad en desarrollo")

        except Exception as e:
            st.error(f"Se produjo un error inesperado. Por favor, recarga la página. Error: {str(e)}")
            st.stop()

    def _process_pdf(self, uploaded_file, show_images: bool = False) -> str:
        """Procesa el PDF subido y retorna el texto completo"""
        try:
            # Convertir PDF a imágenes
            images = self.pdf_processor.convert_pdf_to_images(uploaded_file.read())
            texto_completo = ""
            
            # Procesar cada página
            progress_bar = st.progress(0)
            for i, image in enumerate(images):
                # Actualizar barra de progreso
                progress = (i + 1) / len(images)
                progress_bar.progress(progress)
                
                # Extraer y corregir texto de la página
                with st.spinner(f'Procesando página {i+1} de {len(images)}...'):
                    base64_img = self.pdf_processor.image_to_base64(image)
                    texto_pagina = self.openai_service.extract_text_from_image(base64_img)
                    texto_corregido = self.openai_service.correct_text(texto_pagina)
                    texto_completo += f"\n\nPágina {i+1}:\n{texto_corregido}"
            
            progress_bar.empty()
            return texto_completo
            
        except Exception as e:
            st.error(f"Error al procesar el PDF: {str(e)}")
            return ""

def main():
    """Función principal de la aplicación"""
    config = Config()
    pdf_processor = PDFProcessor()
    openai_service = OpenAIService(config)
    ui = StreamlitUI(pdf_processor, openai_service)
    ui.render()

if __name__ == "__main__":
    main()