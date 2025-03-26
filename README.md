# JNCI - Procesamiento de Dictámenes

Aplicación para procesar y analizar dictámenes de pérdida de capacidad laboral y determinación de origen.

## Requisitos

- Python 3.8+
- Poppler (instalado automáticamente en Streamlit Cloud)

## Instalación Local

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/JNCI.git
cd JNCI
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Instalar Poppler:
- Windows: Descargar e instalar desde [poppler releases](http://blog.alivate.com.au/poppler-windows/)
- Linux: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

5. Crear archivo `.env` con tu API key de OpenAI:
```
OPENAI_API_KEY=tu-api-key
```

6. Ejecutar la aplicación:
```bash
streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Subir el código a GitHub
2. Conectar el repositorio con Streamlit Cloud
3. Configurar las variables de entorno en Streamlit Cloud:
   - OPENAI_API_KEY

## Estructura del Proyecto

```
JNCI/
├── .streamlit/
│   └── config.toml
├── app.py
├── requirements.txt
├── packages.txt
└── README.md
```

## Características

- Procesamiento de PDFs
- Extracción de texto usando OCR
- Análisis de dictámenes de PCL
- Determinación de origen
- Generación de plantillas