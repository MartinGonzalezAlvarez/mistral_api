import os
import json
import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener la URL del servidor LLM desde una variable de entorno
LLM_SERVER_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:11434")  

class Query(BaseModel):
    """
    Modelo para las solicitudes a la API.
    """
    prompt: str
    model: str = "mistral:7b"  # Modelo por defecto
    stream: Optional[bool] = True

async def stream_generated_text(prompt: str, model: str = "mistral:7b"):
    """
    Genera texto de forma asíncrona usando Mistral con manejo robusto de streaming.
    """
    url = f"{LLM_SERVER_URL}/api/generate"
    
    # Configuración específica para Mistral
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
        }
    }
    
    logger.debug(f"Payload para streaming: {payload}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.text()
                    logger.error(f"Error de streaming: {response.status_code} - {error_text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Error al conectar con el servidor LLM: {response.status_code} - {error_text}"
                    )

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError as e:
                                logger.warning(f"Error decodificando JSON: {e}")
                                continue

                # Procesar cualquier contenido restante en el buffer
                if buffer.strip():
                    try:
                        data = json.loads(buffer)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        pass

        except httpx.RequestError as e:
            logger.error(f"Error de comunicación: {e}")
            raise HTTPException(status_code=500, detail=f"Error de comunicación con el servidor LLM: {e}")
        except Exception as e:
            logger.error(f"Error inesperado: {e}")
            raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")

async def get_generated_text(prompt: str, model: str = "mistral:7b"):
    """
    Obtiene el texto completo generado por Mistral.
    """
    url = f"{LLM_SERVER_URL}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
        }
    }
    
    logger.debug(f"Payload para texto completo: {payload}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()

            combined_response = ""
            for line in response.text.splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            combined_response += data["response"]
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decodificando JSON: {e}")
                        continue

            return {"response": combined_response}

        except httpx.RequestError as e:
            logger.error(f"Error de comunicación: {e}")
            raise HTTPException(status_code=500, detail=f"Error de comunicación con el servidor LLM: {e}")

@app.post("/api/generate")
async def generate_text(query: Query):
    try:
        if query.stream:
            return StreamingResponse(
                stream_generated_text(query.prompt, query.model),
                media_type="text/plain"
            )
        else:
            response = await get_generated_text(query.prompt, query.model)
            return JSONResponse(response)
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        return JSONResponse(
            status_code=e.status_code, 
            content={"detail": str(e.detail)}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"detail": f"Error interno del servidor: {str(e)}"}
        )

@app.post("/api/models/download")
async def download_model(llm_name: str = Body(..., embed=True)):
    url = f"{LLM_SERVER_URL}/api/pull"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"name": llm_name})
            response.raise_for_status()
            logger.info(f"Modelo {llm_name} descargado exitosamente")
            return {"message": f"Model {llm_name} downloaded successfully"}
    except httpx.RequestError as e:
        logger.error(f"Error descargando modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descargar el modelo: {e}")

@app.get("/api/models")
async def list_models():
    url = f"{LLM_SERVER_URL}/api/tags"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return {"models": response.json()["models"]}
    except httpx.RequestError as e:
        logger.error(f"Error obteniendo lista de modelos: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener la lista de modelos: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3335)