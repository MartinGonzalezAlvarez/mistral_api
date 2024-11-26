import os
import json
import logging
import asyncio
from typing import Optional, AsyncGenerator

import httpx
import urllib3
import socket
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuraciones de red
socket.setdefaulttimeout(60)  # Timeout global de socket
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
LLM_SERVER_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:3335")

# Cliente HTTP configurado globalmente
client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=30.0,   # Timeout de conexión
        read=60.0,      # Timeout de lectura
        write=30.0,     # Timeout de escritura
        pool=30.0       # Timeout de pool
    ),
    headers={
        "Connection": "keep-alive",
        "Accept": "application/json"
    }
)

class Query(BaseModel):
    """
    Modelo para las solicitudes a la API.
    """
    prompt: str
    model: str = "mistral:7b"  # Modelo por defecto
    stream: Optional[bool] = True

async def robust_stream_handler(
    prompt: str,
    model: str = "mistral:7b",
    max_retries: int = 3
) -> AsyncGenerator[str, None]:
    """
    Genera texto de forma asíncrona con manejo robusto de streaming y reintento.
    """
    url = f"{LLM_SERVER_URL}/api/generate"
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

    for attempt in range(max_retries):
        try:
            logger.debug(f"Intento {attempt + 1}: Payload para streaming - {payload}")

            async with client.stream("POST", url, json=payload) as response:
                if response.status_code == 404:
                    logger.error(f"Endpoint no encontrado: {response.status_code}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Endpoint no encontrado en el servidor LLM: {response.status_code}"
                    )
                elif response.status_code != 200:
                    error_text = await response.text()
                    logger.error(f"Error de streaming: {response.status_code} - {error_text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Error al conectar con el servidor LLM: {response.status_code}"
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

                # Procesar cualquier contenido restante
                if buffer.strip():
                    try:
                        data = json.loads(buffer)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        pass

                break  # Salir del bucle de reintentos si el streaming es exitoso

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Intento {attempt + 1} fallido: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)  # Delay entre reintentos
            else:
                logger.error(f"Todos los intentos fallidos: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error de comunicación con el servidor LLM después de {max_retries} intentos"
                )

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

        return {"response": combined_response}

    except httpx.RequestError as e:
        logger.error(f"Error de comunicación: {e}")
        raise HTTPException(status_code=500, detail=f"Error de comunicación con el servidor LLM")

@app.post("/api/generate")
async def generate_text(query: Query):
    try:
        if query.stream:
            return StreamingResponse(
                robust_stream_handler(query.prompt, query.model),
                media_type="text/plain"
            )
        else:
            response = await get_generated_text(query.prompt, query.model)
            return JSONResponse(response)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": str(e.detail)}
        )
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Error interno del servidor"}
        )

# Cerrar el cliente HTTP al finalizar la aplicación
@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3335)
