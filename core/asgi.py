import os
from typing import List

from django.core.asgi import get_asgi_application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_ia.main import fastapi_app

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

django_app = get_asgi_application()


def _get_allowed_origins() -> List[str]:
	raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
	if not raw_origins.strip():
		return [
			"http://localhost:3000",
			"http://localhost:5173",
			"http://127.0.0.1:3000",
			"http://127.0.0.1:5173",
		]
	return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

# Creamos la app principal
app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=_get_allowed_origins(),
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Montamos FastAPI en /ai
app.mount("/ai", fastapi_app)

# Montamos Django en la raíz
app.mount("/", django_app)