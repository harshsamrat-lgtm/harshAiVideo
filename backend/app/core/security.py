"""
Authentication and security dependencies for API endpoints.
"""
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(header_key: str = Security(api_key_header)) -> str:
    """
    Validate incoming API Key for secure endpoints.
    In local development, allows requests if API_KEY is set to default or matches header.
    """
    if not header_key:
        if settings.APP_ENV == "development" and not settings.API_KEY:
            return "dev-key"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required API Key header (X-API-Key)"
        )
    
    if header_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return header_key
