from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field, PostgresDsn, RedisDsn
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra='ignore')
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    # # Security
    # SECRET_KEY: str = "your-secret-key-here-change-in-production"
    # ALGORITHM: str = "HS256"
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = 300
    
    # # Redis
    # REDIS_URL: str
    
#   # --------------------------
#     # Email Configuration
#     # --------------------------
#     # Mailjet API (Recommended)
#     MAILJET_API_KEY: Optional[str] = Field(default=None)
#     MAILJET_SECRET_KEY: Optional[str] = Field(default=None)
#     MAILJET_SENDER_EMAIL: str = Field(default="no-reply@yourdomain.com")

#     # Email
#     SMTP_SERVER: str = "smtp.gmail.com"
#     SMTP_PORT: int = 587
#     SMTP_USERNAME: str = "farooqtalha4@gmail.com"
#     SMTP_PASSWORD: str = "gwgr xdzv lhoc eucq"
    
#     # CORS
#     ALLOWED_HOSTS: List[str] = ["*"]
    
#     # Frontend URL
#     FRONTEND_URL: str
    
    # Environment
    ENVIRONMENT: str = "development"


settings = Settings()