from sqlalchemy import Column, Integer, String, Boolean, LargeBinary as Bytea
from sqlalchemy.orm import relationship
from model_code.core.database import Base

class LawFile(Base):
    __tablename__ = 'law_files'

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(256), nullable=False)
    file_data = Column(Bytea, nullable=False)
    chunked = Column(Boolean, default=False)

    chunks = relationship("Chunk", back_populates="file")