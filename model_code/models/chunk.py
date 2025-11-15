from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from model_code.core.database import Base

class Chunk(Base):
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True, index=True)
    chunk_name = Column(String(256), nullable=False)
    chunk_data = Column(String, nullable=False)
    file_id = Column(Integer, ForeignKey("law_files.id"), nullable=False)
    
    file = relationship("LawFile", back_populates="chunks")