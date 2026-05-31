from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory 

episodic_memory = EpisodicMemory()
semantic_memory = SemanticMemory()           

__all__ = ["episodic_memory", "semantic_memory"]