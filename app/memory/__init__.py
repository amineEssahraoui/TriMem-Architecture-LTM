from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory 
from app.memory.procedural import ProceduralMemory

episodic_memory = EpisodicMemory()
semantic_memory = SemanticMemory()      
procedural_memory = ProceduralMemory()     

__all__ = ["episodic_memory", "semantic_memory", "procedural_memory"]