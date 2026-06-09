from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory 
from app.memory.procedural import ProceduralMemory
from app.memory.consolidation import MemoryConsolidation

episodic_memory = EpisodicMemory()
semantic_memory = SemanticMemory()      
procedural_memory = ProceduralMemory()
memory_consolidation = MemoryConsolidation()

__all__ = ["episodic_memory", "semantic_memory", "procedural_memory", "memory_consolidation"]