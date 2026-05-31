import os
import sys
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.memory.semantic import semantic_memory
from pyvis.network import Network

def run_extraction_test(user_id: str):
    """Simulates a sequence of user messages to stress-test the Semantic Knowledge Graph."""
    print("\n--- 1. Running Groq Fact Extraction Test Suite ---")
    
    test_messages = [
        # TEST 1: The Baseline Identity (1-to-1 & 1-to-N mix)
        "Hi assistant, my name is Amine. I currently live in Morocco, and I speak Arabic, French, and English.",
        
        # TEST 2: Academic & Ambitions (1-to-1 & 1-to-N)
        "I am an engineering student studying at ENSAM Meknès. My ultimate goal is to pursue a PhD in France around 2027.",
        
        # TEST 3: Technical Stack (Heavy 1-to-N appending)
        "For my development environment, I strictly use Linux. I build AI architectures with PyTorch and Python, and I have strong skills in RAG and Large Language Models.",
        
        # TEST 4: Hobbies & Interests (Testing LIKES and INTERESTED_IN)
        "When I'm not coding, I really like playing chess and football. I'm also deeply interested in philosophy, epistemology, and the history of science.",
        
        # TEST 5: The Professional Role (1-to-1 insertion)
        "I just started a new position. My role is now NLP Chatbot Intern at JESA.",
        
        # TEST 6: The 1-to-1 Overwrite Test (Role & OS Change)
        "Big update! I got promoted. My role is now Project Manager at NeuraDeepAI Startup. Oh, and I switched my machine, so I use Windows now.",
        
        # TEST 7: The Security Code (1-to-1 insertion)
        "Hey assistant, please remember that my server access code is 1234.",
        
        # TEST 8: The Security Code Overwrite Test (1-to-1 update)
        "Actually, that old code was compromised. Change my server access code to 9999 immediately.",
        
        # TEST 9: The Technical Expansion (1-to-N appending without losing old data)
        "Lately, I've been diving deep into high-performance backend stuff. I'm actively adding C++, CUDA, and Docker to my toolkit.",
        
        # TEST 10: Multi-Entity Project Tracking (Complex NLP parsing)
        "Right now, my main focus is developing a MARL engine called Stealth Swarm, but I'm also the Technical Lead organizing the CyberIA CTF competition.",
        
        # TEST 11: Future Goals & Learning (1-to-N appending)
        "My next big objective is to improve my English and get an IELTS 7.0 band score.",
        
        # TEST 12: The Restraint Test (Should extract little to nothing)
        "Just wanted to chat today. I'm feeling pretty good about how this TriMem architecture is turning out, we are making great progress!"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n[Test {i}/12] Processing: '{msg[:50]}...'")
        
        # Using a unique source_id for each test to simulate different conversation turns
        semantic_memory.extract_and_save_facts(
            user_id=user_id, 
            text=msg, 
            source_id=f"test_interaction_{i:03d}"
        )
    
    

def generate_graph_visualization(user_id: str, output_filename: str = "tests/graph_output.html"):
    """Reads the extracted facts from the semantic database and plots an interactive HTML graph."""
    print("\n 2. Generating Interactive Knowledge Graph ")
    
    sqlite_path = os.path.join(settings.KNOWLEDGR_GRAPHE_PATH)
    
    if not os.path.exists(sqlite_path):
        print(f"Error: Semantic database not found at {sqlite_path}. Run extraction first.")
        return

    # Connect and pull triplets
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute('SELECT subject, relation, object FROM semantic_triplets WHERE user_id = ?', (user_id,))
    triplets = cursor.fetchall()
    conn.close()
    
    if not triplets:
        print(f"No graph data found in the database for user: {user_id}")
        return
        
    print(f"Found {len(triplets)} triplets in database. Building PyVis Network...")
    
    # Initialize a sleek dark-themed interactive physics canvas
    net = Network(height='750px', width='100%', bgcolor='#1a1a1a', font_color='white', directed=True)
    
    # Populate the canvas with extracted entities and relational edges
    for subject, relation, obj in triplets:
        # Subject node (Central User Node)
        net.add_node(subject, label=subject, color="#4f46e5", shape="dot", size=25)
        # Object node (Target Entities)
        net.add_node(obj, label=obj, color="#06b6d4", shape="dot", size=18)
        # Directed edge connecting them
        net.add_edge(subject, obj, title=relation, label=relation, color="#94a3b8", width=2)
        
    # Configure smooth node physics configuration
    net.force_atlas_2based(spring_length=160, damping=0.4)
    
    # Render and save the interactive HTML file
    net.save_graph(output_filename)
    print(f"Success! Interactive graph generated at: {output_filename}")
    print("Open this HTML file directly in any browser to inspect the graph structure.")

if __name__ == "__main__":
    # Define a clean test user profile identifier
    TEST_USER = "amine_test_user"
    
    # Execute the end-to-end integration test
    run_extraction_test(user_id=TEST_USER)
    generate_graph_visualization(user_id=TEST_USER)