import urllib.request
import json
import time

API_URL = "http://127.0.0.1:8000/chat"
WAIT = 90  # seconds for background tasks to complete

def send_message(user_id: str, message: str, wait: int = WAIT):
    print(f"\n[{user_id.upper()}] >>> {message}")
    data = json.dumps({"user_id": user_id, "message": message}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"[TRIMEM]  >>> {result['response']}")
    except Exception as e:
        print(f"[ERROR] {e}")
    print(f"(waiting {wait}s...)")
    time.sleep(wait)

if __name__ == "__main__":

    print("\n" + "="*70)
    print("TRIMEM — FULL MEMORY INTEGRATION TEST")
    print("3 Users · Episodic · Semantic · Procedural")
    print("="*70)

    # ─────────────────────────────────────────────────────────────
    # BLOCK 1 — IDENTITY & SEMANTIC SEEDING
    # ─────────────────────────────────────────────────────────────
    print("\n── BLOCK 1: Identity & Semantic Seeding ──")

    # Episodic + Semantic: name, OS, project, vault code
    send_message("user_amine", "Hey, I'm Amine. I strictly use Linux, I'm building TriMem, and my secret vault code is 8080.")

    # Episodic + Semantic: name, OS, profession
    send_message("user_sarah", "Hi! I'm Sarah, I use macOS for UI/UX design work.")

    # Episodic + Semantic: name, role, tools
    send_message("user_alex", "Hello, I'm Alex. I'm a project manager. I work with Python and Jira daily.")

    # ─────────────────────────────────────────────────────────────
    # BLOCK 2 — PROCEDURAL SEEDING (Behavioral Rules)
    # ─────────────────────────────────────────────────────────────
    print("\n── BLOCK 2: Procedural Seeding ──")

    # Procedural: global language rule
    send_message("user_amine", "Always respond to me in French, no matter what.")

    # Procedural: task-specific coding rule
    send_message("user_amine", "When you write code for me, always use functional programming, no classes.")

    # Procedural: global formatting rule
    send_message("user_sarah", "Always use bullet points in your responses, I find it easier to read.")

    # Procedural: task-specific sprint rule
    send_message("user_alex", "When I ask for a sprint update, always output a Markdown table with: Task, Status, Blocker.")

    # ─────────────────────────────────────────────────────────────
    # BLOCK 3 — SEMANTIC RETRIEVAL (Facts Ground Truth)
    # ─────────────────────────────────────────────────────────────
    print("\n── BLOCK 3: Semantic Retrieval ──")

    # Must recall: Linux, TriMem, vault code 8080
    send_message("user_amine", "Quick check — what OS do I use, what am I building, and do you have my vault code?")

    # Must recall: macOS, design
    send_message("user_sarah", "Remind me, what do you know about my work setup?")

    # Must recall: project manager, Python, Jira
    send_message("user_alex", "What do you know about my role and tools?")

    # ─────────────────────────────────────────────────────────────
    # BLOCK 4 — PROCEDURAL ENFORCEMENT
    # ─────────────────────────────────────────────────────────────
    print("\n── BLOCK 4: Procedural Enforcement ──")

    # Must reply in French + use functional programming style
    send_message("user_amine", "Write me a Python function to calculate the factorial of a number.")

    # Must use bullet points
    send_message("user_sarah", "What are the main principles of good UI design?")

    # Must output Markdown table with Task / Status / Blocker
    send_message("user_alex", "Give me a sprint update based on: finished login page, blocked on API docs, next is dashboard.")

    # ─────────────────────────────────────────────────────────────
    # BLOCK 5 — SEMANTIC OVERWRITE (1-to-1 update)
    # ─────────────────────────────────────────────────────────────
    print("\n── BLOCK 5: Semantic Overwrite ──")

    # Vault code update — must replace 8080 with 9999
    send_message("user_amine", "I changed my vault code. The new one is 9999, forget 8080.")

    # OS update — must replace macOS with Ubuntu
    send_message("user_sarah", "I switched my main machine to Ubuntu Linux today.")

    # Role update
    send_message("user_alex", "I just got promoted, I'm now a Senior Engineering Manager.")

    # ─────────────────────────────────────────────────────────────
    # BLOCK 6 — POST-OVERWRITE VERIFICATION
    # ─────────────────────────────────────────────────────────────
    print("\n── BLOCK 6: Post-Overwrite Verification ──")

    # Must say 9999, NOT 8080
    send_message("user_amine", "What is my current vault code?")

    # Must say Ubuntu, NOT macOS
    send_message("user_sarah", "What OS am I on now?")

    # Must say Senior Engineering Manager
    send_message("user_alex", "What's my current role?")

    # ─────────────────────────────────────────────────────────────
    # BLOCK 7 — USER ISOLATION (Privacy Check)
    # ─────────────────────────────────────────────────────────────
    print("\n── BLOCK 7: User Isolation ──")

    # Must NOT know Amine's code or Sarah's OS
    send_message("user_alex", "Do you know what vault code Amine uses? And what OS does Sarah have?")

    # Must NOT know Alex's role
    send_message("user_sarah", "What is Alex's job title?")

    # ─────────────────────────────────────────────────────────────
    # BLOCK 8 — EPISODIC CONTINUITY
    # ─────────────────────────────────────────────────────────────
    print("\n── BLOCK 8: Episodic Continuity ──")

    # Must reference the earlier factorial function conversation
    send_message("user_amine", "Can you improve the function you wrote for me earlier?")

    # Must reference the UI design discussion
    send_message("user_sarah", "Based on what we discussed, how would you apply those principles to a mobile app?")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)