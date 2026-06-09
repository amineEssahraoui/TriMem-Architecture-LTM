import os
import sys
import sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyvis.network import Network
from app.core.config import settings

def generate_graph(output: str = "tests/knowledge_graph.html"):
    db_path = settings.KNOWLEDGR_GRAPHE_PATH

    if not os.path.exists(db_path):
        print(f"[Error] Database not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, subject, relation, object FROM semantic_triplets")
    triplets = cursor.fetchall()
    conn.close()

    if not triplets:
        print("[Error] No triplets found.")
        return

    print(f"[Graph] Found {len(triplets)} triplets. Building graph...")

    # One color per user
    user_colors = {}
    palette = ["#4f46e5", "#059669", "#dc2626", "#d97706", "#7c3aed", "#0891b2"]
    unique_users = list(set(t[0] for t in triplets))
    for i, uid in enumerate(unique_users):
        user_colors[uid] = palette[i % len(palette)]

    net = Network(height="100vh", width="100%", bgcolor="#111827", font_color="white", directed=True)

    # Push nodes far apart
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.1,
        spring_length=300,
        spring_strength=0.01,
        damping=0.5
    )

    added_nodes = set()

    for uid, subject, relation, obj in triplets:
        color = user_colors[uid]
        subject_node = f"{uid}_{subject}"
        obj_node = f"{uid}_{obj}"

        # Subject node — big, shows user_id
        if subject_node not in added_nodes:
            net.add_node(
                subject_node,
                label=f"{uid}",
                title=f"User: {uid}",
                color=color,
                size=45,
                shape="dot",
                font={"size": 18, "color": "white", "bold": True}
            )
            added_nodes.add(subject_node)

        # Object node — medium, shows the attribute value
        if obj_node not in added_nodes:
            net.add_node(
                obj_node,
                label=obj,
                title=f"{relation}: {obj}",
                color="#cbd5e1",
                size=22,
                shape="dot",
                font={"size": 14, "color": "white"}
            )
            added_nodes.add(obj_node)

        net.add_edge(
            subject_node,
            obj_node,
            title=relation,
            label=relation,
            color={"color": color, "opacity": 0.8},
            width=2,
            font={"size": 11, "color": "#94a3b8", "strokeWidth": 0}
        )

    net.save_graph(output)
    print(f"[Graph] Saved at: {output}")
    print(f"[Graph] Users: {unique_users}")


if __name__ == "__main__":
    generate_graph()