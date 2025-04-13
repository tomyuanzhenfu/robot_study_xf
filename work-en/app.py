import streamlit as st
import json
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import HtmlFormatter
from streamlit.components.v1 import html
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch

# Initialize page configuration
st.set_page_config(layout="wide", page_title="Robot Capability Event Analysis System")

class NodeMapper:
    def __init__(self):
        self.real_to_mapped = {}
        self.mapped_to_real = {}
        self.next_id = 0
        self.mapping_file = "node_mapping.json"
        self.load_mapping()

    def load_mapping(self):
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r') as f:
                data = json.load(f)
                self.real_to_mapped = data.get("real_to_mapped", {})
                self.mapped_to_real = data.get("mapped_to_real", {})
                self.next_id = data.get("next_id", 0)

    def save_mapping(self):
        with open(self.mapping_file, 'w') as f:
            json.dump({
                "real_to_mapped": self.real_to_mapped,
                "mapped_to_real": self.mapped_to_real,
                "next_id": self.next_id
            }, f, indent=2)

    def map_node(self, real_name):
        if real_name not in self.real_to_mapped:
            mapped_id = self._generate_mapped_id()
            self.real_to_mapped[real_name] = mapped_id
            self.mapped_to_real[mapped_id] = real_name
            self.next_id += 1
            self.save_mapping()
        return self.real_to_mapped[real_name]

    def _generate_mapped_id(self):
        if self.next_id < 26:
            return chr(ord('A') + self.next_id)
        else:
            prefix = chr(ord('A') + (self.next_id // 26 - 1))
            suffix = chr(ord('A') + (self.next_id % 26))
            return f"{prefix}{suffix}"

    def get_real_name(self, mapped_id):
        return self.mapped_to_real.get(mapped_id, mapped_id)

    def get_node_type(self, mapped_id):
        real_name = self.get_real_name(mapped_id)
        return real_name.split(':')[0] if ':' in real_name else 'unknown'

node_mapper = NodeMapper()

def init_session():
    session_keys = {
        "json_data": None,
        "raw_json": "",
        "editor_key": 0,
        "file_processed": False,
        "show_paths": False,
        "path_results": {},
        "min_time": 0.0,
        "max_time": 100.0,
        "graph_layout": "Hierarchical Layout",
        "self_implemented_nodes": set(),
        "node_connectivity": {},
        "show_real_names": False,
        "show_mapping_table": False
    }
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = session_keys[key]

def process_capability_data(raw_data):
    if not isinstance(raw_data, list):
        raise ValueError("Input data must be a JSON array")

    nodes = set()
    edges = []
    timestamps = []
    capability_provider_pairs = defaultdict(set)

    for event in raw_data:
        if not isinstance(event, dict):
            continue

        msg = event.get("msg", {})
        source = msg.get("source", {})
        target = msg.get("target", {})

        stamp = msg.get("header", {}).get("stamp", {})
        event_time = stamp.get("secs", 0) + stamp.get("nsecs", 0) * 1e-9
        timestamps.append(event_time)

        if source.get("capability") and source.get("provider"):
            capability_provider_pairs[source["capability"]].add(source["provider"])

        if source.get("capability"):
            real_name = f"capability:{source['capability']}"
            mapped_id = node_mapper.map_node(real_name)
            nodes.add((mapped_id, real_name, "capability"))
        if source.get("provider"):
            real_name = f"provider:{source['provider']}"
            mapped_id = node_mapper.map_node(real_name)
            nodes.add((mapped_id, real_name, "provider"))

        if target.get("capability"):
            real_name = f"capability:{target['capability']}"
            mapped_id = node_mapper.map_node(real_name)
            nodes.add((mapped_id, real_name, "capability"))
        if target.get("provider"):
            real_name = f"provider:{target['provider']}"
            mapped_id = node_mapper.map_node(real_name)
            nodes.add((mapped_id, real_name, "provider"))

    self_implemented = set()
    for cap, providers in capability_provider_pairs.items():
        if cap in providers:
            real_name = f"capability:{cap}"
            mapped_id = node_mapper.map_node(real_name)
            self_implemented.add(mapped_id)
            real_name = f"provider:{cap}"
            mapped_id = node_mapper.map_node(real_name)
            self_implemented.add(mapped_id)
    st.session_state.self_implemented_nodes = self_implemented

    min_timestamp = min(timestamps) if timestamps else 0

    node_list = []
    node_colors = {
        "capability": "#4B8BBE",
        "provider": "#FFA500",
        "unknown": "#95a5a6"
    }
    for mapped_id, real_name, node_type in nodes:
        node_color = "#9b59b6" if mapped_id in self_implemented else node_colors.get(node_type, "#95a5a6")
        node_list.append({
            "id": mapped_id,
            "real_name": real_name,
            "label": mapped_id,
            "type": node_type,
            "color": node_color,
            "size": 1500 if node_type == "capability" else 1200,
            "shape": "d" if mapped_id in self_implemented else "o"
        })

    for event in raw_data:
        if not isinstance(event, dict):
            continue

        msg = event.get("msg", {})
        source = msg.get("source", {})
        target = msg.get("target", {})

        stamp = msg.get("header", {}).get("stamp", {})
        event_time = stamp.get("secs", 0) + stamp.get("nsecs", 0) * 1e-9
        rel_time = event_time - min_timestamp

        source_id = None
        if source.get("capability"):
            real_name = f"capability:{source['capability']}"
            source_id = node_mapper.map_node(real_name)
        elif source.get("provider"):
            real_name = f"provider:{source['provider']}"
            source_id = node_mapper.map_node(real_name)

        target_id = None
        if target.get("capability"):
            real_name = f"capability:{target['capability']}"
            target_id = node_mapper.map_node(real_name)
        elif target.get("provider"):
            real_name = f"provider:{target['provider']}"
            target_id = node_mapper.map_node(real_name)

        if source_id and target_id:
            edges.append({
                "source": source_id,
                "target": target_id,
                "label": msg.get("text", "")[:50] + "..." if len(msg.get("text", "")) > 50 else msg.get("text", ""),
                "time": round(rel_time, 3),
                "color": "#2ecc71" if rel_time < 5 else "#e74c3c",
                "width": 1.0 + rel_time / 5,
                "weight": rel_time
            })

    max_time = max([e["time"] for e in edges]) if edges else 100.0

    return {
        "nodes": node_list,
        "edges": edges,
        "min_timestamp": min_timestamp,
        "max_time": max_time
    }

def json_editor_component():
    formatter = HtmlFormatter(style="colorful", full=True, cssclass="highlight")

    col1, col2 = st.columns([1, 1])
    with col1:
        new_json = st.text_area(
            "📝 Edit JSON Content",
            value=st.session_state.raw_json,
            height=600,
            key=f"editor_{st.session_state.editor_key}",
            help="Click save button after modification to update charts"
        )

        if st.button("💾 Save Changes", type="primary"):
            try:
                if "uploaded_file" in st.session_state:
                    del st.session_state.uploaded_file
                    st.session_state.file_processed = False

                parsed = json.loads(new_json)
                processed = process_capability_data(parsed)
                st.session_state.json_data = processed
                st.session_state.raw_json = json.dumps(
                    parsed,
                    indent=2,
                    ensure_ascii=False
                )
                st.session_state.max_time = processed["max_time"]
                st.session_state.editor_key += 1
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {str(e)}")

    with col2:
        st.markdown("Live Preview")
        if st.session_state.raw_json:
            html_code = highlight(
                st.session_state.raw_json,
                JsonLexer(),
                formatter
            )
            html(f'<div class="highlight">{html_code}</div>', height=600, scrolling=True)

def build_network_graph(data):
    G = nx.DiGraph()

    for node in data["nodes"]:
        G.add_node(node["id"],
                   label=node["label"],
                   real_name=node["real_name"],
                   node_type=node["type"],
                   color=node["color"],
                   size=node["size"],
                   shape=node.get("shape", "o"),
                   subset=int(node["id"].split(":")[0] == "provider"))

    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            label=edge["label"],
            time=edge["time"],
            weight=edge["time"],
            color=edge["color"],
            width=edge["width"]
        )
    return G

def generate_network_graph(data):
    try:
        G = build_network_graph(data)

        if st.session_state.graph_layout == "Hierarchical Layout":
            pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal")
        else:
            pos = nx.spring_layout(G, k=1.5, iterations=100)

        fig, ax = plt.subplots(figsize=(14, 10), dpi=120)
        title = "Capability Event Flow" + (" (Show Real Names)" if st.session_state.show_real_names else "")
        plt.title(title, pad=20, fontsize=16)

        regular_nodes = [n for n in G.nodes if n not in st.session_state.self_implemented_nodes]
        self_implemented_nodes = [n for n in G.nodes if n in st.session_state.self_implemented_nodes]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular_nodes,
            node_size=[G.nodes[n]['size'] for n in regular_nodes],
            node_color=[G.nodes[n]['color'] for n in regular_nodes],
            edgecolors="white",
            linewidths=1.5,
            node_shape="o",
            ax=ax
        )

        if self_implemented_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=self_implemented_nodes,
                node_size=[G.nodes[n]['size'] * 1.2 for n in self_implemented_nodes],
                node_color=[G.nodes[n]['color'] for n in self_implemented_nodes],
                edgecolors="white",
                linewidths=2,
                node_shape="d",
                ax=ax
            )

        for u, v, data in G.edges(data=True):
            arrow = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                arrowstyle="->",
                color=data['color'],
                mutation_scale=20,
                linewidth=data['width'] * 0.8,
                connectionstyle="arc3,rad=0.2",
                zorder=3
            )
            ax.add_patch(arrow)

        labels = {}
        for n in G.nodes:
            if st.session_state.show_real_names:
                real_name = G.nodes[n]['real_name']
                node_type = real_name.split(':')[0]
                labels[n] = f"{node_mapper.map_node(real_name)}\n({real_name.split(':')[1]})"
            else:
                labels[n] = G.nodes[n]['label']

        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=10,
            font_family='sans-serif',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5),
            ax=ax
        )

        legend_elements = [
            plt.Line2D([0], [0], marker='d', color='w', label='Self-Implemented',
                       markerfacecolor='#9b59b6', markersize=10),
            plt.Line2D([0], [0], color='#2ecc71', lw=2, label='Early Event (t<5s)'),
            plt.Line2D([0], [0], color='#e74c3c', lw=2, label='Late Event (t≥5s)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.axis("off")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Graph generation failed: {str(e)}")
        return None

def plot_path(G, path, title):
    path_edges = list(zip(path[:-1], path[1:]))

    plt.figure(figsize=(8, 4))

    if st.session_state.graph_layout == "Hierarchical Layout":
        pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal")
    else:
        pos = nx.spring_layout(G, k=0.8, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgray')

    path_colors = []
    for n in path:
        if n in st.session_state.self_implemented_nodes:
            path_colors.append('#9b59b6')
        elif 'capability' in node_mapper.get_node_type(n):
            path_colors.append('#4B8BBE')
        else:
            path_colors.append('#FFA500')

    nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=500, node_color=path_colors)

    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, arrows=True)

    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color='red', width=2, arrows=True)

    labels = {}
    for n in path:
        if st.session_state.show_real_names:
            real_name = G.nodes[n]['real_name']
            labels[n] = f"{n}\n({real_name.split(':')[1]})"
        else:
            labels[n] = n

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

def path_analysis_panel():
    st.header("⏱️ Path Analysis")

    st.checkbox("Show Real Names", key="show_real_names",
                help="Toggle between mapped IDs and original names")

    if not st.session_state.json_data:
        st.warning("Please load data first")
        return

    G = build_network_graph(st.session_state.json_data)

    all_nodes = [n["id"] for n in st.session_state.json_data["nodes"]]
    node_labels = {n["id"]: n["real_name"] for n in st.session_state.json_data["nodes"]}

    st.session_state.graph_layout = st.radio(
        "Select Layout",
        ["Force-Directed Layout", "Hierarchical Layout"],
        index=0,
        horizontal=True
    )

    st.subheader("Time Range Filter")
    min_time, max_time = st.slider(
        "Select analysis time range (seconds)",
        min_value=0.0,
        max_value=st.session_state.max_time,
        value=(0.0, st.session_state.max_time),
        step=0.1,
        format="%.1f"
    )

    st.subheader("Node Selection")
    col1, col2 = st.columns(2)
    with col1:
        start_node = st.selectbox(
            "Start Node",
            options=all_nodes,
            index=0,
            format_func=lambda x: f"{x} ({node_mapper.get_node_type(x)})",
            key="path_start"
        )
    with col2:
        end_node = st.selectbox(
            "End Node",
            options=all_nodes,
            index=min(1, len(all_nodes) - 1),
            format_func=lambda x: f"{x} ({node_mapper.get_node_type(x)})",
            key="path_end"
        )

    if st.button("🔍 Calculate Optimal Path", type="primary"):
        st.session_state.show_paths = True
        try:
            filtered_edges = [
                (u, v) for u, v, d in G.edges(data=True)
                if min_time <= d['time'] <= max_time
            ]
            subgraph = G.edge_subgraph(filtered_edges)

            if not nx.has_path(subgraph, start_node, end_node):
                error_report = get_node_connectivity_report(
                    G, start_node, end_node, (min_time, max_time)
                )
                st.markdown(f'<div class="path-error">{error_report}</div>', unsafe_allow_html=True)
                return

            time_path = nx.shortest_path(subgraph, start_node, end_node, weight='time')
            time_cost = sum(G.edges[u, v]['time'] for u, v in zip(time_path[:-1], time_path[1:]))

            hop_path = nx.shortest_path(subgraph, start_node, end_node)
            hop_cost = len(hop_path) - 1

            for u, v in subgraph.edges():
                subgraph.edges[u, v]['mixed_weight'] = subgraph.edges[u, v]['time'] + 0.3
            mixed_path = nx.shortest_path(subgraph, start_node, end_node, weight='mixed_weight')
            mixed_cost = sum(G.edges[u, v]['time'] for u, v in zip(mixed_path[:-1], mixed_path[1:]))

            st.session_state.path_results = {
                "time_path": (time_path, time_cost),
                "hop_path": (hop_path, hop_cost),
                "mixed_path": (mixed_path, mixed_cost)
            }

        except Exception as e:
            st.error(f"Path calculation error: {str(e)}")

    if st.session_state.show_paths and 'path_results' in st.session_state:
        st.subheader("📊 Path Analysis Results")

        tab1, tab2, tab3 = st.tabs(["⏱️ Shortest Time", "↔️ Minimum Hops", "⚖️ Balanced Path"])

        with tab1:
            path, cost = st.session_state.path_results["time_path"]
            display_path = " → ".join([
                f"{node}\n({node_labels[node].split(':')[1]})"
                if st.session_state.show_real_names
                else node
                for node in path
            ])
            st.success(f"""
                Shortest Time Path (Total time: {cost:.2f}s)
                ```
                {display_path}
                ```
            """)
            plot_path(G, path, "Time-Optimal Path")

        with tab2:
            path, cost = st.session_state.path_results["hop_path"]
            display_path = " → ".join([
                f"{node}\n({node_labels[node].split(':')[1]})"
                if st.session_state.show_real_names
                else node
                for node in path
            ])
            st.info(f"""
                Minimum Hops Path (Hops: {cost})
                ```
                {display_path}
                ```
            """)
            plot_path(G, path, "Hops-Optimal Path")

        with tab3:
            path, cost = st.session_state.path_results["mixed_path"]
            display_path = " → ".join([
                f"{node}\n({node_labels[node].split(':')[1]})"
                if st.session_state.show_real_names
                else node
                for node in path
            ])
            st.warning(f"""
                Balanced Path (Time: {cost:.2f}s)
                ```
                {display_path}
                ```
            """)
            plot_path(G, path, "Balanced Path")

def main():
    init_session()

    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🤖 Robot Capability Event Analysis System</h1>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload JSON File", type=["json"], key="uploaded_file")

    if uploaded_file and not st.session_state.file_processed:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            raw_data = json.loads(content)
            processed_data = process_capability_data(raw_data)

            st.session_state.json_data = processed_data
            st.session_state.raw_json = json.dumps(
                raw_data,
                indent=2,
                ensure_ascii=False
            )
            st.session_state.max_time = processed_data["max_time"]
            st.session_state.file_processed = True
            st.success("File processed successfully!")
        except Exception as e:
            st.error(f"File processing failed: {str(e)}")

    col1, col2, col3 = st.columns([2.5, 4, 1.5])

    with col1:
        st.header("📂 Data Input")
        if st.session_state.raw_json:
            json_editor_component()
        else:
            st.info("Please upload a JSON file or edit content directly")

    with col2:
        st.header("🌐 Network Visualization")
        if st.session_state.json_data:
            fig = generate_network_graph(st.session_state.json_data)
            if fig:
                st.pyplot(fig)
        else:
            st.info("Waiting for data input...")

    with col3:
        if st.session_state.json_data:
            path_analysis_panel()
        else:
            st.info("Please upload data first")

if __name__ == "__main__":
    main()