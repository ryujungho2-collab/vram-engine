import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import torch
import random, time

STATUS_COLORS = {
    "NORMAL": "#00ff00",
    "WARNING_PURGE": "#ffaa00",
    "CRITICAL_ROLLBACK": "#ff0044",
}


class AutonomousGUIDashboard:
    """
    엔진 상태 모니터링 대시보드.

    retriever(예: interfaces/run_hybrid_demo.py의 LegacyRetrieverAdapter,
    `.index()`/`.query()`/`.documents`/`.doc_adj`/`.prev_entropy`를 가진
    아무 객체)를 넘기면 실제 하이브리드 검색 파이프라인을 그대로 시각화한다.
    retriever가 없으면 circuit_breaker/verifier만 가지고 독립적으로 동작하는
    데모 모드(합성 그래프)로 동작한다. (구조 변경 없음 — 기존 dashboard.py
    그대로 이식)
    """

    def __init__(self, root, iso_engine, verifier, circuit_breaker,
                 retriever=None, demo_queries=None, labels=None):
        self.root = root
        self.root.title("VRAM Autonomous Knowledge Engine - Dashboard")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e1e")
        self.step_count = 0
        self.entropy_history = []

        self.iso_engine = iso_engine
        self.verifier = verifier
        self.circuit_breaker = circuit_breaker
        self.retriever = retriever
        self.demo_queries = demo_queries or []
        self._query_cursor = 0

        if self.retriever is not None and getattr(self.retriever, "doc_adj", None) is not None:
            n_docs = len(self.retriever.documents)
            self.labels = labels or [f"Doc{i + 1}" for i in range(n_docs)]
            self.num_nodes = n_docs
            self.state_adj = self.retriever.doc_adj.clone()
            self.prev_entropy = self.retriever.prev_entropy
        else:
            self.labels = labels or ["Topology", "Algebra", "Quant", "Softmax", "Gumbel", "RL"]
            self.num_nodes = len(self.labels)
            self.state_adj = self._init_adjacency(self.num_nodes)
            self.prev_entropy = float(self.verifier.calc_entropy(self.state_adj).item())
            self.circuit_breaker.update_snapshot(self.state_adj, self.prev_entropy)

        self._build_layout()
        self.root.after(1000, self.update_loop)

    def _init_adjacency(self, n):
        adj = torch.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    w = random.uniform(0.5, 1.0)
                    adj[i, j] = w
                    adj[j, i] = w
        return adj

    def _build_layout(self):
        header = tk.Frame(self.root, bg="#1e1e1e")
        header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        tk.Label(header, text="AUTONOMOUS ENGINE MONITOR", fg="#00ffcc", bg="#1e1e1e", font=("Consolas", 14, "bold")).pack(side=tk.LEFT)
        self.status_label = tk.Label(header, text="STATUS: NORMAL", fg="#00ff00", bg="#2d2d2d", font=("Consolas", 11, "bold"), padx=10)
        self.status_label.pack(side=tk.RIGHT)

        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig, (self.ax_graph, self.ax_entropy) = plt.subplots(1, 2, figsize=(9, 4), facecolor='#1e1e1e')
        plt.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(self.root, height=7, bg="#121212", fg="#00ff00", font=("Consolas", 9))
        self.log_text.pack(fill=tk.X, padx=10, pady=5)

    def update_loop(self):
        self.step_count += 1

        if self.retriever is not None and self.demo_queries:
            self._step_retriever_mode()
        else:
            self._step_standalone_mode()

        self._redraw_graph()
        self.canvas.draw()
        self.root.after(1000, self.update_loop)

    def _step_retriever_mode(self):
        query = self.demo_queries[self._query_cursor % len(self.demo_queries)]
        self._query_cursor += 1

        result = self.retriever.query(query, top_k=1)
        self.state_adj = self.retriever.doc_adj.clone()
        entropy = result["entropy"]
        status = result["engine_status"]
        top = result["results"][0] if result["results"] else None

        self.entropy_history.append(entropy)
        self.status_label.config(text=f"STATUS: {status}", fg=STATUS_COLORS.get(status, "#ffffff"))

        top_desc = f"{top['document'][:24]}... (hybrid={top['hybrid_score']:.2f})" if top else "N/A"
        self.log_text.insert(
            tk.END,
            f"[{time.strftime('%H:%M:%S')}] Q: '{query}' -> {top_desc} | "
            f"H(G): {entropy:.4f} | {status}\n"
        )
        self.log_text.see(tk.END)
        self._draw_entropy_plot()

    def _step_standalone_mode(self):
        u, v = random.sample(range(self.num_nodes), 2)
        candidate_adj = self.state_adj.clone()
        candidate_adj[u, v] = random.uniform(0.0, 1.0)
        candidate_adj[v, u] = candidate_adj[u, v]

        is_valid, _delta_h, log_msg = self.verifier.verify_hypothesis(self.state_adj, u, v)
        current_entropy = float(self.verifier.calc_entropy(candidate_adj).item())

        healed_state, status = self.circuit_breaker.inspect_and_heal(
            current_entropy, self.prev_entropy, not is_valid, candidate_adj
        )
        self.state_adj = healed_state
        self.prev_entropy = current_entropy

        self.entropy_history.append(current_entropy)
        self.status_label.config(text=f"STATUS: {status}", fg=STATUS_COLORS.get(status, "#ffffff"))

        self.log_text.insert(
            tk.END,
            f"[{time.strftime('%H:%M:%S')}] Step #{self.step_count:03d} | "
            f"H(G): {current_entropy:.4f} | {status} | {log_msg}\n"
        )
        self.log_text.see(tk.END)
        self._draw_entropy_plot()

    def _redraw_graph(self):
        self.ax_graph.clear()
        self.ax_graph.set_facecolor('#1e1e1e')
        self.ax_graph.set_title("VRAM Topology", color='#00ffcc')
        G = nx.from_numpy_array(self.state_adj.detach().cpu().numpy())
        G.remove_edges_from(nx.selfloop_edges(G))
        nx.relabel_nodes(G, {i: lbl for i, lbl in enumerate(self.labels)}, copy=False)
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=self.step_count), ax=self.ax_graph, node_color='#007acc', font_color='white')
        self.ax_graph.axis('off')

    def _draw_entropy_plot(self):
        self.ax_entropy.clear()
        self.ax_entropy.set_facecolor('#1e1e1e')
        self.ax_entropy.set_title("Entropy H(G)", color='#00ffcc')
        self.ax_entropy.plot(self.entropy_history[-20:], color='#ff007f', marker='o')
