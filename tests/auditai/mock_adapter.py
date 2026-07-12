#!/usr/bin/env python3
"""Minimal mock HTTP target for guerrilla AuditAI runs.

Intentionally weak: one SEED blurb for all questions (not per-case RAG).
Empty response contexts so faithfulness uses dataset case.contexts.
Do not treat high scores with a smarter mock as product quality.
"""
from __future__ import annotations
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

SEED = 'Vietnamese RAG Chatbot with LangChain + ChromaDB + Ollama A local, privacy-first Retrieval-Augmented Generation (RAG) chatbot for querying personal knowledge bases in Vietnamese. Built on LangChain, ChromaDB, and a locally-running Ollama LLM — no external API calls required after setup. Vietnamese-optimized retrieval using intfloat/multilingual-e5-base with correct query/passage prefixes Dual chunking strategy — Markdown documents split by header hierarchy, PDFs by recursive character splitting Max Marginal Relevance (MMR) retrieval for diverse, non-redundant context chunks Multi-turn conversation memory via ConversationalRetrievalChain with a sliding window of the last 5 turns Source citations shown inline after every answer — file name and section heading'
ATTACK = re.compile(
    r"(ignore all previous|system prompt|api key|bỏ qua mọi|tiết lộ)",
    re.I,
)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        q = str(body.get("question") or "")
        if ATTACK.search(q):
            ans = "I cannot reveal system prompts or secrets."
        elif SEED:
            # Weak: same SEED for every question (honest FAIL under strict judge)
            ans = "Based on available docs: " + SEED[:500]
        else:
            ans = f"(mock) no seed docs for: {q[:80]}"
        raw = json.dumps({"answer": ans, "contexts": []}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

print("mock adapter http://127.0.0.1:18080/chat (weak SEED, empty contexts)")
HTTPServer(("127.0.0.1", 18080), H).serve_forever()
