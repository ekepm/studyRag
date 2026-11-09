"""Small, local, citation-first RAG baseline using only the Python standard library."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MIN_SCORE = 0.12
MAX_CHUNK_CHARS = 500


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


def tokenize(text: str) -> list[str]:
    """Tokenize Latin words and Japanese runs (unigrams/bigrams)."""
    tokens: list[str] = []
    for part in re.findall(r"[A-Za-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]+", text.lower()):
        if part.isascii():
            tokens.append(part)
        elif len(part) > 1:
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
    return tokens


def load_chunks(data_dir: Path = DATA_DIR, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Read Markdown documents and split paragraphs into bounded chunks."""
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        for paragraph in re.split(r"\n\s*\n", body):
            text = paragraph.strip()
            if not text:
                continue
            while len(text) > max_chars:
                split_at = text.rfind("。", 0, max_chars)
                if split_at < max_chars // 2:
                    split_at = max_chars
                else:
                    split_at += 1
                chunks.append(Chunk(path.name, text[:split_at].strip()))
                text = text[split_at:].strip()
            if text:
                chunks.append(Chunk(path.name, text))
    return chunks


def retrieve(question: str, chunks: list[Chunk], limit: int = 3) -> list[dict[str, object]]:
    """Rank chunks with a compact BM25-style score over tokenized text."""
    query_tokens = set(tokenize(question))
    if not query_tokens or not chunks:
        return []
    tokenized = [tokenize(chunk.text) for chunk in chunks]
    average_length = sum(map(len, tokenized)) / len(tokenized) or 1
    results: list[dict[str, object]] = []
    for chunk, terms in zip(chunks, tokenized):
        frequencies = {term: terms.count(term) for term in query_tokens if term in terms}
        score = 0.0
        for term, frequency in frequencies.items():
            document_frequency = sum(term in document for document in tokenized)
            inverse_frequency = math.log(1 + (len(chunks) - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + 1.2 * (0.25 + 0.75 * len(terms) / average_length)
            score += inverse_frequency * (frequency * 2.2 / denominator)
        if score:
            results.append({"source": chunk.source, "text": chunk.text, "score": score})
    results.sort(key=lambda item: float(item["score"]), reverse=True)
    maximum = float(results[0]["score"]) if results else 0.0
    return [
        {**item, "relevance": round(float(item["score"]) / maximum, 3)}
        for item in results[:limit]
    ]


def answer(question: str, chunks: list[Chunk] | None = None) -> dict[str, object]:
    """Return retrieved evidence, abstaining when the match is too weak."""
    if not question.strip():
        raise ValueError("質問を入力してください。")
    hits = retrieve(question, chunks if chunks is not None else load_chunks())
    if not hits or float(hits[0]["score"]) < MIN_SCORE:
        return {"answer": "資料から確かな根拠を見つけられませんでした。質問を言い換えるか、資料を追加してください。", "abstained": True, "sources": []}
    return {
        "answer": "\n\n".join(f"{hit['text']}【{hit['source']}】" for hit in hits),
        "abstained": False,
        "sources": [{"source": hit["source"], "relevance": hit["relevance"]} for hit in hits],
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._json(200, {"status": "ok"})
            return
        if path != "/":
            self._json(404, {"error": "Not found"})
            return
        page = """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>studyRag</title>
<style>body{font:16px system-ui;max-width:760px;margin:3rem auto;padding:0 1rem;color:#19232d}textarea{width:100%;min-height:5rem}button{padding:.6rem 1rem;margin-top:.6rem}pre{white-space:pre-wrap;background:#f2f5f7;padding:1rem;border-radius:8px}</style>
<h1>studyRag</h1><p>サンプル文書から根拠を検索します。回答は文書の抜粋です。</p><form id="f"><textarea id="q" required placeholder="例：休暇の申請方法は？"></textarea><br><button>検索する</button></form><pre id="out">質問を入力してください。</pre>
<script>document.querySelector('#f').onsubmit=async e=>{e.preventDefault();let r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.querySelector('#q').value})});let d=await r.json();document.querySelector('#out').textContent=d.answer||d.error;};</script></html>"""
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/query":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                self._json(413, {"error": "Request too large"})
                return
            body = json.loads(self.rfile.read(length))
            question = body.get("question", "")
            if not isinstance(question, str):
                raise ValueError("質問は文字列で指定してください。")
            self._json(200, answer(question))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"studyRag listening at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping studyRag")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

