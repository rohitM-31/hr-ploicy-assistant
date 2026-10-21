

import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
POLICY_DIR = Path(__file__).parent / "policies"

NOT_FOUND = "I couldn't find this in the policy. Please contact HR at hr@company.com."

SYSTEM_PROMPT = f"""You are an HR policy assistant for employees.
Rules:
1. Answer ONLY from the policy excerpts given. Never use outside knowledge.
2. Copy exact numbers, limits, deadlines and conditions as written.
3. If a rule depends on a condition (probation, city, amount), state the condition.
4. End with one line: Source: <policy name > section>
5. If the excerpts do not answer the question, reply exactly:
   "{NOT_FOUND}"
6. Keep it short: 1-3 sentences."""

def load_chunks(folder=POLICY_DIR):
    """Each '## Section' in each policy file becomes one chunk."""
    chunks = []
    for file in sorted(Path(folder).glob("*.md")):
        text = file.read_text(encoding="utf-8").strip()
        title_match = re.match(r"#\s+(.+)", text)
        doc_name = title_match.group(1).strip() if title_match else file.stem

        for sec in re.split(r"\n##\s+", text)[1:]:
            heading, _, body = sec.partition("\n")
            chunks.append({
                "source": f"{doc_name} > {heading.strip()}",
                "text": f"{heading.strip()}: {' '.join(body.split())}",
            })
    return chunks


# 2. RETRIEVERS

class GeminiRetriever:
    """Semantic search: understands meaning ('food' finds 'Meals')."""

    def __init__(self, chunks, client):
        from google.genai import types
        self.chunks, self.client, self.types = chunks, client, types
        docs = [c["source"] + "\n" + c["text"] for c in chunks]
        self.matrix = self._embed(docs, "RETRIEVAL_DOCUMENT")

    def _embed(self, texts, task):
        res = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=self.types.EmbedContentConfig(task_type=task),
        )
        vecs = np.array([e.values for e in res.embeddings], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def search(self, query, k=3, min_score=0.45):
        q = self._embed([query], "RETRIEVAL_QUERY")[0]
        scores = self.matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [(round(float(scores[i]), 3), self.chunks[i]) for i in top if scores[i] >= min_score]


class TfidfRetriever:
    """Keyword search fallback (no API key needed)."""

    def __init__(self, chunks):
        self.chunks = chunks
        hr_stop = {"company", "employee", "employees", "policy", "pay", "paid", "does", "can"}
        self.vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | hr_stop), ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform([c["source"] + " " + c["text"] for c in chunks])

    def search(self, query, k=3, min_score=0.15):
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = np.argsort(scores)[::-1][:k]
        return [(round(float(scores[i]), 3), self.chunks[i]) for i in top if scores[i] >= min_score]


# ---------------------------------------------------------------
# 3. RAG PIPELINE
# ---------------------------------------------------------------
class PolicyRAG:
    def __init__(self):
        self.chunks = load_chunks()
        self.client = None
        if GEMINI_API_KEY:
            from google import genai
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.retriever = GeminiRetriever(self.chunks, self.client)
            self.mode = "gemini"
        else:
            self.retriever = TfidfRetriever(self.chunks)
            self.mode = "offline"

    @property
    def policies(self):
        return sorted({c["source"].split(" > ")[0] for c in self.chunks})

    def _generate(self, question, hits):
        if not hits:
            return NOT_FOUND
        if not self.client:  # offline: show the exact policy text
            best = hits[0][1]
            return f"{best['text']}\nSource: {best['source']}"

        from google.genai import types
        context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for _, c in hits)
        resp = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=f"Policy excerpts:\n{context}\n\nEmployee question: {question}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,  # same question -> same answer
            ),
        )
        return (resp.text or NOT_FOUND).strip()

    def ask(self, question):
        hits = self.retriever.search(question)
        answer = self._generate(question, hits)
        found = bool(hits) and NOT_FOUND not in answer
        return {
            "question": question,
            "answer": answer,
            "found": found,
            "mode": self.mode,
            "sources": [
                {"source": c["source"], "score": s, "text": c["text"]} for s, c in hits
            ] if found else [],
        }


if __name__ == "__main__":
    rag = PolicyRAG()
    print(f"Mode: {rag.mode} | {len(rag.chunks)} sections from {len(rag.policies)} policies\n")
    for q in [
        "How many days of annual leave can I carry forward?",
        "Can I work from home while on probation?",
        "How much can I spend on food during a trip?",
        "Does the company pay for gym membership?",
    ]:
        r = rag.ask(q)
        print(f"Q: {q}\nA: {r['answer']}\n")
