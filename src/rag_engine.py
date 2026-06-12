"""
Arabic RAG Engine - Core logic for document processing and retrieval
Uses Groq (free) + simple TF-IDF retrieval (no paid embeddings needed)
"""

import os
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from groq import Groq


# ─── Simple Arabic-aware TF-IDF Retriever (no external vector DB needed) ───

class ArabicRetriever:
    """
    Lightweight TF-IDF retriever with Arabic text normalization.
    No external vector DB required — runs fully offline.
    """

    def __init__(self):
        self.chunks: List[Dict] = []          # {text, source, chunk_id}
        self.tfidf_matrix: List[Dict] = []    # term frequencies per chunk
        self.idf: Dict[str, float] = {}       # inverse document frequencies

    # ── Text normalization ──────────────────────────────────────────────────

    @staticmethod
    def normalize_arabic(text: str) -> str:
        """Normalize Arabic text for better matching."""
        # Remove tashkeel (diacritics)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        # Normalize alef variants → ا
        text = re.sub(r'[أإآ]', 'ا', text)
        # Normalize teh marbuta → ه
        text = re.sub(r'ة', 'ه', text)
        # Normalize yeh variants → ي
        text = re.sub(r'ى', 'ي', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        normalized = ArabicRetriever.normalize_arabic(text)
        return [t for t in normalized.split() if len(t) > 1]

    # ── Chunking ────────────────────────────────────────────────────────────

    def chunk_text(self, text: str, source: str, chunk_size: int = 400, overlap: int = 80) -> List[Dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': f"{source}_{chunk_id}"
            })
            chunk_id += 1
            start += chunk_size - overlap

        return chunks

    # ── Indexing ────────────────────────────────────────────────────────────

    def add_documents(self, texts: List[Tuple[str, str]]):
        """
        Add documents to the index.
        texts: list of (content, source_name) tuples
        """
        new_chunks = []
        for content, source in texts:
            new_chunks.extend(self.chunk_text(content, source))

        self.chunks.extend(new_chunks)
        self._build_tfidf()
        print(f"✅ Indexed {len(new_chunks)} chunks from {len(texts)} documents")

    def _build_tfidf(self):
        """Build TF-IDF index from chunks."""
        n_docs = len(self.chunks)
        df: Dict[str, int] = {}

        # Term frequency per chunk
        tf_list = []
        for chunk in self.chunks:
            tokens = self.tokenize(chunk['text'])
            tf: Dict[str, float] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            # Normalize TF
            total = sum(tf.values()) or 1
            tf = {k: v / total for k, v in tf.items()}
            tf_list.append(tf)
            for token in tf:
                df[token] = df.get(token, 0) + 1

        # IDF
        self.idf = {
            token: math.log((n_docs + 1) / (count + 1)) + 1
            for token, count in df.items()
        }

        # TF-IDF
        self.tfidf_matrix = []
        for tf in tf_list:
            tfidf = {token: tf_val * self.idf.get(token, 1)
                     for token, tf_val in tf.items()}
            self.tfidf_matrix.append(tfidf)

    # ── Retrieval ───────────────────────────────────────────────────────────

    def _cosine_similarity(self, vec_a: Dict, vec_b: Dict) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        return dot / (norm_a * norm_b + 1e-10)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        if not self.chunks:
            return []

        # Build query vector
        tokens = self.tokenize(query)
        query_tf: Dict[str, float] = {}
        for token in tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
        total = sum(query_tf.values()) or 1
        query_tfidf = {k: (v / total) * self.idf.get(k, 1)
                       for k, v in query_tf.items()}

        # Score all chunks
        scores = []
        for i, chunk_vec in enumerate(self.tfidf_matrix):
            score = self._cosine_similarity(query_tfidf, chunk_vec)
            scores.append((score, i))

        # Return top_k
        scores.sort(reverse=True)
        return [
            {**self.chunks[i], 'score': round(score, 4)}
            for score, i in scores[:top_k]
            if score > 0
        ]

    def save_index(self, path: str = "data/index.json"):
        """Persist index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': self.chunks,
                'idf': self.idf,
                'tfidf_matrix': self.tfidf_matrix
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 Index saved to {path}")

    def load_index(self, path: str = "data/index.json"):
        """Load index from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        self.idf = data['idf']
        self.tfidf_matrix = data['tfidf_matrix']
        print(f"📂 Loaded {len(self.chunks)} chunks from {path}")


# ─── RAG Pipeline ───────────────────────────────────────────────────────────

class ArabicRAGChatbot:
    """
    Full RAG pipeline: retrieve relevant chunks → augment prompt → generate answer.
    Powered by Groq (free tier, fast inference).
    """

    SYSTEM_PROMPT = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة باللغة العربية بناءً على المستندات المقدمة.

قواعد صارمة:
1. أجب فقط بناءً على المعلومات الموجودة في السياق المقدم
2. إذا لم تجد المعلومة في السياق، قل بوضوح: "لم أجد هذه المعلومة في المستندات المتاحة"
3. اذكر المصدر عند الإجابة (اسم الملف)
4. كن دقيقاً ومختصراً
5. تكلم بالعربية الفصحى الواضحة"""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.retriever = ArabicRetriever()
        self.chat_history: List[Dict] = []

    def add_documents(self, texts: List[Tuple[str, str]]):
        """Add documents to the knowledge base."""
        self.retriever.add_documents(texts)

    def ask(self, question: str, top_k: int = 4) -> Dict:
        """
        Ask a question and get an answer based on the documents.
        Returns: {answer, sources, chunks_used, model}
        """
        # 1. Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve(question, top_k=top_k)

        if not relevant_chunks:
            return {
                'answer': 'لم يتم تحميل أي مستندات بعد. يرجى رفع ملفات أولاً.',
                'sources': [],
                'chunks_used': 0,
                'model': self.model
            }

        # 2. Build context from chunks
        context_parts = []
        sources = list({chunk['source'] for chunk in relevant_chunks})

        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(
                f"[مقتطف {i} من: {chunk['source']} | درجة الصلة: {chunk['score']}]\n{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # 3. Build augmented prompt
        augmented_prompt = f"""السياق المستخرج من المستندات:
━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━

السؤال: {question}

أجب على السؤال بناءً على السياق أعلاه فقط."""

        # 4. Generate answer (with conversation history)
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(self.chat_history[-6:])  # last 3 turns
        messages.append({"role": "user", "content": augmented_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1000,
            temperature=0.3,  # lower = more factual
        )

        answer = response.choices[0].message.content

        # 5. Update history
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            'answer': answer,
            'sources': sources,
            'chunks_used': len(relevant_chunks),
            'model': self.model,
            'tokens_used': response.usage.total_tokens
        }

    def clear_history(self):
        """Reset conversation history."""
        self.chat_history = []
        print("🔄 Conversation history cleared")
