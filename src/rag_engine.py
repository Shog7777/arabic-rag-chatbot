"""
RAG Engine — Gemini 2.0 Flash + Arabic TF-IDF Retriever
"""
import re, math, json
from pathlib import Path
from typing import List, Dict, Tuple
import google.generativeai as genai


class ArabicRetriever:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.tfidf_matrix: List[Dict] = []
        self.idf: Dict[str, float] = {}

    @staticmethod
    def normalize(text: str) -> str:
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        text = re.sub(r'[أإآ]', 'ا', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t for t in ArabicRetriever.normalize(text).split() if len(t) > 1]

    def chunk_text(self, text: str, source: str, size=350, overlap=70) -> List[Dict]:
        words = text.split()
        chunks, start, cid = [], 0, 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append({'text': ' '.join(words[start:end]), 'source': source, 'chunk_id': f"{source}_{cid}"})
            cid += 1
            start += size - overlap
        return chunks

    def add_documents(self, docs: List[Tuple[str, str]]):
        new_chunks = []
        for content, source in docs:
            new_chunks.extend(self.chunk_text(content, source))
        self.chunks.extend(new_chunks)
        self._build_tfidf()

    def _build_tfidf(self):
        n = len(self.chunks)
        df: Dict[str, int] = {}
        tf_list = []
        for chunk in self.chunks:
            tokens = self.tokenize(chunk['text'])
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            total = sum(tf.values()) or 1
            tf = {k: v/total for k, v in tf.items()}
            tf_list.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n+1)/(c+1))+1 for t, c in df.items()}
        self.tfidf_matrix = [{t: v*self.idf.get(t,1) for t,v in tf.items()} for tf in tf_list]

    def retrieve(self, query: str, top_k=5) -> List[Dict]:
        if not self.chunks:
            return []
        tokens = self.tokenize(query)
        qtf = {}
        for t in tokens:
            qtf[t] = qtf.get(t, 0) + 1
        total = sum(qtf.values()) or 1
        qvec = {t: (v/total)*self.idf.get(t,1) for t,v in qtf.items()}
        scores = []
        for i, cvec in enumerate(self.tfidf_matrix):
            common = set(qvec) & set(cvec)
            if not common:
                scores.append((0, i))
                continue
            dot = sum(qvec[k]*cvec[k] for k in common)
            na = math.sqrt(sum(v**2 for v in qvec.values()))
            nb = math.sqrt(sum(v**2 for v in cvec.values()))
            scores.append((dot/(na*nb+1e-10), i))
        scores.sort(reverse=True)
        return [{**self.chunks[i], 'score': round(s,4)} for s,i in scores[:top_k] if s > 0]


class GeminiRAGChatbot:
    SYSTEM = """أنت مساعد ذكي متخصص في تحليل المستندات والإجابة بالعربية.
قواعد:
1. أجب فقط من المعلومات الموجودة في السياق المقدم
2. إذا لم تجد الإجابة قل: "لم أجد هذه المعلومة في المستندات"
3. كن دقيقاً ومفيداً واذكر المصدر دائماً
4. استخدم العربية الفصحى الواضحة"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=self.SYSTEM
        )
        self.retriever = ArabicRetriever()
        self.chat = self.model.start_chat(history=[])
        self.docs_loaded = 0
        self.total_chunks = 0

    def add_documents(self, docs: List[Tuple[str, str]]):
        self.retriever.add_documents(docs)
        self.docs_loaded += len(docs)
        self.total_chunks = len(self.retriever.chunks)

    def ask(self, question: str, top_k=5) -> Dict:
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            return {'answer': 'لم يتم تحميل أي مستندات بعد.', 'sources': [], 'chunks_used': 0}

        context = "\n\n---\n\n".join(
            f"[مقتطف {i+1} من: {c['source']} | صلة: {c['score']}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        sources = list({c['source'] for c in chunks})

        prompt = f"""السياق من المستندات:
━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━

السؤال: {question}

أجب بناءً على السياق فقط."""

        response = self.chat.send_message(prompt)
        return {
            'answer': response.text,
            'sources': sources,
            'chunks_used': len(chunks),
        }

    def reset(self):
        self.retriever = ArabicRetriever()
        self.chat = self.model.start_chat(history=[])
        self.docs_loaded = 0
        self.total_chunks = 0
