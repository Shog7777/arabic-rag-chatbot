"""
Unit tests for ArabicRetriever — no API key needed.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.rag_engine import ArabicRetriever


@pytest.fixture
def retriever_with_docs():
    r = ArabicRetriever()
    docs = [
        ("رؤية 2030 تهدف إلى تنويع الاقتصاد وتقليل الاعتماد على النفط وخلق فرص عمل للشباب السعودي.", "doc1.txt"),
        ("الذكاء الاصطناعي هو مجال من مجالات علوم الحاسوب يهتم ببناء أنظمة ذكية قادرة على التعلم.", "doc2.txt"),
        ("المملكة العربية السعودية تستثمر بشكل كبير في مجال التقنية والابتكار ضمن مبادرات رؤية 2030.", "doc3.txt"),
    ]
    r.add_documents(docs)
    return r


class TestArabicNormalization:
    def test_removes_tashkeel(self):
        text = "مُحَمَّدٌ"
        result = ArabicRetriever.normalize_arabic(text)
        assert 'ُ' not in result
        assert 'َ' not in result
        assert 'ً' not in result

    def test_normalizes_alef(self):
        assert ArabicRetriever.normalize_arabic("أحمد") == "احمد"
        assert ArabicRetriever.normalize_arabic("إبراهيم") == "ابراهيم"
        assert ArabicRetriever.normalize_arabic("آمن") == "امن"

    def test_normalizes_teh_marbuta(self):
        result = ArabicRetriever.normalize_arabic("مدرسة")
        assert result == "مدرسه"

    def test_tokenize_returns_list(self):
        tokens = ArabicRetriever.tokenize("الذكاء الاصطناعي مستقبل التقنية")
        assert isinstance(tokens, list)
        assert len(tokens) > 0


class TestRetrieval:
    def test_retrieval_returns_results(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("رؤية 2030", top_k=2)
        assert len(results) > 0

    def test_retrieval_scores_are_positive(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("الاقتصاد", top_k=3)
        for r in results:
            assert r['score'] >= 0

    def test_retrieval_returns_source(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("الذكاء الاصطناعي", top_k=1)
        assert 'source' in results[0]
        assert 'doc2.txt' in results[0]['source']

    def test_relevant_doc_ranked_first(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("الذكاء الاصطناعي التعلم", top_k=3)
        top_source = results[0]['source']
        assert 'doc2' in top_source

    def test_top_k_respected(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("السعودية", top_k=2)
        assert len(results) <= 2

    def test_empty_retriever(self):
        empty = ArabicRetriever()
        results = empty.retrieve("أي سؤال")
        assert results == []


class TestChunking:
    def test_chunking_splits_long_text(self):
        r = ArabicRetriever()
        long_text = " ".join(["كلمة"] * 1000)
        chunks = r.chunk_text(long_text, "test.txt", chunk_size=200, overlap=40)
        assert len(chunks) > 1

    def test_chunk_contains_source(self):
        r = ArabicRetriever()
        chunks = r.chunk_text("نص قصير للاختبار", "myfile.txt")
        assert all(c['source'] == 'myfile.txt' for c in chunks)


class TestPersistence:
    def test_save_and_load(self, retriever_with_docs, tmp_path):
        path = str(tmp_path / "test_index.json")
        retriever_with_docs.save_index(path)

        new_r = ArabicRetriever()
        new_r.load_index(path)

        assert len(new_r.chunks) == len(retriever_with_docs.chunks)

        # Retrieval should work after loading
        results = new_r.retrieve("رؤية 2030")
        assert len(results) > 0
