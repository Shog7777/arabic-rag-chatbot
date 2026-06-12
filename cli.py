"""
CLI interface — test the RAG chatbot from terminal.
Usage: python cli.py --folder ./data/docs --api-key gsk_...
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src.rag_engine import ArabicRAGChatbot
from src.document_loader import load_documents_from_folder

DEMO_TEXT = """
رؤية المملكة العربية السعودية 2030 هي خطة طموحة أطلقها ولي العهد الأمير محمد بن سلمان عام 2016.
تهدف الرؤية إلى تنويع مصادر الدخل الوطني وتقليل الاعتماد على النفط.
تشمل أهداف الرؤية ثلاثة محاور رئيسية: مجتمع حيوي، اقتصاد مزدهر، ووطن طموح.
تسعى الرؤية إلى رفع مساهمة القطاع الخاص في الناتج المحلي من 40% إلى 65%.
كما تهدف إلى رفع نسبة توطين الوظائف وخفض معدل البطالة إلى 7%.
تتضمن الرؤية مشاريع عملاقة مثل نيوم، والقدية، والبحر الأحمر.
في مجال الترفيه، تم رفع الحظر عن السينما وإقامة الفعاليات الرياضية والثقافية.
"""


def main():
    parser = argparse.ArgumentParser(description='Arabic RAG Chatbot CLI')
    parser.add_argument('--folder', type=str, help='Folder containing documents')
    parser.add_argument('--api-key', type=str,
                        default=os.getenv('GROQ_API_KEY', ''),
                        help='Groq API key')
    parser.add_argument('--model', type=str,
                        default='llama-3.3-70b-versatile',
                        help='Model name')
    parser.add_argument('--demo', action='store_true',
                        help='Run with demo Arabic text')
    args = parser.parse_args()

    if not args.api_key:
        print("❌ No API key. Use --api-key or set GROQ_API_KEY env var.")
        sys.exit(1)

    print("\n" + "━" * 50)
    print("  🧠 Arabic RAG Chatbot — CLI Mode")
    print("━" * 50 + "\n")

    chatbot = ArabicRAGChatbot(api_key=args.api_key, model=args.model)

    # Load documents
    if args.demo:
        print("📄 Using demo text about Vision 2030...\n")
        chatbot.add_documents([(DEMO_TEXT, "رؤية_2030_demo.txt")])
    elif args.folder:
        docs = load_documents_from_folder(args.folder)
        if not docs:
            print("❌ No documents found in folder.")
            sys.exit(1)
        chatbot.add_documents(docs)
    else:
        print("⚠️  No documents specified. Using demo text.")
        chatbot.add_documents([(DEMO_TEXT, "رؤية_2030_demo.txt")])

    print("\n✅ Ready! Type your question in Arabic (or 'exit' to quit)\n")
    print("━" * 50)

    while True:
        try:
            question = input("\n❓ سؤالك: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 مع السلامة!")
            break

        if question.lower() in ('exit', 'quit', 'خروج'):
            print("\n👋 مع السلامة!")
            break

        if not question:
            continue

        print("\n🔍 جاري البحث...\n")
        try:
            result = chatbot.ask(question)
            print("🤖 الإجابة:")
            print("─" * 40)
            print(result['answer'])
            print("─" * 40)
            print(f"📄 المصادر: {', '.join(result['sources'])}")
            print(f"📊 المقاطع المستخدمة: {result['chunks_used']} | التوكنز: {result.get('tokens_used', 'N/A')}")
        except Exception as e:
            print(f"\n❌ خطأ: {e}")


if __name__ == '__main__':
    main()
