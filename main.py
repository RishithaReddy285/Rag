from dotenv import load_dotenv

from rag import query_rag


load_dotenv()


def main():
    print("Advanced RAG ready. Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = query_rag(question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
