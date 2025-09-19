from google.cloud import bigquery
from causal_rag import vector_search
from gemini import completion
import constant
import sys

def finalize_answer(question,  chunks):
    prompt = """
    You are a financial analyst assistant.
    Using only the information provided in the supporting evidence (chunks) below, answer the user's question as clearly and concisely as possible.
    Do not make up information or use outside knowledge.

    ## Supporting Evidence (Chunks)
    {chunks}

    ## Question
    {question}
    """

    prompt = prompt.format(
        chunks="\n".join([str(chunk) for chunk in chunks]),
        question=question
    )

    answer = completion(
        prompt,
        temperature=0.7
    )

    return answer

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print("\n================= Testing Vector Search =================\n")
        chunks = vector_search(question)
        print(f"Retrieved {len(chunks)} relevant chunks from vector search.")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---\n{chunk}\n")
        answer = finalize_answer(question, chunks)
        print("\n================= Final Answer =================\n")
        print(answer)
    