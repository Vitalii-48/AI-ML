# semantic_search_day3.py

import os

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


corpus = [
    "Python is a high-level, interpreted programming language known for its readability "
    "and simple syntax. It was created by Guido van Rossum and first released in 1991. "
    "Python emphasizes code readability, often using significant whitespace instead of "
    "curly braces to define blocks of code.",

    "In Python, data structures like lists, tuples, and dictionaries help organize data. "
    "A list is a mutable, ordered collection of items that can be changed after creation. "
    "A tuple is similar to a list but immutable, meaning its contents cannot be modified "
    "once created. Dictionaries store data as key-value pairs, allowing fast lookups by key.",

    "Python supports object-oriented programming (OOP) through the use of classes and "
    "objects. A class acts as a blueprint for creating objects, and the 'self' keyword "
    "refers to the current instance of the class. Methods defined inside a class can "
    "access and modify the object's attributes.",

    "Functions in Python are defined using the 'def' keyword and can accept parameters, "
    "return values, and have default arguments. List comprehensions offer a concise way "
    "to create lists based on existing iterables, often replacing longer for-loops with "
    "a single readable line of code.",

    "Managing dependencies in Python projects is commonly done using 'pip', the standard "
    "package manager. Virtual environments, created with tools like 'venv', help isolate "
    "project-specific packages so they don't conflict with other projects or the system "
    "Python installation.",

    "Exception handling in Python is done using 'try', 'except', 'else', and 'finally' "
    "blocks. This allows a program to catch and handle errors gracefully instead of "
    "crashing. Custom exceptions can also be created by subclassing the built-in "
    "'Exception' class.",

    "Python's built-in 'open()' function is used to read and write files. Files can be "
    "opened in different modes such as read ('r'), write ('w'), or append ('a'). Using "
    "the 'with' statement ensures that a file is properly closed after its block of code "
    "finishes executing, even if an error occurs.",

    "A module in Python is simply a file containing Python code that can be imported and "
    "reused in other programs. A package is a collection of modules organized in "
    "directories with an '__init__.py' file. The 'import' statement is used to bring "
    "modules or packages into your current script.",

    "Python supports several ways to iterate over data, including 'for' loops and 'while' "
    "loops. Generators, created using the 'yield' keyword, allow values to be produced "
    "one at a time, which is more memory-efficient than returning a full list at once.",

    "Decorators in Python are functions that modify the behavior of other functions "
    "without changing their code directly. They are applied using the '@' symbol above "
    "a function definition. Common use cases include logging, timing, and access control.",

    "Python's standard library includes modules like 'os' for interacting with the "
    "operating system, 'sys' for system-specific parameters, and 'datetime' for working "
    "with dates and times. These built-in modules reduce the need for external "
    "dependencies in many common tasks.",

    "Type hints in Python, introduced in PEP 484, allow developers to specify expected "
    "data types for variables, function parameters, and return values. While Python "
    "remains dynamically typed at runtime, type hints improve code readability and help "
    "tools like linters and IDEs catch potential errors earlier.",
]


model = SentenceTransformer("all-MiniLM-L6-v2")

corpus_embeddings = model.encode(corpus)


def cosine_similarity(a, b):
    """Рахує косинусну схожість між двома векторами."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, top_n=3):
    """
    Приймає питання користувача, повертає top_n
    найбільш схожих абзаців з корпусу.
    """
    query_embedding = model.encode(query)

    similarities = []
    for i, doc_embedding in enumerate(corpus_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


load_dotenv()

# Отримуємо API-ключ та ініціалізуємо офіційний клієнт Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file with "
        "GROQ_API_KEY=<your key> next to this script."
    )
client = Groq(api_key=api_key)


def generate_answer(query, search_results):
    """
    Приймає питання і результати пошуку, формує контекст
     і питає Groq дати фінальну відповідь.
    """
    context_parts = []
    for rank, (idx, score) in enumerate(search_results, start=1):
        context_parts.append(f"[Source {rank}]: {corpus[idx]}")
    context = "\n\n".join(context_parts)

    prompt = f"""Use the following context to answer the question.
If the context doesn't contain the answer, say so honestly.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[                              # type: ignore
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content


def main():
    print(f"Завантажено {len(corpus)} документів у корпус.\n")

    query = input("> You: ")
    results = search(query, top_n=3)

    print(f"> You: {query}\n")
    print("-> Top matches:")
    for rank, (idx, score) in enumerate(results, start=1):
        print(f"[{rank}] (score={score:.3f}) {corpus[idx][:80]}...")

    answer = generate_answer(query, results)
    print(f"\n-> GPT says:\n{answer}")


if __name__ == "__main__":
    main()