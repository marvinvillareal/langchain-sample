"""
Step 6: Structured Output with Pydantic
-----------------------------------------
Shows how to make the LLM return validated, typed data
using .with_structured_output() and Pydantic models.

Usage:
    python 06_structured_output.py
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

LLM_MODEL = "llama3.2"


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class SentimentAnalysis(BaseModel):
    """Result of analysing a piece of text for sentiment."""
    sentiment: str = Field(description="'positive', 'negative', or 'neutral'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="One-sentence explanation for the sentiment")


class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type: PERSON, ORG, LOCATION, DATE, etc.")


class EntityExtraction(BaseModel):
    """Named entities extracted from text."""
    entities: List[ExtractedEntity]
    summary: str = Field(description="One-sentence summary of the text")


class CodeReview(BaseModel):
    """Structured code review."""
    issues: List[str] = Field(description="List of problems found in the code")
    suggestions: List[str] = Field(description="Actionable improvement suggestions")
    severity: str = Field(description="'low', 'medium', or 'high'")
    overall_score: int = Field(description="Code quality score from 1 (worst) to 10 (best)")
    approved: bool = Field(description="Whether the code is approved as-is")


# ── Demo Functions ────────────────────────────────────────────────────────────

def demo_sentiment():
    print("\n── 1. Sentiment Analysis ───────────────────────────────────────")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(SentimentAnalysis)

    texts = [
        "LangChain makes building LLM apps really enjoyable!",
        "I wasted three hours debugging this broken API.",
        "The weather today is neither good nor bad.",
    ]
    for text in texts:
        result: SentimentAnalysis = structured_llm.invoke(
            f"Analyse the sentiment of this text: {text}"
        )
        print(f"\n  Text     : {text}")
        print(f"  Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
        print(f"  Reason   : {result.reasoning}")


def demo_entity_extraction():
    print("\n── 2. Entity Extraction ────────────────────────────────────────")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(EntityExtraction)

    text = (
        "Elon Musk founded SpaceX in 2002 in Hawthorne, California. "
        "The company launched the Falcon 9 rocket in June 2010."
    )
    result: EntityExtraction = structured_llm.invoke(
        f"Extract all named entities from this text: {text}"
    )
    print(f"\n  Text    : {text}")
    print(f"  Summary : {result.summary}")
    print("  Entities:")
    for entity in result.entities:
        print(f"    [{entity.type}] {entity.name}")


def demo_code_review():
    print("\n── 3. Code Review ──────────────────────────────────────────────")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Python code reviewer. Be thorough and specific."),
        ("human", "Review this Python code:\n\n```python\n{code}\n```"),
    ])

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = prompt | llm.with_structured_output(CodeReview)

    bad_code = """\
def get_user(id):
    import sqlite3
    conn = sqlite3.connect('db.sqlite3')
    query = f"SELECT * FROM users WHERE id = {id}"
    result = conn.execute(query).fetchone()
    return result
"""
    result: CodeReview = chain.invoke({"code": bad_code})
    print(f"\n  Score   : {result.overall_score}/10")
    print(f"  Severity: {result.severity}")
    print(f"  Approved: {result.approved}")
    print("  Issues:")
    for issue in result.issues:
        print(f"    ✗ {issue}")
    print("  Suggestions:")
    for suggestion in result.suggestions:
        print(f"    → {suggestion}")


if __name__ == "__main__":
    print("🏗️  Structured Output Demo")
    demo_sentiment()
    demo_entity_extraction()
    demo_code_review()
    print("\n✅ Done")
