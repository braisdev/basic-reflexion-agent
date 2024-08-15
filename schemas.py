from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field


class Reflection(BaseModel):
    """Your reflection on the initial answer."""
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")

    class Config:
        extra = "forbid"  # Disallow additional properties


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    references: List[str] = Field(description="Citations that motivated your updated answer.")
