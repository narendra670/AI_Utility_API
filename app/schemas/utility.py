from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt to send to the model")
    system: str | None = Field(None, description="Optional system instruction")
    max_tokens: int | None = Field(None, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to summarize")
    max_words: int = Field(100, ge=10, le=1000, description="Target maximum length in words")


class GrammarRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to correct")


class AIResponse(BaseModel):
    success: bool = True
    result: str
    model: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
