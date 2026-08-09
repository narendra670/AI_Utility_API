from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.utility import (
    AIResponse,
    GenerateRequest,
    GrammarRequest,
    HealthResponse,
    SummarizeRequest,
)
from app.services.ai_service import AIProviderError, AIService
from app.utils.logger import logger
from config import get_settings

router = APIRouter(prefix="/utility", tags=["utility"])

settings = get_settings()


def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service


@router.get("/health", response_model=HealthResponse, summary="Check API health")
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=settings.ai_model)


@router.post("/generate", response_model=AIResponse, summary="Generate text from a prompt")
def generate(
    request: GenerateRequest,
    service: AIService = Depends(get_ai_service),
) -> AIResponse:
    try:
        result = service.generate(
            request.prompt,
            system=request.system,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except AIProviderError as exc:
        logger.warning("Generate failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    return AIResponse(result=result, model=service.model)


@router.post("/summarize", response_model=AIResponse, summary="Summarize a block of text")
def summarize(
    request: SummarizeRequest,
    service: AIService = Depends(get_ai_service),
) -> AIResponse:
    try:
        result = service.summarize(request.text, max_words=request.max_words)
    except AIProviderError as exc:
        logger.warning("Summarize failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    return AIResponse(result=result, model=service.model)


@router.post("/grammar", response_model=AIResponse, summary="Correct grammar and spelling")
def grammar(
    request: GrammarRequest,
    service: AIService = Depends(get_ai_service),
) -> AIResponse:
    try:
        result = service.grammar_check(request.text)
    except AIProviderError as exc:
        logger.warning("Grammar check failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    return AIResponse(result=result, model=service.model)
