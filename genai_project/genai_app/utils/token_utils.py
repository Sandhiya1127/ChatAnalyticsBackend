# genai_app/utils/token_utils.py

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limits (fallback logic)"""
    if not text:
        return ""
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    char_limit = max_tokens * 4
    return text[:char_limit - 100] + "\n\n[Note: Content truncated due to length]"
