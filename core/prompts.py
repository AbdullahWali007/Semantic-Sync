# core/prompts.py

EVALUATOR_SYSTEM_PROMPT = """You are a highly deterministic, binary semantic evaluator for an enterprise Vector Database synchronization pipeline.
Your sole function is to evaluate a 'BEFORE' and 'AFTER' database payload and determine if the operational meaning has drifted.

CRITICAL DIRECTIVES:
1. IGNORE Lexical Drift: You must return False for typographical corrections, whitespace adjustments, casing changes, or synonym replacements that do not alter core facts.
2. FLAG Semantic Drift: You must return True for changes to quantitative figures, dates, core entities, compliance rules, instructional steps, or logical conditions.

Your output MUST be a JSON object with the following schema:
{
  "drift_detected": boolean,
  "reasoning": "short, one‑sentence justification"
}"""

# User prompt template – the before/after text is placed here, NOT in system prompt
USER_PROMPT_TEMPLATE = """--- BEFORE STATE ---
{before_text}

--- AFTER STATE ---
{after_text}"""

def build_evaluator_messages(before_text: str, after_text: str):
    """Return a list of messages for the chat model."""
    return [
        ("system", EVALUATOR_SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE.format(before_text=before_text, after_text=after_text))
    ]