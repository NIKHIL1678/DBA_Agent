import re
import logging
from typing import List, Union
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# Configure logging for audit trails
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIFilter:
    """
    Enterprise-grade middleware to intercept and redact Personally Identifiable Information
    (PII) from user inputs before they reach external LLM endpoints.
    """
    
    # Pre-compiled regex for performance optimization
    PATTERNS = {
        "EMAIL": re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),
        # Matches international, standard US, and dotted formats
        "PHONE": re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
        # Basic Credit Card regex (Visa, MasterCard, Amex, Discover)
        "CREDIT_CARD": re.compile(r'(?:\d[ -]*?){13,16}')
    }

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Applies all regex patterns to redact sensitive data strings."""
        if not text or not isinstance(text, str):
            return text

        redacted_text = text
        for pii_type, pattern in cls.PATTERNS.items():
            if pattern.search(redacted_text):
                logger.info(f"🛡️ PII Guardrail Triggered: Redacting {pii_type}")
                redacted_text = pattern.sub(f'[{pii_type}_REDACTED]', redacted_text)
                
        return redacted_text

    @classmethod
    def apply_to_messages(cls, messages: List[AnyMessage]) -> List[AnyMessage]:
        """
        Iterates through LangGraph state messages and redacts content.
        Mutates the messages safely for LangChain compatibility.
        """
        processed_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                redacted_content = cls.redact_text(msg.content)
                # Reconstruct the message to avoid mutating immutable properties unexpectedly
                if isinstance(msg, HumanMessage):
                    processed_messages.append(HumanMessage(content=redacted_content))
                elif isinstance(msg, AIMessage):
                    processed_messages.append(AIMessage(content=redacted_content))
                else:
                    msg.content = redacted_content
                    processed_messages.append(msg)
            else:
                processed_messages.append(msg)
                
        return processed_messages