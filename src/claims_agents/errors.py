"""Domain errors. Callers translate these into HTTP status codes or CLI exits."""


class ClaimsError(Exception):
    """Base class for expected workflow failures."""


class LLMConfigError(ClaimsError):
    """The configured model provider cannot be constructed."""


class StructuredOutputError(ClaimsError):
    """The model did not return schema-valid JSON within the retry budget."""


class ExtractionError(ClaimsError):
    """A notice of loss could not be parsed into a claim."""


class PolicyNotFoundError(ClaimsError):
    """The requested policy id is not in the form corpus."""


class ClaimNotFoundError(ClaimsError):
    """No workflow exists for this claim id."""


class ClaimStateError(ClaimsError):
    """The workflow is not in the state the caller assumed."""


class DuplicateClaimError(ClaimsError):
    """A claim id was submitted twice."""
