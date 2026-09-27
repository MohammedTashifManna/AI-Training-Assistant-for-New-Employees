"""Consistent logging setup for every entry point.

ADK is chatty by default and emits several notices that are correct but not
actionable for a user of this project (context-cache advice for multi-agent
apps, an experimental-feature warning from the transfer tool). They are quieted
here so that real warnings remain visible.
"""

from __future__ import annotations

import logging
import warnings

#: ADK loggers that emit informational notices we deliberately accept.
_QUIET_LOGGERS = (
    # Emits context-cache advice for any multi-agent app. Accepted: this
    # assistant makes one transfer per question, so prompt-prefix caching across
    # transfers is not worth the added configuration.
    "google_adk.google.adk.runners",
    "google_adk.google.adk.apps.app",
    "google_adk.google.adk.flows.llm_flows.functions",
    "google_adk.google.adk.models.registry",
    "google.adk.runners",
    "google.adk.apps.app",
)


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging and suppress known-noisy third-party output.

    Args:
        verbose: When True, show DEBUG from this project and leave third-party
            notices in place.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    # Our own package stays at the requested level.
    logging.getLogger("training_assistant").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )

    if verbose:
        return

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Quiet the transfer tool's experimental-feature notice. The feature is
    # ADK's own function-declaration path, which we do not control.
    warnings.filterwarnings(
        "ignore",
        message=r".*JSON_SCHEMA_FOR_FUNC_DECL.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*\[EXPERIMENTAL\].*",
        category=UserWarning,
    )


def quiet_third_party() -> None:
    """Suppress third-party noise without touching root logging config.

    Used by the Streamlit app, which manages its own logging.
    """
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=r".*\[EXPERIMENTAL\].*", category=UserWarning)
