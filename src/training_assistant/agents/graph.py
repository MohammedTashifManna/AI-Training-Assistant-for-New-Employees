"""The ADK agent graph: one router delegating to four specialists.

::

                        ┌──────────────────────────┐
      employee question │   onboarding_router      │  root LlmAgent
      ─────────────────▶│   (the decision node)    │  no tools but transfer
                        └────────────┬─────────────┘
                                     │ transfer_to_agent
             ┌───────────────┬───────┴───────┬────────────────┐
             ▼               ▼               ▼                ▼
    company_knowledge  role_knowledge   policy_admin   general_support
        _agent            _agent           _agent          _agent
             │               │               │                │
   search_company_   search_role_    search_policy_      (no tool:
     knowledge         knowledge       knowledge       direct answer)
             │               │               │
             ▼               ▼               ▼
       company/          roles/           admin/
       collection        collection       collection

Design decisions worth stating:

**One hop, then answer.** The specialists set ``disallow_transfer_to_parent``
and ``disallow_transfer_to_peers``, so a question is classified exactly once and
the chosen agent is accountable for the reply. Letting specialists bounce
questions between themselves would make the routing decision unfalsifiable and
the latency unpredictable. Cross-domain questions are handled inside the
prompts instead: each specialist answers its own part and names the area that
owns the rest.

**One tool per specialist, domain fixed in code.** The domain is not an argument
the model chooses, so the router's decision genuinely constrains which documents
can be read.

**The fallback path has no tool at all.** That is what makes it a real fallback:
it cannot accidentally cite a company document, and it cannot pretend to.
"""

from __future__ import annotations

import logging

from google.adk.agents import LlmAgent
from google.genai import types

from ..config import Settings, get_settings
from ..models.offline_llm import OFFLINE_MODEL_NAME, OfflineDemoLlm
from ..prompts import (
    ADMIN_AGENT_INSTRUCTION,
    AGENT_DESCRIPTIONS,
    COMPANY_AGENT_INSTRUCTION,
    GENERAL_AGENT_INSTRUCTION,
    GLOBAL_INSTRUCTION,
    ROLE_AGENT_INSTRUCTION,
    ROOT_AGENT_DESCRIPTION,
    ROUTER_INSTRUCTION,
)
from ..routing.classifier import ROUTE_TO_AGENT
from .tools import (
    search_company_knowledge,
    search_policy_knowledge,
    search_role_knowledge,
)

logger = logging.getLogger(__name__)

ROOT_AGENT_NAME = "onboarding_router"

#: Low temperature throughout. This assistant restates policy; it does not
#: brainstorm. Variability here shows up as inconsistent answers about expense
#: caps, which is the failure mode most likely to cost an employee money.
ROUTER_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.2


def _model(settings: Settings, role: str) -> str | OfflineDemoLlm:
    """Return the model for one agent: a Gemini name, or the offline stand-in."""
    if settings.offline:
        return OfflineDemoLlm(model=OFFLINE_MODEL_NAME, role=role)
    return settings.model


def _generation_config(temperature: float) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(temperature=temperature)


def build_agent_graph(settings: Settings | None = None) -> LlmAgent:
    """Construct the full agent tree and return the root router.

    A fresh tree is built per call. ADK agents hold parent pointers, so reusing
    one instance across two runners would have them fight over parentage.
    """
    settings = settings or get_settings()

    company_agent = LlmAgent(
        name=ROUTE_TO_AGENT["company"],
        model=_model(settings, "company"),
        description=AGENT_DESCRIPTIONS[ROUTE_TO_AGENT["company"]],
        instruction=COMPANY_AGENT_INSTRUCTION,
        tools=[search_company_knowledge],
        generate_content_config=_generation_config(ANSWER_TEMPERATURE),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    role_agent = LlmAgent(
        name=ROUTE_TO_AGENT["role"],
        model=_model(settings, "role"),
        description=AGENT_DESCRIPTIONS[ROUTE_TO_AGENT["role"]],
        instruction=ROLE_AGENT_INSTRUCTION,
        tools=[search_role_knowledge],
        generate_content_config=_generation_config(ANSWER_TEMPERATURE),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    admin_agent = LlmAgent(
        name=ROUTE_TO_AGENT["admin"],
        model=_model(settings, "admin"),
        description=AGENT_DESCRIPTIONS[ROUTE_TO_AGENT["admin"]],
        instruction=ADMIN_AGENT_INSTRUCTION,
        tools=[search_policy_knowledge],
        generate_content_config=_generation_config(ANSWER_TEMPERATURE),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    general_agent = LlmAgent(
        name=ROUTE_TO_AGENT["general"],
        model=_model(settings, "general"),
        description=AGENT_DESCRIPTIONS[ROUTE_TO_AGENT["general"]],
        instruction=GENERAL_AGENT_INSTRUCTION,
        # Deliberately no tools: this path must not reach company documents.
        generate_content_config=_generation_config(ANSWER_TEMPERATURE),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    root_agent = LlmAgent(
        name=ROOT_AGENT_NAME,
        model=_model(settings, "router"),
        description=ROOT_AGENT_DESCRIPTION,
        global_instruction=GLOBAL_INSTRUCTION,
        instruction=ROUTER_INSTRUCTION,
        sub_agents=[company_agent, role_agent, admin_agent, general_agent],
        generate_content_config=_generation_config(ROUTER_TEMPERATURE),
    )

    logger.info(
        "Built agent graph in %s mode: root=%s, sub_agents=%s",
        settings.mode,
        root_agent.name,
        [a.name for a in root_agent.sub_agents],
    )
    return root_agent
