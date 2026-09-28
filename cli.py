"""AGENTIC STAR Marketplace entry adapter for RET-C2-337."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from framework.schemas.invocation_context import InvocationContext
from framework.secrets.base import SecretProvider
from framework.secrets.context import current_secrets

try:
    from framework.utils.config_loader import load_agent_config
except ImportError:  # AgentCore 1.0.0 compatibility for this repository's test image.
    from framework.utils.config_loader import load_config

    def load_agent_config(agent_dir: Path) -> dict[str, Any]:
        config_path = agent_dir / "config" / "config.yaml"
        return load_config(str(config_path)) if config_path.exists() else {}


try:
    from shared.bootstrap import run_agent_marketplace
except ModuleNotFoundError as exc:  # pragma: no cover - old local wheel has no Marketplace extra
    if exc.name != "shared.bootstrap":
        raise

    def run_agent_marketplace(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("The installed AgentCore wheel does not include Marketplace support")


from shared.services.llm.base_llm import BaseLLM

from src.entry_adapter.auth import (
    STG_DEFAULT_ORG_ENV,
    STG_DEFAULT_TENANT_ENV,
    VERIFIED_ORG_STATE_ATTR,
    VERIFIED_TENANT_STATE_ATTR,
)
from src.graph.graph import RETDemandAnomalyAgent

try:
    # Newer entry-adapter parts expose the complete contract directly.
    from src.entry_adapter.auth import SCOPE_KEYS, STG_DEFAULT_SCOPE_ENVS
except ImportError:
    # The pinned v0.2.0 part exposes the same contract as paired constants.
    SCOPE_KEYS = (VERIFIED_TENANT_STATE_ATTR, VERIFIED_ORG_STATE_ATTR)
    STG_DEFAULT_SCOPE_ENVS = {
        VERIFIED_TENANT_STATE_ATTR: STG_DEFAULT_TENANT_ENV,
        VERIFIED_ORG_STATE_ATTR: STG_DEFAULT_ORG_ENV,
    }


_NAMESPACE = "ret"
_AGENT_NAME = "RET-C2-337"
_PROVIDER_AGENT_NAME = "ret_c2_337"

_AZURE_OPENAI_SECRET_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)
_TIMESFM_SECRET_KEYS = ("timesfm_endpoint", "timesfm_api_token")
_SCOPE_ENV_KEYS = tuple(STG_DEFAULT_SCOPE_ENVS[key] for key in SCOPE_KEYS)
_REQUIRED_SECRET_KEYS = (*_AZURE_OPENAI_SECRET_KEYS, *_SCOPE_ENV_KEYS)
_ALLOWED_SECRET_KEYS = frozenset((*_REQUIRED_SECRET_KEYS, *_TIMESFM_SECRET_KEYS))


class _AllowlistedEnvironmentProvider(SecretProvider):
    """Expose only this single-agent container's declared environment values."""

    def __init__(self) -> None:
        self._namespace = _NAMESPACE
        self._agent_name = _PROVIDER_AGENT_NAME

    def get(self, key: str, default: str | None = None) -> str | None:
        if key not in _ALLOWED_SECRET_KEYS:
            return default
        value = os.environ.get(key)
        return value if value else default


def _require_nonblank(provider: SecretProvider, key: str) -> str:
    value = provider.require(key)
    if not value.strip():
        raise ValueError(f"Marketplace environment variable {key!r} must not be blank")
    return value


def _require_azure_endpoint(provider: SecretProvider) -> str:
    """Reject ``AZURE_OPENAI_ENDPOINT`` values that are not a bare Azure resource endpoint.

    ``AzureOpenAIClient`` appends its own ``/openai/v1/`` path, so anything
    beyond a scheme and host silently produces a broken request URL, and the
    failure surfaces inside ``complete()``. A node that swallows that exception
    degrades quietly while the invocation still reports ``status: success``,
    which is why this has to be a startup check. The value is reported here,
    never repaired.
    """

    endpoint = _require_nonblank(provider, "AZURE_OPENAI_ENDPOINT")
    parts = urlsplit(endpoint)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or port == -1
        or parts.path.strip("/")
        or parts.query
        or parts.fragment
        or any(character.isspace() for character in endpoint)
    ):
        raise ValueError(
            "Marketplace environment variable 'AZURE_OPENAI_ENDPOINT' must be a bare Azure "
            "resource endpoint -- scheme and host only, with no path (such as "
            "'/openai'), query or fragment"
        )
    return endpoint


def _scope_from_environment(provider: SecretProvider) -> dict[str, str]:
    """Resolve every entry-adapter scope key without normalising bad IDs."""

    scope: dict[str, str] = {}
    for key in SCOPE_KEYS:
        env_name = STG_DEFAULT_SCOPE_ENVS[key]
        value = provider.require(env_name)
        if not value or any(character.isspace() for character in value):
            raise ValueError(
                f"Marketplace scope environment variable {env_name!r} " "must be a non-empty string without whitespace"
            )
        scope[key] = value
    return scope


class MarketplaceAzureLLM(BaseLLM):  # type: ignore[misc]
    """Create an Azure client per operation; never retain connection values."""

    def __init__(self, *, tools: tuple[Any, ...] = ()) -> None:
        if tools:
            self._tools = tools

    def _new_client(self) -> BaseLLM:
        provider = current_secrets()
        values = {key: _require_nonblank(provider, key) for key in _AZURE_OPENAI_SECRET_KEYS}

        # The provider integration is optional at import time and is the only
        # dependency intentionally imported on an invocation worker thread.
        from shared.services.llm.azure_openai_client import AzureOpenAIClient

        client: BaseLLM = AzureOpenAIClient(
            {
                "api_key": values["AZURE_OPENAI_API_KEY"],
                "azure_endpoint": values["AZURE_OPENAI_ENDPOINT"],
                "azure_deployment": values["AZURE_OPENAI_DEPLOYMENT"],
            }
        )
        tools = getattr(self, "_tools", ())
        return client.bind_tools(list(tools)) if tools else client

    def complete(self, messages: list[Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self._new_client().complete(messages))

    def stream(self, messages: list[Any]) -> Iterator[str]:
        yield from self._new_client().stream(messages)

    def bind_tools(self, tools: list[Any]) -> "MarketplaceAzureLLM":
        return type(self)(tools=tuple(tools))


class MarketplaceGraph(RETDemandAnomalyAgent):
    """Marketplace-only scope, HITL, secret, and Azure wiring."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # The runner may provision secrets only after compile(). Inject the
        # credential-free LLM factory before register_nodes() captures config.
        marketplace_config = dict(config or {})
        marketplace_config["llm"] = MarketplaceAzureLLM()
        super().__init__(config=marketplace_config)

    def invoke(
        self,
        user_input: str,
        session_id: str = "",
        ctx: InvocationContext | None = None,
        input_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Runner history is untrusted free text outside the S-2 scanned fields.
        # Discard every runner-owned key and construct only trusted scope.
        del input_context
        scope = _scope_from_environment(_AllowlistedEnvironmentProvider())

        # Marketplace has no resume channel, so no invocation may enter HITL.
        marketplace_ctx = replace(ctx, hitl_allowed=False) if ctx is not None else None
        return cast(
            dict[str, Any],
            super().invoke(
                user_input,
                session_id=session_id,
                ctx=marketplace_ctx,
                input_context=scope,
            ),
        )

    def provision_secrets(self, provider: SecretProvider) -> None:
        # Do not inherit the runner's broader/dotenv-capable provider boundary.
        del provider
        environment_provider = _AllowlistedEnvironmentProvider()
        for key in _AZURE_OPENAI_SECRET_KEYS:
            _require_nonblank(environment_provider, key)
        _scope_from_environment(environment_provider)
        _require_azure_endpoint(environment_provider)
        super().provision_secrets(environment_provider)


def main() -> None:
    run_agent_marketplace(
        MarketplaceGraph,
        agent_name=_AGENT_NAME,
        namespace=_NAMESPACE,
        config=load_agent_config(Path(__file__).resolve().parent),
    )


if __name__ == "__main__":
    main()
