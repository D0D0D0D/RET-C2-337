"""Marketplace entry-adapter boundary tests."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.base import MissingSecret, NullProvider
from framework.secrets.context import bound_secrets

import cli
from src.graph.graph import RETDemandAnomalyAgent


def _set_required_environment(monkeypatch, suffix: str = "one") -> dict[str, str]:
    values = {key: f"test-{suffix}-{key.lower()}" for key in cli._REQUIRED_SECRET_KEYS}
    # The startup gate requires a bare Azure resource endpoint, so
    # this value cannot be a placeholder like the other secrets.
    values["AZURE_OPENAI_ENDPOINT"] = f"https://{suffix}.services.ai.azure.example"
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def _contains(value, needle: str) -> bool:
    if isinstance(value, dict):
        return any(_contains(key, needle) or _contains(item, needle) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains(item, needle) for item in value)
    return needle in value if isinstance(value, str) else False


def _request() -> str:
    return json.dumps(
        {
            "series": {"SKU1": [10, 11, 9, 10, 12, 10, 11] * 2},
            "covariates": {"promo": [0] * 14},
            "inference_mode": "baseline",
        }
    )


def test_marketplace_discards_runner_context_and_rebuilds_scope(monkeypatch):
    values = _set_required_environment(monkeypatch)
    sentinel = "UNSCANNED-CONVERSATION-337"
    captured = {}

    def parent_invoke(self, user_input, session_id="", ctx=None, input_context=None):
        captured.update(user_input=user_input, ctx=ctx, input_context=input_context)
        return {"status": "success"}

    monkeypatch.setattr(RETDemandAnomalyAgent, "invoke", parent_invoke)
    original_ctx = InvocationContext(
        caller_id="marketplace-user",
        caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        hitl_allowed=True,
    )
    result = cli.MarketplaceGraph().invoke(
        _request(),
        ctx=original_ctx,
        input_context={
            "conversation_history": [{"role": "user", "content": sentinel}],
            "production_company_id": "attacker-company",
            "future_runner_key": sentinel,
        },
    )

    expected_scope = {
        "production_company_id": values["STG_DEFAULT_COMPANY_ID"],
        "project_id": values["STG_DEFAULT_PROJECT_ID"],
    }
    assert result["status"] == "success"
    assert captured["input_context"] == expected_scope
    assert not _contains(captured["input_context"], sentinel)
    assert captured["ctx"] is not original_ctx
    assert captured["ctx"].caller_trust_level is TrustLevel.VERIFIED_EXTERNAL
    assert captured["ctx"].hitl_allowed is False


def test_conversation_history_never_enters_real_graph_state(monkeypatch):
    _set_required_environment(monkeypatch)
    sentinel = "UNSCANNED-CONVERSATION-STATE-337"
    agent = cli.MarketplaceGraph()
    agent.compile()
    original_invoke = agent._compiled.invoke
    captured = {}

    def capture_state(initial_state, config=None):
        captured["initial"] = initial_state
        final_state = original_invoke(initial_state, config=config)
        captured["final"] = final_state
        return final_state

    agent._compiled.invoke = capture_state
    result = agent.invoke(
        _request(),
        ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
        input_context={"conversation_history": [{"role": "user", "content": sentinel}]},
    )

    assert not _contains(captured["initial"], sentinel)
    assert not _contains(captured["final"], sentinel)
    assert not _contains(result, sentinel)


@pytest.mark.parametrize("invalid_scope", ["", "   ", "company id", "company\tid", "company\nid"])
def test_invalid_scope_fails_during_startup(monkeypatch, invalid_scope):
    _set_required_environment(monkeypatch)
    monkeypatch.setenv("STG_DEFAULT_COMPANY_ID", invalid_scope)

    with pytest.raises((MissingSecret, ValueError), match="STG_DEFAULT_COMPANY_ID"):
        cli.MarketplaceGraph().provision_secrets(NullProvider())


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
        "https://example.services.ai.azure.example/openai",
        "https://example.services.ai.azure.example/v1",
        "https://example.services.ai.azure.example?api-version=2024-10-21",
        "example.services.ai.azure.example",
    ],
    ids=["api-path", "openai-path", "base-path", "query-only", "missing-scheme"],
)
def test_unusable_azure_endpoint_shape_fails_during_startup(monkeypatch, endpoint):
    """AzureOpenAIClient appends its own path, so anything but a bare endpoint breaks.

    The resulting error surfaces inside complete(); a node that swallows it
    degrades silently while the invocation still reports status: success, so the
    adapter has to reject the shape at startup instead.
    """

    _set_required_environment(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)

    with pytest.raises(ValueError, match="bare"):
        cli.MarketplaceGraph().provision_secrets(NullProvider())


@pytest.mark.parametrize("missing_key", cli._REQUIRED_SECRET_KEYS)
def test_missing_required_environment_fails_during_startup(monkeypatch, missing_key):
    _set_required_environment(monkeypatch)
    monkeypatch.delenv(missing_key)

    with pytest.raises(MissingSecret, match=missing_key):
        cli.MarketplaceGraph().provision_secrets(NullProvider())


def test_azure_llm_resolves_fresh_credentials_for_every_call(monkeypatch):
    configs = []

    class FakeAzureOpenAIClient:
        def __init__(self, config):
            configs.append(dict(config))

        def complete(self, messages):
            return {"content": "ok", "messages": messages}

        def stream(self, messages):
            del messages
            yield "ok"

        def bind_tools(self, tools):
            del tools
            return self

    azure_module = ModuleType("shared.services.llm.azure_openai_client")
    azure_module.AzureOpenAIClient = FakeAzureOpenAIClient
    monkeypatch.setitem(sys.modules, azure_module.__name__, azure_module)

    provider = cli._AllowlistedEnvironmentProvider()
    llm = cli.MarketplaceAzureLLM()
    assert vars(llm) == {}

    with bound_secrets(provider):
        first = _set_required_environment(monkeypatch, "first")
        llm.complete([{"role": "user", "content": "first"}])
        second = _set_required_environment(monkeypatch, "second")
        llm.complete([{"role": "user", "content": "second"}])

    assert vars(llm) == {}
    assert configs == [
        {
            "api_key": first["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": first["AZURE_OPENAI_ENDPOINT"],
            "azure_deployment": first["AZURE_OPENAI_DEPLOYMENT"],
        },
        {
            "api_key": second["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": second["AZURE_OPENAI_ENDPOINT"],
            "azure_deployment": second["AZURE_OPENAI_DEPLOYMENT"],
        },
    ]


def test_marketplace_graph_injects_llm_before_compile():
    agent = cli.MarketplaceGraph(config={"llm": object()})

    assert isinstance(agent.config["llm"], cli.MarketplaceAzureLLM)
    agent.compile()
    assert agent._nodes["main"]._inner_config["llm"] is agent.config["llm"]


def test_allowlisted_provider_hides_unrelated_environment(monkeypatch):
    _set_required_environment(monkeypatch)
    monkeypatch.setenv("UNRELATED_PLATFORM_SECRET", "must-not-be-visible")
    agent = cli.MarketplaceGraph()

    agent.provision_secrets(NullProvider())

    assert agent._secrets_provider.get("UNRELATED_PLATFORM_SECRET") is None


def test_main_passes_repo_config_to_marketplace_runner(monkeypatch):
    called = {}

    def fake_runner(agent_cls, **kwargs):
        called["agent_cls"] = agent_cls
        called.update(kwargs)

    monkeypatch.setattr(cli, "run_agent_marketplace", fake_runner)
    cli.main()

    assert called["agent_cls"] is cli.MarketplaceGraph
    assert called["agent_name"] == "RET-C2-337"
    assert called["namespace"] == "ret"
    assert called["config"]["max_retry"] == 3
