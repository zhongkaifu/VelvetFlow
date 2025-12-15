"""Integration helpers to register Composio tools with VelvetFlow."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

from tools.base import Tool
from tools.registry import get_registered_tool, register_tool
from velvetflow.action_registry import register_dynamic_actions


def _load_default_toolset() -> Any:
    """Instantiate the default Composio toolset.

    Import is deferred so that Composio remains an optional dependency.
    """

    try:
        import composio_openai  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time dependency
        raise ImportError(
            "composio_openai is required to auto-load Composio tools. Install via 'pip install composio-openai'."
        ) from exc

    candidates: list[Any] = []

    # Prefer top-level export first (works for most package versions).
    for name in ("ComposioToolSet", "ComposioToolset"):
        if hasattr(composio_openai, name):
            candidates.append(getattr(composio_openai, name))

    # Some releases expose the class from the toolset module instead of __init__.
    try:  # pragma: no cover - exercised by integration tests with fake modules
        from composio_openai.toolset import ComposioToolSet as ToolSetFromModule  # type: ignore

        candidates.append(ToolSetFromModule)
    except Exception:
        pass

    # Newer releases may tuck the class behind other submodules; scan lazily to avoid
    # hard failures when the class is present but unexported.
    try:  # pragma: no cover - exercised by integration tests with fake modules
        import importlib
        import pkgutil

        if hasattr(composio_openai, "__path__"):
            for module_info in pkgutil.walk_packages(composio_openai.__path__, composio_openai.__name__ + "."):
                try:
                    module = importlib.import_module(module_info.name)
                except Exception:
                    continue

                for name in ("ComposioToolSet", "ComposioToolset"):
                    if hasattr(module, name):
                        candidates.append(getattr(module, name))
                if candidates:
                    break
    except Exception:
        pass

    for candidate in candidates:
        if callable(candidate):
            return candidate()

    # Some packages expose a provider() factory that can vend the toolset
    # without exporting the class directly. Use soft probing to avoid
    # breaking when the provider requires runtime configuration.
    provider = getattr(composio_openai, "provider", None)
    if callable(provider):  # pragma: no cover - exercised by integration tests with fake modules
        try:
            provider_instance = provider()

            if provider_instance is not None:
                if hasattr(provider_instance, "get_toolset"):
                    toolset = provider_instance.get_toolset()
                    if toolset:
                        return toolset

                for attr in ("toolset", "tool_set"):
                    if hasattr(provider_instance, attr):
                        toolset = getattr(provider_instance, attr)
                        if toolset:
                            return toolset
        except Exception:
            pass

    raise ImportError(
        "Unable to locate composio_openai.ComposioToolSet. Available attributes: "
        f"{', '.join(sorted(attr for attr in dir(composio_openai) if not attr.startswith('_')))}"
    )


def collect_composio_tool_specs(
    *, toolset: Any | None = None, selected_actions: Sequence[Any] | None = None
) -> List[Mapping[str, Any]]:
    """Return Composio OpenAI tool specifications.

    Parameters
    ----------
    toolset:
        Optional Composio toolset instance. If omitted, ``composio_openai.ComposioToolSet``
        is instantiated lazily.
    selected_actions:
        Optional list of actions to include. Items are forwarded directly to
        ``toolset.get_openai_tools(actions=...)``.

    Returns
    -------
    list of mapping
        Raw OpenAI-compatible tool definitions.
    """

    resolved_toolset = toolset or _load_default_toolset()
    return list(_iter_openai_tools(resolved_toolset, selected_actions))


def _iter_openai_tools(toolset: Any, actions: Sequence[Any] | None) -> Iterable[Mapping[str, Any]]:
    if hasattr(toolset, "get_openai_tools"):
        getter = getattr(toolset, "get_openai_tools")
        return getter(actions=actions) if actions is not None else getter()

    raise ValueError("Provided toolset does not expose get_openai_tools(actions=...) API")


def _invoke_composio_action(toolset: Any, action: Any, params: MutableMapping[str, Any]) -> Any:
    if hasattr(toolset, "execute_action"):
        return toolset.execute_action(action, params)
    if hasattr(toolset, "run_action"):
        return toolset.run_action(action, params)
    if hasattr(toolset, "run"):
        return toolset.run(action, params)

    raise ValueError("Provided toolset cannot execute actions (expected execute_action/run_action/run)")


def register_composio_tools(
    *,
    toolset: Any | None = None,
    namespace: str = "composio",
    selected_actions: Sequence[Any] | None = None,
    register_actions: bool = True,
) -> List[str]:
    """Register Composio tools into the global business registry.

    Parameters
    ----------
    toolset:
        Optional Composio toolset instance. If omitted, ``composio_openai.ComposioToolSet``
        is instantiated lazily.
    namespace:
        Prefix used for registered tool names (defaults to ``"composio"``).
    selected_actions:
        Optional list of actions to include. Items are forwarded directly to
        ``toolset.get_openai_tools(actions=...)``.
    register_actions:
        When ``True`` (default), business actions are added into the global
        action registry so that workflows can reference them by ``action_id``.

    Returns
    -------
    list of str
        Names of tools registered into ``GLOBAL_TOOL_REGISTRY``.
    """

    resolved_toolset = toolset or _load_default_toolset()
    tool_specs = collect_composio_tool_specs(toolset=resolved_toolset, selected_actions=selected_actions)

    registered_tool_names: List[str] = []
    new_actions: List[Mapping[str, Any]] = []

    for spec in tool_specs:
        if not isinstance(spec, Mapping):
            continue

        func_spec = spec.get("function") if isinstance(spec.get("function"), Mapping) else None
        if not func_spec:
            continue

        func_name = func_spec.get("name") or ""
        if not func_name:
            continue

        description = func_spec.get("description") or func_name
        parameters = func_spec.get("parameters")
        if not isinstance(parameters, Mapping):
            continue

        action_identifier = (
            spec.get("action")
            or spec.get("name")
            or func_spec.get("name")
            or func_spec.get("operation_id")
            or func_name
        )

        tool_name = f"{namespace}.{func_name}"
        action_id = f"{tool_name}.v1"

        if not get_registered_tool(tool_name):
            register_tool(
                Tool(
                    name=tool_name,
                    description=description,
                    function=lambda _action=action_identifier, **kwargs: _invoke_composio_action(
                        resolved_toolset, _action, kwargs
                    ),
                    args_schema=parameters,
                )
            )
        registered_tool_names.append(tool_name)

        if register_actions:
            new_actions.append(
                {
                    "action_id": action_id,
                    "name": func_name,
                    "domain": namespace,
                    "description": description,
                    "arg_schema": parameters,
                    "tool_name": tool_name,
                    "tags": ["composio", namespace],
                    "enabled": True,
                }
            )

    if register_actions and new_actions:
        register_dynamic_actions(new_actions)

    return registered_tool_names


__all__ = ["collect_composio_tool_specs", "register_composio_tools"]
