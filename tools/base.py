# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Base definitions for simple callable tools."""
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass
class Tool:
    """Metadata wrapper around a callable tool.

    提供同步 `function` 与可选异步 `async_function`，便于在执行阶段根据需求切换调用模式。
    """

    name: str
    description: str
    function: Callable[..., Any]
    args_schema: Optional[Mapping[str, Any]] = None
    async_function: Optional[Callable[..., Any]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - thin wrapper
        return self.function(*args, **kwargs)


__all__ = ["Tool"]
