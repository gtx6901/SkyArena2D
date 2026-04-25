from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventLog:
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event_type: str, **payload: Any) -> None:
        item = {"type": event_type}
        item.update(payload)
        self.events.append(item)

    def clear(self) -> None:
        self.events.clear()
