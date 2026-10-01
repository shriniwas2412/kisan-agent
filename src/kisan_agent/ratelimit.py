"""Per-user daily quota, so one user (or a bot) can't run up the API bill on the public demo."""

from __future__ import annotations

from datetime import date
from typing import Callable


class DailyQuota:
    def __init__(self, limit: int = 20, today: Callable[[], date] = date.today):
        self.limit = limit
        self._today = today
        self._day = today()
        self._used: dict[int, int] = {}

    def allow(self, user_id: int) -> bool:
        """Count one request; False once the user has used today's quota."""
        if self._today() != self._day:  # new day: reset everyone
            self._day, self._used = self._today(), {}
        if self._used.get(user_id, 0) >= self.limit:
            return False
        self._used[user_id] = self._used.get(user_id, 0) + 1
        return True

    def remaining(self, user_id: int) -> int:
        return max(self.limit - self._used.get(user_id, 0), 0)
