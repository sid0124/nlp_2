"""Route modules, one per area of the dashboard.

Split by what the client asks for rather than by HTTP verb: ``system`` answers
"what is this server and what can it do", ``dashboard`` answers the aggregate
panels, ``papers`` answers everything about an individual paper, and
``analytics`` answers the research-intelligence panels (gaps, methodology,
citations).
"""

from __future__ import annotations

__all__ = ["analytics", "dashboard", "papers", "system"]
