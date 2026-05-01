"""
ui/terminal_ui.py — Rich terminal dashboard

Shows system status at a glance. Refreshes every 500ms.
Never displays audio content — only levels, scores, and status.

Layout:
  ┌─ QuietReach ──────────────────────────────────────────────────────┐
  │  Status: RUNNING    Calibrated: YES    Backend: tflite            │
  │                                                                   │
  │  Threat Level  ████████░░░░░░░░░░  0.42  [ELEVATED]              │
  │  Noise Level   ████░░░░░░░░░░░░░░  LOW                           │
  │                                                                   │
  │  Consecutive:  3.0s / 8.0s required                              │
  │  Last Alert:   14:32:05  (27 min ago)                            │
  │  Cooldown:     —                                                  │
  │                                                                   │
  │  ─────────────────────────────────────────────────────────────── │
  │  🔒 Audio is never recorded or transmitted                        │
  └───────────────────────────────────────────────────────────────────┘
"""

import datetime
import time
from typing import Optional, Protocol

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── Data source protocol ──────────────────────────────────────────────────────
# The UI reads from the rest of the system through this lightweight protocol.
# Avoids importing every module into ui/ — main.py wires it up.

class UIDataSource(Protocol):
    """Anything that exposes these properties can drive the dashboard."""

    @property
    def current_threat_score(self) -> float: ...

    @property
    def current_raw_rms(self) -> float: ...

    @property
    def consecutive_seconds(self) -> float: ...

    @property
    def consecutive_seconds_required(self) -> float: ...

    @property
    def calibrated(self) -> bool: ...

    @property
    def classifier_backend(self) -> str: ...

    @property
    def in_cooldown(self) -> bool: ...

    # cooldown_remaining is a plain float attribute updated each loop tick,
    # not a computed property — protocol reflects that
    cooldown_remaining: float

    @property
    def occurrence_count(self) -> int: ...

    @property
    def last_alert_time(self) -> Optional[datetime.datetime]: ...

    @property
    def is_running(self) -> bool: ...


# ── Threat level thresholds for color/label ───────────────────────────────────
def _threat_label(score: float) -> tuple[str, str]:
    """Returns (label, rich color) for a threat score."""
    if score < 0.35:
        return "CLEAR", "green"
    elif score < 0.55:
        return "LOW", "yellow"
    elif score < 0.72:
        return "ELEVATED", "dark_orange"
    else:
        return "HIGH", "red"


def _rms_label(rms: float) -> tuple[str, str]:
    """Returns (label, color) for ambient noise level."""
    if rms < 0.01:
        return "QUIET", "green"
    elif rms < 0.04:
        return "LOW", "green"
    elif rms < 0.08:
        return "MODERATE", "yellow"
    else:
        return "LOUD", "red"


def _bar(value: float, width: int = 20, color: str = "green") -> Text:
    """Render a simple progress bar as a Rich Text object."""
    filled = int(value * width)
    filled = max(0, min(width, filled))
    bar_str = "█" * filled + "░" * (width - filled)
    t = Text()
    t.append(bar_str, style=color)
    return t


class TerminalDashboard:
    """
    Rich Live dashboard for QuietReach.

    Usage:
        dashboard = TerminalDashboard(data_source)
        dashboard.run()   # blocks — run in its own thread or as main loop
    """

    def __init__(
        self,
        source: UIDataSource,
        refresh_interval: float = 0.5,
    ) -> None:
        self._source = source
        self._refresh_interval = refresh_interval
        self._console = Console()
        self._running = False

    def run(self) -> None:
        """Start the Live dashboard. Blocks until KeyboardInterrupt."""
        self._running = True
        try:
            with Live(
                self._render(),
                console=self._console,
                refresh_per_second=int(1 / self._refresh_interval),
                screen=False,
            ) as live:
                while self._running:
                    live.update(self._render())
                    time.sleep(self._refresh_interval)
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        self._running = False

    def _render(self) -> Panel:
        s = self._source

        # ── Status row ────────────────────────────────────────────────────
        status_color = "green" if s.is_running else "red"
        status_text = "RUNNING" if s.is_running else "STOPPED"
        cal_text = "[green]YES[/green]" if s.calibrated else "[red]NO[/red]"
        backend = s.classifier_backend.upper()

        status_row = Text.assemble(
            ("Status: ", "dim"),
            (status_text, status_color),
            ("    Calibrated: ", "dim"),
        )
        # Rich markup in assemble is finicky — build status line as table row instead

        # ── Threat meter ──────────────────────────────────────────────────
        score = s.current_threat_score
        threat_lbl, threat_color = _threat_label(score)
        threat_bar = _bar(score, width=24, color=threat_color)

        # ── Noise level ───────────────────────────────────────────────────
        rms = s.current_raw_rms
        noise_lbl, noise_color = _rms_label(rms)
        # Normalize rms to ~0–1 for the bar (0.1 rms ≈ full bar)
        noise_bar = _bar(min(rms / 0.1, 1.0), width=24, color=noise_color)

        # ── Consecutive counter ───────────────────────────────────────────
        consec = s.consecutive_seconds
        required = s.consecutive_seconds_required
        consec_pct = min(consec / required, 1.0) if required > 0 else 0.0
        consec_color = "green" if consec_pct < 0.5 else "yellow" if consec_pct < 0.9 else "red"
        consec_bar = _bar(consec_pct, width=16, color=consec_color)

        # ── Last alert ────────────────────────────────────────────────────
        if s.last_alert_time is not None:
            delta = datetime.datetime.now() - s.last_alert_time
            minutes_ago = int(delta.total_seconds() / 60)
            last_alert_str = (
                f"{s.last_alert_time.strftime('%H:%M:%S')}  "
                f"({minutes_ago} min ago)"
            )
        else:
            last_alert_str = "None this session"

        # ── Cooldown ──────────────────────────────────────────────────────
        if s.in_cooldown and s.cooldown_remaining > 0:
            cooldown_str = f"[yellow]{s.cooldown_remaining:.0f}s remaining[/yellow]"
        else:
            cooldown_str = "[dim]—[/dim]"

        # ── Assemble grid ─────────────────────────────────────────────────
        grid = Table.grid(padding=(0, 2))
        grid.add_column(width=18)
        grid.add_column()

        grid.add_row(
            Text("Status", style="dim"),
            Text.assemble(
                (status_text, status_color),
                ("  ·  Calibrated: ", "dim"),
                ("YES" if s.calibrated else "NO", "green" if s.calibrated else "red"),
                ("  ·  Model: ", "dim"),
                (backend, "cyan"),
            )
        )

        grid.add_row("", "")  # spacer

        # Threat row
        threat_cell = Text()
        threat_cell.append_text(threat_bar)
        threat_cell.append(f"  {score:.2f}  ")
        threat_cell.append(f"[{threat_lbl}]", style=threat_color)
        grid.add_row(Text("Threat Level", style="bold"), threat_cell)

        # Noise row
        noise_cell = Text()
        noise_cell.append_text(noise_bar)
        noise_cell.append(f"  {noise_lbl}", style=noise_color)
        grid.add_row(Text("Noise Level", style="bold"), noise_cell)

        grid.add_row("", "")  # spacer

        # Consecutive
        consec_cell = Text()
        consec_cell.append_text(consec_bar)
        consec_cell.append(f"  {consec:.1f}s / {required:.0f}s required", style="dim")
        grid.add_row(Text("Burst", style="dim"), consec_cell)

        # Occurrence counter
        occ = s.occurrence_count
        occ_color = "green" if occ == 0 else "yellow" if occ == 1 else "red"
        occ_cell = Text()
        for i in range(3):
            occ_cell.append("● " if i < occ else "○ ", style=occ_color if i < occ else "dim")
        occ_cell.append(f" {occ}/3 occurrences in 15s window", style="dim")
        grid.add_row(Text("Occurrences", style="dim"), occ_cell)

        grid.add_row(
            Text("Last Alert", style="dim"),
            Text(last_alert_str, style="dim"),
        )
        grid.add_row(
            Text("Cooldown", style="dim"),
            Text.from_markup(cooldown_str),
        )

        grid.add_row("", "")  # spacer

        # Privacy reminder — always visible, always at the bottom
        reminder = Text("🔒  Audio is never recorded or transmitted", style="dim italic")
        grid.add_row("", reminder)

        return Panel(
            grid,
            title="[bold cyan]QuietReach[/bold cyan]",
            subtitle="[dim]Ctrl+C to stop[/dim]",
            border_style="cyan",
        )