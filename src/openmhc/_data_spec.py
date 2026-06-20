"""DataSpec — the single declaration of the shape of participant data a Method wants.

A model sets ``data_spec = DataSpec(...)`` as a class attribute; the benchmark
materializes each participant's eligible data at that ``(resolution, window)`` and
hands it over — eagerly as a ``list`` for small specs, or streamed one participant
at a time (via a :class:`~openmhc.CohortStream`) for specs too large to hold in RAM.
This is the **single** public knob for input shape: submitters never see the loader,
the lookup, or the segment store.

Supported menu — a **closed, validated** set (anything else fails at construction):

============================  =====================  ===============================
``DataSpec``                  per-participant shape   backed by
============================  =====================  ===============================
``("hourly", "day")``         ``(n_days, 24, 38)``    ``daily_hourly_hf`` (24/day)
``("hourly", "series", N)``   ``(N, 38)``             one left-padded ``N``-hour window
``("minute", "day")``         ``(n_days, 1440, 38)``  ``daily_hf`` (1440/day, streamed)
============================  =====================  ===============================

Channels are always 0-18 raw sensor values (NaN at missing positions) and 19-37
the missingness mask (1 = missing, 0 = observed) — the same raw contract at every
spec, so a model's normalization stays model-side.

The menu is pinned deliberately: ``minute x series`` is omitted (a continuous
window of minute data is enormous and unrequested), and weekly windows stay
internal to the bundled SSL baseline rather than shipping as a public shape.

A model that doesn't set ``data_spec`` falls back to the legacy loose attributes
(``input_granularity`` / ``segment_resolution``), which the engine reads directly;
``DataSpec`` is the structured input-shape declaration for models that opt in.
"""

from __future__ import annotations

from dataclasses import dataclass

# Axis A — which segment store / DataLoader resolution backs the spec.
_RESOLUTIONS = ("hourly", "minute")
# Axis B — how a participant's eligible days are shaped before hand-off.
_WINDOWS = ("day", "series")

# The closed menu of shipped (resolution, window) pairs.
SUPPORTED_SPECS = frozenset({
    ("hourly", "day"),
    ("hourly", "series"),
    ("minute", "day"),
})


@dataclass(frozen=True)
class DataSpec:
    """The shape of participant data a :class:`~openmhc.Method` is fed.

    Construction validates against the closed menu, so an invalid spec cannot
    exist — illegal states are unrepresentable rather than caught later.

    Attributes:
        resolution: ``"hourly"`` (24 samples/day, ``daily_hourly_hf``) or
            ``"minute"`` (1440 samples/day, ``daily_hf``). Minute resolution is
            delivered by streaming — the full minute cohort does not fit in RAM.
        window: ``"day"`` (one tensor per eligible day) or ``"series"`` (one
            continuous, calendar-gap-aware, left-padded window per participant).
        window_units: for ``window="series"`` the window length in **hours**
            (e.g. 2048); must be a positive int for ``"series"`` and ``None`` for
            ``"day"``.

    Example::

        class MyMinuteModel:
            data_spec = DataSpec("minute", "day")
            def fit(self, data, labels, task_type): ...
            def predict(self, data): ...
    """

    resolution: str
    window: str
    window_units: int | None = None

    def __post_init__(self) -> None:
        if self.resolution not in _RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {_RESOLUTIONS}, got {self.resolution!r}"
            )
        if self.window not in _WINDOWS:
            raise ValueError(f"window must be one of {_WINDOWS}, got {self.window!r}")
        if (self.resolution, self.window) not in SUPPORTED_SPECS:
            raise ValueError(
                f"unsupported DataSpec {(self.resolution, self.window)}; "
                f"supported (resolution, window): {sorted(SUPPORTED_SPECS)}"
            )
        if self.window == "series":
            if not isinstance(self.window_units, int) or self.window_units <= 0:
                raise ValueError(
                    "window='series' requires window_units = positive int (hours), "
                    f"got {self.window_units!r}"
                )
        elif self.window_units is not None:
            raise ValueError(
                f"window={self.window!r} takes no window_units, got {self.window_units!r}"
            )

    @property
    def loader_resolution(self) -> str:
        """The :class:`DataLoader` resolution backing this spec (``"hourly"`` / ``"minute"``)."""
        return self.resolution

    @property
    def provider_granularity(self) -> str:
        """The ``TaskDataProvider`` granularity — which labels lookup supplies eligibility.

        ``series`` eligibility = valid days broadcast to the continuous timeline;
        ``day`` windows read the daily lookup directly.
        """
        return "series" if self.window == "series" else "daily"

    @property
    def is_streaming_required(self) -> bool:
        """True when the full cohort is too large to materialize eagerly (minute store)."""
        return self.resolution == "minute"
