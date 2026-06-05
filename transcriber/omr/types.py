"""Shared data structures for the OMR pipeline.

These are deliberately lightweight: the engines either emit MusicXML (which we
parse into a :class:`music21.stream.Score`) or, in the built-in recogniser,
emit :class:`OMRNote` events that we assemble.  The pipeline always exposes a
music21 score as the canonical result, but the structured types below make the
intermediate recognition results inspectable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OMRNote:
    """A single recognised note head.

    Positions are expressed in *quarter lengths* within a part timeline, which
    is what music21 consumes directly.

    Attributes:
        pitch: MIDI pitch number (0-127).
        onset: Onset position in quarter lengths from the start of the part.
        duration: Note duration in quarter lengths.
        voice: 1-based voice index within the staff.
        staff: 1-based staff index (1 = top staff of a system).
        accidental: ``"sharp"``, ``"flat"``, ``"natural"`` or ``None``.
        confidence: Recogniser confidence in ``[0, 1]`` (1 = certain).
    """

    pitch: int
    onset: float
    duration: float
    voice: int = 1
    staff: int = 1
    accidental: str | None = None
    confidence: float = 1.0


@dataclass
class StaffSystem:
    """Geometry of one detected staff (five lines) on a page.

    Attributes:
        line_ys: Pixel y-coordinates of the five staff lines, top to bottom.
        x_start: Left edge of the staff in pixels.
        x_end: Right edge of the staff in pixels.
        staff_space: Mean vertical gap between adjacent staff lines (pixels).
    """

    line_ys: list[float]
    x_start: int
    x_end: int
    staff_space: float

    @property
    def top_line_y(self) -> float:
        return self.line_ys[0]

    @property
    def bottom_line_y(self) -> float:
        return self.line_ys[-1]

    def step_at(self, y: float) -> int:
        """Return the diatonic *staff step* of a note head centred at ``y``.

        Steps are measured in half-staff-space units from the **bottom** staff
        line (step 0).  Each diatonic degree (one line/space) is one step.
        Larger steps are higher on the page (smaller ``y``).
        """
        half_space = self.staff_space / 2.0
        return int(round((self.bottom_line_y - y) / half_space))


@dataclass
class RecognizedScore:
    """A structured recognition result before MusicXML assembly.

    Attributes:
        notes: All recognised notes across staves/pages.
        systems: Detected staff systems (geometry), useful for debugging.
        page_count: Number of pages processed.
        title: Suggested score title.
    """

    notes: list[OMRNote] = field(default_factory=list)
    systems: list[StaffSystem] = field(default_factory=list)
    page_count: int = 1
    title: str = "Optical transcription"
    key_sharps: int = 0  # signed key signature: +N sharps, -N flats
