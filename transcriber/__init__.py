"""Audio-to-MusicXML transcription pipeline.

Accepts an audio recording and produces a MusicXML score by:

1. Separating the recording into stems (drums, bass, vocals, other).
2. Analysing the rhythm (tempo/beats) and pitches of each part.
3. Assembling the parts into a quantised :class:`music21.stream.Score`
   and exporting it as MusicXML.

The heavy machine-learning back-ends (Demucs for separation, Spotify's
``basic-pitch`` for polyphonic pitch detection) are optional.  When they are
not installed the pipeline transparently falls back to pure-``librosa``
implementations so that a score is always produced.
"""

from .pipeline import TranscriptionConfig, transcribe
from .rhythm import RhythmInfo, analyze_rhythm
from .stems import Stem

__all__ = [
    "TranscriptionConfig",
    "transcribe",
    "RhythmInfo",
    "analyze_rhythm",
    "Stem",
]

__version__ = "0.1.0"
