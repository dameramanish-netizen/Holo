"""Desk zone topology and small enums, ported from Zone.swift."""
from __future__ import annotations
from enum import Enum


class DeskZone(Enum):
    LEFT_TOP = 0
    LEFT_BOTTOM = 1
    RIGHT_TOP = 2
    RIGHT_BOTTOM = 3

    @property
    def vertical_index(self) -> int:
        return self.value % 2

    @property
    def is_left(self) -> bool:
        return self.value < 2

    @property
    def position_name(self) -> str:
        return "Rear" if self.vertical_index == 0 else "Front"

    @property
    def short_name(self) -> str:
        return ["LR", "LF", "RR", "RF"][self.value]

    @property
    def display_name(self) -> str:
        return ["Left Rear", "Left Front", "Right Rear", "Right Front"][self.value]

    @property
    def instruction(self) -> str:
        edge = "screen" if self.vertical_index == 0 else "keyboard"
        side = "left" if self.is_left else "right"
        return f"Tap beside the laptop on the {side}, near the {edge} edge"

    @classmethod
    def all(cls):
        return [cls.LEFT_TOP, cls.LEFT_BOTTOM, cls.RIGHT_TOP, cls.RIGHT_BOTTOM]


class RejectionReason(Enum):
    WEAK_SIGNAL = "weak_signal"
    LOW_SNR = "low_signal_to_noise"
    CLIPPED = "clipped_signal"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    AMBIGUOUS = "ambiguous_zone"
    RESEMBLES_NEGATIVE = "resembles_negative_example"
    SCHEMA_MISMATCH = "schema_mismatch"
    PAUSED = "paused"

    @property
    def display_name(self) -> str:
        return {
            RejectionReason.WEAK_SIGNAL: "Signal too weak",
            RejectionReason.LOW_SNR: "Background noise too high",
            RejectionReason.CLIPPED: "Signal clipped",
            RejectionReason.OUT_OF_DISTRIBUTION: "Unlike calibrated taps",
            RejectionReason.AMBIGUOUS: "Zone ambiguous",
            RejectionReason.RESEMBLES_NEGATIVE: "Recognized non-desk sound",
            RejectionReason.SCHEMA_MISMATCH: "Profile is incompatible",
            RejectionReason.PAUSED: "Listening paused",
        }[self]


class ZoneActionKind(Enum):
    NONE = "none"
    COPY_TEXT = "copy_text"
    SPEAK_TEXT = "speak_text"
    OPEN_URL = "open_url"
    RUN_SCRIPT = "run_script"
    OPEN_APPLICATION = "open_application"
    OPEN_ITEM = "open_item"
    RUN_SHELL_COMMAND = "run_shell_command"
    SCREENSHOT_CLIPBOARD = "screenshot_clipboard"

    @property
    def display_name(self) -> str:
        return {
            ZoneActionKind.NONE: "Visual only",
            ZoneActionKind.COPY_TEXT: "Copy text",
            ZoneActionKind.SPEAK_TEXT: "Speak text",
            ZoneActionKind.OPEN_URL: "Open website",
            ZoneActionKind.RUN_SCRIPT: "Run script (.bat/.ps1/.exe)",
            ZoneActionKind.OPEN_APPLICATION: "Open application",
            ZoneActionKind.OPEN_ITEM: "Open file or folder",
            ZoneActionKind.RUN_SHELL_COMMAND: "Run shell command",
            ZoneActionKind.SCREENSHOT_CLIPBOARD: "Screenshot to clipboard",
        }[self]
