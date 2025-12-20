"""Dashboard - Terminal display for real-time meeting intelligence"""
import os
import sys
import time
from typing import Optional

import psutil


# ANSI color codes for terminal
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


# Vibe colors
VIBE_COLORS = {
    "Excited": Colors.GREEN,
    "Frustrated": Colors.RED,
    "Uncertain": Colors.YELLOW,
    "Confident": Colors.BLUE,
    "Engaged": Colors.CYAN,
    "Neutral": Colors.WHITE,
}


def clear_line():
    """Clear current terminal line"""
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.flush()


def display_header():
    """Display dashboard header"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Real-Time Meeting Intelligence Agent{Colors.RESET}")
    print(f"{Colors.DIM}  Powered by LFM2-Audio | 100% Local Processing{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")


def display_status(message: str):
    """Display a status message"""
    print(f"{Colors.DIM}[STATUS]{Colors.RESET} {message}")


def display_update(
    transcript: str,
    vibe: str,
    vibe_emoji: str,
    confidence: float,
    context_preview: Optional[str] = None,
):
    """
    Display real-time update on single line (overwriting previous).

    Args:
        transcript: Latest transcription text
        vibe: Dominant emotion category
        vibe_emoji: Emoji for the emotion
        confidence: RAG match confidence (0-1)
        context_preview: Optional preview of matched RAG context
    """
    # Get system stats
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    # Format confidence
    conf_pct = confidence * 100
    if conf_pct >= 50:
        conf_color = Colors.GREEN
    elif conf_pct >= 25:
        conf_color = Colors.YELLOW
    else:
        conf_color = Colors.RED

    # Get vibe color
    vibe_color = VIBE_COLORS.get(vibe, Colors.WHITE)

    # Truncate transcript for display
    max_transcript = 40
    if len(transcript) > max_transcript:
        transcript_display = transcript[:max_transcript] + "..."
    else:
        transcript_display = transcript

    # Build status line
    status = (
        f"\r{Colors.DIM}[CPU:{cpu:4.0f}% RAM:{ram:4.0f}%]{Colors.RESET} "
        f"{vibe_color}{vibe_emoji} {vibe:12}{Colors.RESET} "
        f"{conf_color}Conf:{conf_pct:3.0f}%{Colors.RESET} "
        f"{Colors.CYAN}│{Colors.RESET} {transcript_display}"
    )

    # Print without newline, flush immediately
    sys.stdout.write(status + " " * 10)  # Pad to clear old content
    sys.stdout.flush()


def display_transcript_line(timestamp: str, transcript: str, vibe: str, confidence: float):
    """Display a full transcript line (for logging/review mode)"""
    conf_pct = confidence * 100
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {transcript} "
          f"{Colors.DIM}| {vibe} | {conf_pct:.0f}%{Colors.RESET}")


def display_summary(total_chunks: int, dominant_vibes: dict, avg_confidence: float):
    """Display meeting summary"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Meeting Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"  Total chunks processed: {total_chunks}")
    print(f"  Average RAG confidence: {avg_confidence*100:.1f}%")
    print(f"  Vibe breakdown:")
    for vibe, count in sorted(dominant_vibes.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = (count / total_chunks) * 100
            print(f"    {vibe}: {count} ({pct:.1f}%)")
    print()
