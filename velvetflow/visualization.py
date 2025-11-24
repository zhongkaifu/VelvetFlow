"""Utilities to render a workflow DAG as a JPEG image without external dependencies."""

import math
import io
from pathlib import Path
from typing import Dict, List, Tuple

from velvetflow.models import Workflow

RGB = Tuple[int, int, int]

# 5x7 bitmap font (monospace) for common ASCII characters.
BITMAP_FONT: Dict[str, Tuple[str, ...]] = {
    "A": (
        " 1 ",
        "1 1",
        "111",
        "1 1",
        "1 1",
        "   ",
        "   ",
    ),
    "B": (
        "11 ",
        "1 1",
        "11 ",
        "1 1",
        "11 ",
        "   ",
        "   ",
    ),
    "C": (
        " 11",
        "1  ",
        "1  ",
        "1  ",
        " 11",
        "   ",
        "   ",
    ),
    "D": (
        "11 ",
        "1 1",
        "1 1",
        "1 1",
        "11 ",
        "   ",
        "   ",
    ),
    "E": (
        "111",
        "1  ",
        "11 ",
        "1  ",
        "111",
        "   ",
        "   ",
    ),
    "F": (
        "111",
        "1  ",
        "11 ",
        "1  ",
        "1  ",
        "   ",
        "   ",
    ),
    "G": (
        " 11",
        "1  ",
        "1  ",
        "1 1",
        " 11",
        "   ",
        "   ",
    ),
    "H": (
        "1 1",
        "1 1",
        "111",
        "1 1",
        "1 1",
        "   ",
        "   ",
    ),
    "I": (
        "111",
        " 1 ",
        " 1 ",
        " 1 ",
        "111",
        "   ",
        "   ",
    ),
    "J": (
        " 11",
        "  1",
        "  1",
        "1 1",
        " 1 ",
        "   ",
        "   ",
    ),
    "K": (
        "1 1",
        "1 1",
        "11 ",
        "1 1",
        "1 1",
        "   ",
        "   ",
    ),
    "L": (
        "1  ",
        "1  ",
        "1  ",
        "1  ",
        "111",
        "   ",
        "   ",
    ),
    "M": (
        "1 1",
        "111",
        "111",
        "1 1",
        "1 1",
        "   ",
        "   ",
    ),
    "N": (
        "1 1",
        "111",
        "111",
        "111",
        "1 1",
        "   ",
        "   ",
    ),
    "O": (
        "111",
        "1 1",
        "1 1",
        "1 1",
        "111",
        "   ",
        "   ",
    ),
    "P": (
        "111",
        "1 1",
        "111",
        "1  ",
        "1  ",
        "   ",
        "   ",
    ),
    "Q": (
        "111",
        "1 1",
        "1 1",
        "111",
        "  1",
        "   ",
        "   ",
    ),
    "R": (
        "111",
        "1 1",
        "111",
        "1 1",
        "1 1",
        "   ",
        "   ",
    ),
    "S": (
        " 11",
        "1  ",
        "111",
        "  1",
        "11 ",
        "   ",
        "   ",
    ),
    "T": (
        "111",
        " 1 ",
        " 1 ",
        " 1 ",
        " 1 ",
        "   ",
        "   ",
    ),
    "U": (
        "1 1",
        "1 1",
        "1 1",
        "1 1",
        "111",
        "   ",
        "   ",
    ),
    "V": (
        "1 1",
        "1 1",
        "1 1",
        "1 1",
        " 1 ",
        "   ",
        "   ",
    ),
    "W": (
        "1 1",
        "1 1",
        "111",
        "111",
        "1 1",
        "   ",
        "   ",
    ),
    "X": (
        "1 1",
        "1 1",
        " 1 ",
        "1 1",
        "1 1",
        "   ",
        "   ",
    ),
    "Y": (
        "1 1",
        "1 1",
        "111",
        " 1 ",
        " 1 ",
        "   ",
        "   ",
    ),
    "Z": (
        "111",
        "  1",
        " 1 ",
        "1  ",
        "111",
        "   ",
        "   ",
    ),
    "0": (
        "111",
        "1 1",
        "1 1",
        "1 1",
        "111",
        "   ",
        "   ",
    ),
    "1": (
        " 1 ",
        "11 ",
        " 1 ",
        " 1 ",
        "111",
        "   ",
        "   ",
    ),
    "2": (
        "111",
        "  1",
        "111",
        "1  ",
        "111",
        "   ",
        "   ",
    ),
    "3": (
        "111",
        "  1",
        "111",
        "  1",
        "111",
        "   ",
        "   ",
    ),
    "4": (
        "1 1",
        "1 1",
        "111",
        "  1",
        "  1",
        "   ",
        "   ",
    ),
    "5": (
        "111",
        "1  ",
        "111",
        "  1",
        "111",
        "   ",
        "   ",
    ),
    "6": (
        "111",
        "1  ",
        "111",
        "1 1",
        "111",
        "   ",
        "   ",
    ),
    "7": (
        "111",
        "  1",
        " 1 ",
        "1  ",
        "1  ",
        "   ",
        "   ",
    ),
    "8": (
        "111",
        "1 1",
        "111",
        "1 1",
        "111",
        "   ",
        "   ",
    ),
    "9": (
        "111",
        "1 1",
        "111",
        "  1",
        "111",
        "   ",
        "   ",
    ),
    "-": (
        "   ",
        "   ",
        "111",
        "   ",
        "   ",
        "   ",
        "   ",
    ),
    "_": (
        "   ",
        "   ",
        "   ",
        "   ",
        "111",
        "   ",
        "   ",
    ),
    " ": (
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
    ),
    "?": (
        "111",
        "  1",
        " 1 ",
        "   ",
        " 1 ",
        "   ",
        "   ",
    ),
    "(": (
        " 1 ",
        " 1 ",
        " 1 ",
        " 1 ",
        " 1 ",
        "   ",
        "   ",
    ),
    ")": (
        "1  ",
        "1  ",
        "1  ",
        "1  ",
        "1  ",
        "   ",
        "   ",
    ),
    ":": (
        "   ",
        " 1 ",
        "   ",
        " 1 ",
        "   ",
        "   ",
        "   ",
    ),
}

NODE_COLORS: Dict[str, RGB] = {
    "start": (76, 175, 80),
    "end": (156, 39, 176),
    "action": (33, 150, 243),
    "condition": (255, 152, 0),
    "loop": (96, 125, 139),
    "parallel": (0, 150, 136),
}

BACKGROUND: RGB = (245, 247, 250)
EDGE_COLOR: RGB = (50, 50, 50)
TEXT_COLOR: RGB = (20, 20, 20)


def _safe_char(ch: str) -> str:
    """Return a character that exists in the bitmap font."""
    if ch in BITMAP_FONT:
        return ch
    if ch.upper() in BITMAP_FONT:
        return ch.upper()
    return "?"


def _encode_text(text: str) -> str:
    return "".join(_safe_char(ch) for ch in text)


def _topological_levels(workflow: Workflow) -> Dict[str, int]:
    adjacency: Dict[str, List[str]] = {n.id: [] for n in workflow.nodes}
    indegree: Dict[str, int] = {n.id: 0 for n in workflow.nodes}
    for edge in workflow.edges:
        adjacency[edge.from_node].append(edge.to_node)
        indegree[edge.to_node] += 1

    queue: List[str] = [nid for nid, deg in indegree.items() if deg == 0]
    level: Dict[str, int] = {nid: 0 for nid in queue}
    while queue:
        current = queue.pop(0)
        for nxt in adjacency[current]:
            level[nxt] = max(level.get(nxt, 0), level[current] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return level


class _ImageCanvas:
    def __init__(self, width: int, height: int, background: RGB = BACKGROUND):
        self.width = width
        self.height = height
        self.buffer = bytearray([background[0], background[1], background[2]] * width * height)

    def _index(self, x: int, y: int) -> int:
        return (y * self.width + x) * 3

    def draw_pixel(self, x: int, y: int, color: RGB):
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = self._index(x, y)
            self.buffer[idx : idx + 3] = bytes(color)

    def draw_rect(self, x: int, y: int, w: int, h: int, color: RGB, fill: bool = True, thickness: int = 2):
        for i in range(h):
            for j in range(w):
                on_border = i < thickness or j < thickness or i >= h - thickness or j >= w - thickness
                if fill or on_border:
                    self.draw_pixel(x + j, y + i, color if fill else (color if on_border else BACKGROUND))

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, color: RGB, thickness: int = 2):
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        x, y = x1, y1
        while True:
            for tx in range(-thickness, thickness + 1):
                for ty in range(-thickness, thickness + 1):
                    self.draw_pixel(x + tx, y + ty, color)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def draw_arrow(self, x1: int, y1: int, x2: int, y2: int, color: RGB):
        self.draw_line(x1, y1, x2, y2, color)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 8
        left = (
            int(x2 - size * math.cos(angle - math.pi / 6)),
            int(y2 - size * math.sin(angle - math.pi / 6)),
        )
        right = (
            int(x2 - size * math.cos(angle + math.pi / 6)),
            int(y2 - size * math.sin(angle + math.pi / 6)),
        )
        self.draw_line(x2, y2, left[0], left[1], color)
        self.draw_line(x2, y2, right[0], right[1], color)

    def draw_text(self, x: int, y: int, text: str, color: RGB):
        cursor_x = x
        for ch in _encode_text(text):
            glyph = BITMAP_FONT.get(ch.upper(), BITMAP_FONT.get(ch, BITMAP_FONT["?" if "?" in BITMAP_FONT else " "]))
            for row_idx, row in enumerate(glyph):
                for col_idx, bit in enumerate(row):
                    if bit != "1":
                        continue
                    self.draw_pixel(cursor_x + col_idx, y + row_idx, color)
            cursor_x += 4  # 3px glyph + 1px spacing


def _save_jpeg(buffer: bytearray, width: int, height: int, output_path: str, quality: int = 90) -> str:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "Saving workflow DAG as JPEG requires the optional Pillow dependency (pip install pillow)."
        ) from exc

    output = Path(output_path)
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)

    img = Image.frombytes("RGB", (width, height), bytes(buffer))
    with io.BytesIO() as stream:
        img.save(stream, format="JPEG", quality=quality, optimize=True)
        output.write_bytes(stream.getvalue())

    return str(output)


def render_workflow_dag(workflow: Workflow, output_path: str = "workflow_dag.jpg") -> str:
    """Render the final workflow DAG to a JPEG file and return the saved path."""

    levels = _topological_levels(workflow)
    max_level = max(levels.values()) if levels else 0
    nodes_by_level: Dict[int, List[str]] = {}
    for nid, lvl in levels.items():
        nodes_by_level.setdefault(lvl, []).append(nid)

    node_width, node_height = 200, 70
    level_gap, node_gap = 160, 30
    padding = 40

    max_nodes_in_level = max((len(v) for v in nodes_by_level.values()), default=1)
    width = padding * 2 + (max_level + 1) * node_width + max_level * level_gap
    height = padding * 2 + max_nodes_in_level * node_height + max(0, max_nodes_in_level - 1) * node_gap

    canvas = _ImageCanvas(width, height)

    positions: Dict[str, Tuple[int, int]] = {}
    for lvl in range(max_level + 1):
        level_nodes = nodes_by_level.get(lvl, [])
        for idx, nid in enumerate(level_nodes):
            x = padding + lvl * (node_width + level_gap)
            y = padding + idx * (node_height + node_gap)
            positions[nid] = (x, y)

    for edge in workflow.edges:
        start_pos = positions[edge.from_node]
        end_pos = positions[edge.to_node]
        start_x = start_pos[0] + node_width
        start_y = start_pos[1] + node_height // 2
        end_x = end_pos[0]
        end_y = end_pos[1] + node_height // 2
        canvas.draw_arrow(start_x, start_y, end_x, end_y, EDGE_COLOR)

    for node in workflow.nodes:
        x, y = positions[node.id]
        color = NODE_COLORS.get(node.type, (120, 120, 120))
        canvas.draw_rect(x, y, node_width, node_height, color, fill=True, thickness=3)
        label = f"{node.id} ({node.type})"
        canvas.draw_text(x + 8, y + 10, label, TEXT_COLOR)
        if node.display_name:
            canvas.draw_text(x + 8, y + 30, node.display_name[:18], TEXT_COLOR)
        elif node.action_id:
            canvas.draw_text(x + 8, y + 30, node.action_id[:18], TEXT_COLOR)

    return _save_jpeg(canvas.buffer, width, height, output_path)


__all__ = ["render_workflow_dag"]
