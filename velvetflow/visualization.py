"""Utilities to render a workflow DAG as a JPEG image with Unicode text support."""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from velvetflow.models import Node, Workflow

RGB = Tuple[int, int, int]

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
PANEL_BORDER: RGB = (180, 180, 180)


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Load a Unicode-capable font; fall back to Pillow's default if unavailable."""

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


class _ImageCanvas:
    def __init__(self, width: int, height: int, background: RGB = BACKGROUND):
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), background)
        self.draw = ImageDraw.Draw(self.image)
        self.font_regular = _load_font(16)
        self.font_small = _load_font(14)

    def draw_rect(self, x: int, y: int, w: int, h: int, color: RGB, fill: bool = True, thickness: int = 2):
        self.draw.rectangle(
            [x, y, x + w, y + h],
            fill=color if fill else None,
            outline=color,
            width=thickness,
        )

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, color: RGB, thickness: int = 2):
        self.draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)

    def draw_arrow(self, x1: int, y1: int, x2: int, y2: int, color: RGB, thickness: int = 2):
        self.draw_line(x1, y1, x2, y2, color, thickness)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 10
        left = (
            int(x2 - size * math.cos(angle - math.pi / 6)),
            int(y2 - size * math.sin(angle - math.pi / 6)),
        )
        right = (
            int(x2 - size * math.cos(angle + math.pi / 6)),
            int(y2 - size * math.sin(angle + math.pi / 6)),
        )
        self.draw.polygon([(x2, y2), left, right], fill=color)

    def draw_text_block(
        self,
        x: int,
        y: int,
        text: str,
        color: RGB,
        max_width: int,
        font: Optional[ImageFont.FreeTypeFont] = None,
        line_spacing: int = 4,
    ):
        if not text:
            return
        font = font or self.font_regular
        cursor_y = y
        line = ""
        for ch in str(text):
            candidate = line + ch
            bbox = self.draw.textbbox((0, 0), candidate, font=font)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_width:
                line = candidate
                continue
            if line:
                self.draw.text((x, cursor_y), line, font=font, fill=color)
                cursor_y += font.size + line_spacing
                line = ch
            else:
                # Even a single character exceeds the width; draw it and move on.
                self.draw.text((x, cursor_y), candidate, font=font, fill=color)
                cursor_y += font.size + line_spacing
                line = ""
        if line:
            self.draw.text((x, cursor_y), line, font=font, fill=color)


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

    for node in workflow.nodes:
        if node.id not in level:
            level[node.id] = 0
    return level


def _layout_graph(
    workflow: Workflow,
    node_size: Tuple[int, int],
    level_gap: int,
    node_gap: int,
    padding: int,
) -> Dict[str, object]:
    levels = _topological_levels(workflow)
    max_level = max(levels.values()) if levels else 0
    nodes_by_level: Dict[int, List[str]] = {}
    for nid, lvl in levels.items():
        nodes_by_level.setdefault(lvl, []).append(nid)

    node_width, node_height = node_size
    max_nodes_in_level = max((len(v) for v in nodes_by_level.values()), default=1)
    width = padding * 2 + (max_level + 1) * node_width + max_level * level_gap
    height = padding * 2 + max_nodes_in_level * node_height + max(0, max_nodes_in_level - 1) * node_gap

    positions: Dict[str, Tuple[int, int]] = {}
    for lvl in range(max_level + 1):
        level_nodes = nodes_by_level.get(lvl, [])
        for idx, nid in enumerate(level_nodes):
            x = padding + lvl * (node_width + level_gap)
            y = padding + idx * (node_height + node_gap)
            positions[nid] = (x, y)

    return {
        "positions": positions,
        "width": width,
        "height": height,
        "nodes_by_level": nodes_by_level,
    }


def _draw_graph(
    canvas: _ImageCanvas,
    workflow: Workflow,
    layout: Dict[str, object],
    node_size: Tuple[int, int],
    offset: Tuple[int, int] = (0, 0),
):
    node_width, node_height = node_size
    offset_x, offset_y = offset
    positions: Dict[str, Tuple[int, int]] = layout["positions"]  # type: ignore[assignment]

    for edge in workflow.edges:
        start_pos = positions[edge.from_node]
        end_pos = positions[edge.to_node]
        start_x = start_pos[0] + node_width + offset_x
        start_y = start_pos[1] + node_height // 2 + offset_y
        end_x = end_pos[0] + offset_x
        end_y = end_pos[1] + node_height // 2 + offset_y
        canvas.draw_arrow(start_x, start_y, end_x, end_y, EDGE_COLOR)

    for node in workflow.nodes:
        x, y = positions[node.id]
        base_x = x + offset_x
        base_y = y + offset_y
        color = NODE_COLORS.get(node.type, (120, 120, 120))
        canvas.draw_rect(base_x, base_y, node_width, node_height, color, fill=True, thickness=3)
        label = f"{node.id} ({node.type})"
        canvas.draw_text_block(base_x + 10, base_y + 8, label, TEXT_COLOR, max_width=node_width - 20)
        detail = node.display_name or node.action_id
        if detail:
            canvas.draw_text_block(
                base_x + 10,
                base_y + 32,
                str(detail),
                TEXT_COLOR,
                max_width=node_width - 20,
                font=canvas.font_small,
            )


def _extract_loop_body(loop_node: Node) -> Optional[Workflow]:
    params = loop_node.params or {}
    body = params.get("body_subgraph")
    if not isinstance(body, dict):
        return None

    body_nodes = body.get("nodes") or []
    body_edges = body.get("edges") or []
    if not body_nodes:
        return None

    try:
        return Workflow.model_validate(
            {"workflow_name": f"{loop_node.id}_body", "nodes": body_nodes, "edges": body_edges}
        )
    except Exception:
        return None


def _save_jpeg(image: Image.Image, output_path: str, quality: int = 90) -> str:
    output = Path(output_path)
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as fp:
        image.save(fp, format="JPEG", quality=quality, optimize=True)

    return str(output)


def render_workflow_dag(workflow: Workflow, output_path: str = "workflow_dag.jpg") -> str:
    """Render the final workflow DAG to a JPEG file and return the saved path."""

    main_node_size = (200, 80)
    main_layout = _layout_graph(workflow, main_node_size, level_gap=170, node_gap=36, padding=50)

    loop_panels = []
    for node in workflow.nodes:
        if node.type != "loop":
            continue
        body_workflow = _extract_loop_body(node)
        if not body_workflow:
            continue
        body_node_size = (170, 70)
        body_layout = _layout_graph(body_workflow, body_node_size, level_gap=130, node_gap=28, padding=40)
        panel_title = f"循环子图：{node.display_name or node.id}"
        loop_panels.append(
            {
                "loop_id": node.id,
                "workflow": body_workflow,
                "layout": body_layout,
                "node_size": body_node_size,
                "title": panel_title,
                "panel_width": body_layout["width"],
                "panel_height": body_layout["height"] + 32,
            }
        )

    panel_gap = 70
    width_candidates = [main_layout["width"]] + [panel["panel_width"] for panel in loop_panels]
    width = max(width_candidates) if width_candidates else main_layout["width"]

    extra_height = sum(panel["panel_height"] + panel_gap for panel in loop_panels)
    if loop_panels:
        extra_height += panel_gap  # top spacer before the first panel
    height = main_layout["height"] + extra_height

    canvas = _ImageCanvas(width, height)

    _draw_graph(canvas, workflow, main_layout, main_node_size)

    current_y = main_layout["height"] + (panel_gap if loop_panels else 0)
    for panel in loop_panels:
        offset_x = (width - panel["panel_width"]) // 2
        title_height = 28
        panel_height = panel["panel_height"]

        # Connect loop node to its subgraph panel for clarity.
        loop_pos = main_layout["positions"][panel["loop_id"]]
        loop_anchor_x = loop_pos[0] + main_node_size[0] // 2
        loop_anchor_y = loop_pos[1] + main_node_size[1]
        canvas.draw_arrow(loop_anchor_x, loop_anchor_y, offset_x + panel["panel_width"] // 2, current_y, EDGE_COLOR)

        canvas.draw_rect(
            offset_x,
            current_y,
            panel["panel_width"],
            panel_height,
            PANEL_BORDER,
            fill=False,
            thickness=3,
        )
        canvas.draw_text_block(
            offset_x + 12,
            current_y + 8,
            panel["title"],
            TEXT_COLOR,
            max_width=panel["panel_width"] - 24,
            font=canvas.font_regular,
        )
        _draw_graph(
            canvas,
            panel["workflow"],
            panel["layout"],
            panel["node_size"],
            offset=(offset_x, current_y + title_height),
        )
        current_y += panel_height + panel_gap

    return _save_jpeg(canvas.image, output_path)


__all__ = ["render_workflow_dag"]
