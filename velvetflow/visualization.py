# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Utilities to render a workflow DAG as a JPEG image with Unicode text support."""

import math
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Mapping

from PIL import Image, ImageDraw, ImageFont

from velvetflow.models import Edge, Node, Workflow
from velvetflow.action_registry import get_action_by_id

RGB = Tuple[int, int, int]

NODE_COLORS: Dict[str, RGB] = {
    "start": (76, 175, 80),
    "end": (156, 39, 176),
    "action": (33, 150, 243),
    "condition": (255, 152, 0),
    "switch": (121, 85, 72),
    "loop": (96, 125, 139),
    "parallel": (0, 150, 136),
}

BACKGROUND: RGB = (245, 247, 250)
EDGE_COLOR: RGB = (50, 50, 50)
TEXT_COLOR: RGB = (20, 20, 20)
PANEL_BORDER: RGB = (180, 180, 180)


def _action_node_texts(node: Node) -> Tuple[str, str]:
    """Return extra display texts for action nodes (tool name & params)."""

    if node.type != "action":
        return "", ""

    action = get_action_by_id(node.action_id or "") if node.action_id else None
    tool_name = action.get("name") if isinstance(action, dict) else None
    tool_display = tool_name or node.action_id or "未知工具"
    tool_text = f"工具: {tool_display}"

    params_payload = node.params or {}
    try:
        params_serialized = json.dumps(params_payload, ensure_ascii=False, separators=(", ", ": "))
    except Exception:
        params_serialized = str(params_payload)
    params_text = f"参数: {params_serialized}" if params_serialized else ""

    return tool_text, params_text


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Load a Unicode-capable font with common CJK fallbacks.

    The search covers Linux, macOS, and Windows font directories and also honors
    the ``VELVETFLOW_FONT`` environment variable for manual overrides.
    """

    env_font = os.getenv("VELVETFLOW_FONT")
    if env_font:
        try:
            return ImageFont.truetype(env_font, size=size)
        except Exception:
            pass

    font_dir_candidates = [
        Path(ImageFont.__file__).resolve().parent / "fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path("C:/Windows/Fonts"),
    ]
    font_name_candidates = [
        "NotoSansCJK-Regular.ttc",
        "NotoSansSC-Regular.otf",
        "NotoSans-Regular.ttf",
        "SourceHanSansCN-Regular.otf",
        "SourceHanSans-Regular.otf",
        "WenQuanYiMicroHei.ttf",
        "SimHei.ttf",
        "SimSun.ttc",
        "msyh.ttc",
        "msyh.ttf",
        "ArialUnicode.ttf",
        "MicrosoftYaHei.ttf",
        "DejaVuSans.ttf",
    ]

    for directory in font_dir_candidates:
        for name in font_name_candidates:
            candidate = directory / name
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size=size)
                except Exception:
                    continue

    for name in font_name_candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


class _ImageCanvas:
    def __init__(self, width: int, height: int, background: RGB = BACKGROUND):
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), background)
        self.draw = ImageDraw.Draw(self.image)
        self.font_regular = _load_font(16)
        self.font_small = _load_font(14)

    def _text_width(self, text: str, font: ImageFont.FreeTypeFont) -> int:
        bbox = self.draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int, line_spacing: int = 4) -> List[str]:
        if not text:
            return []

        lines: List[str] = []
        line = ""
        for ch in str(text):
            candidate = line + ch
            if self._text_width(candidate, font) <= max_width:
                line = candidate
                continue
            if line:
                lines.append(line)
                line = ch
            else:
                lines.append(candidate)
                line = ""
        if line:
            lines.append(line)
        return lines

    def _ellipsize_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
        ellipsis = "..."
        if self._text_width(text, font) <= max_width:
            return text
        if self._text_width(ellipsis, font) > max_width:
            return ""

        trimmed = text
        while trimmed and self._text_width(trimmed + ellipsis, font) > max_width:
            trimmed = trimmed[:-1]
        return trimmed + ellipsis if trimmed else ellipsis

    def measure_text_block(
        self,
        text: str,
        max_width: int,
        font: Optional[ImageFont.FreeTypeFont] = None,
        line_spacing: int = 4,
        max_height: Optional[int] = None,
    ) -> int:
        if not text:
            return 0
        font = font or self.font_regular
        lines = self._wrap_text(str(text), font, max_width, line_spacing=line_spacing)

        if max_height is not None:
            line_unit = font.size + line_spacing
            max_lines = max(0, int((max_height + line_spacing) // line_unit))
            if not max_lines:
                return 0
            lines = lines[:max_lines]

        if not lines:
            return 0

        return len(lines) * font.size + (len(lines) - 1) * line_spacing

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
        max_height: Optional[int] = None,
    ) -> int:
        if not text:
            return 0
        font = font or self.font_regular
        lines = self._wrap_text(str(text), font, max_width, line_spacing=line_spacing)

        truncated = False
        if max_height is not None:
            line_unit = font.size + line_spacing
            max_lines = max(0, int((max_height + line_spacing) // line_unit))
            if not max_lines:
                return 0
            if len(lines) > max_lines:
                truncated = True
                lines = lines[:max_lines]
            height = len(lines) * font.size + (len(lines) - 1) * line_spacing
        else:
            height = len(lines) * font.size + (len(lines) - 1) * line_spacing

        if truncated and lines:
            lines[-1] = self._ellipsize_text(lines[-1], font, max_width)

        cursor_y = y
        for idx, line in enumerate(lines):
            self.draw.text((x, cursor_y), line, font=font, fill=color)
            cursor_y += font.size
            if idx != len(lines) - 1:
                cursor_y += line_spacing

        return height


def _topological_levels(workflow: Workflow, edges: List[Edge]) -> Dict[str, int]:
    adjacency: Dict[str, List[str]] = {n.id: [] for n in workflow.nodes}
    indegree: Dict[str, int] = {n.id: 0 for n in workflow.nodes}
    for edge in edges:
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


def _median_position(neighbors: List[str], order: List[str]) -> float:
    if not neighbors:
        return float("inf")
    positions = sorted(order.index(n) for n in neighbors if n in order)
    if not positions:
        return float("inf")
    mid = len(positions) // 2
    if len(positions) % 2 == 1:
        return float(positions[mid])
    return (positions[mid - 1] + positions[mid]) / 2


def _refine_order(
    nodes_by_level: Dict[int, List[str]],
    predecessors: Dict[str, List[str]],
    successors: Dict[str, List[str]],
    max_level: int,
) -> Dict[int, List[str]]:
    order = {lvl: list(nodes) for lvl, nodes in nodes_by_level.items()}
    for _ in range(3):
        for lvl in range(1, max_level + 1):
            prev_order = order.get(lvl - 1, [])
            order[lvl] = sorted(order.get(lvl, []), key=lambda n: _median_position(predecessors[n], prev_order))
        for lvl in range(max_level - 1, -1, -1):
            next_order = order.get(lvl + 1, [])
            order[lvl] = sorted(order.get(lvl, []), key=lambda n: _median_position(successors[n], next_order))
    return order


def _build_edge_maps(nodes: List[Node], edges: List[Edge]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    incoming: Dict[str, List[str]] = {n.id: [] for n in nodes}
    outgoing: Dict[str, List[str]] = {n.id: [] for n in nodes}
    for edge in edges:
        incoming[edge.to_node].append(edge.from_node)
        outgoing[edge.from_node].append(edge.to_node)
    return incoming, outgoing


def _compute_uniform_node_height(
    workflow: Workflow,
    edges: List[Edge],
    canvas: _ImageCanvas,
    node_width: int,
    min_height: int,
) -> int:
    if not workflow.nodes:
        return min_height

    incoming, outgoing = _build_edge_maps(workflow.nodes, edges)
    content_width = node_width - 20
    top_padding = 8
    bottom_padding = 8
    spacing_after_title = 6
    spacing_before_io = 12
    io_spacing = 6

    heights: List[int] = []
    for node in workflow.nodes:
        label = f"{node.id} ({node.type})"
        detail = node.display_name or node.action_id or ""
        tool_text, params_text = _action_node_texts(node)
        inputs_text = "输入: " + (", ".join(incoming[node.id]) if incoming[node.id] else "-")
        outputs_text = "输出: " + (", ".join(outgoing[node.id]) if outgoing[node.id] else "-")

        label_height = canvas.measure_text_block(label, max_width=content_width, font=canvas.font_regular)
        detail_height = canvas.measure_text_block(detail, max_width=content_width, font=canvas.font_small)
        tool_height = canvas.measure_text_block(tool_text, max_width=content_width, font=canvas.font_small)
        params_height = canvas.measure_text_block(params_text, max_width=content_width, font=canvas.font_small)
        inputs_height = canvas.measure_text_block(inputs_text, max_width=content_width, font=canvas.font_small)
        outputs_height = canvas.measure_text_block(outputs_text, max_width=content_width, font=canvas.font_small)

        total_height = top_padding + label_height
        if detail:
            total_height += spacing_after_title + detail_height
        if tool_text:
            total_height += spacing_after_title + tool_height
        if params_text:
            total_height += spacing_after_title + params_height
        total_height += (
            spacing_before_io
            + inputs_height
            + io_spacing
            + outputs_height
            + bottom_padding
        )
        heights.append(total_height)

    return max(min_height, max(heights))


def _resolve_display_edges(workflow: Workflow) -> List[Edge]:
    """Derive edges for visualization based on the current workflow definition.

    The renderer should not rely on callers to provide an ``edges`` list. It
    reconstructs topology from node parameter bindings (via ``workflow.edges``)
    and augments it with condition branches declared on each node so that true
    and false flows are always visible in the diagram.
    """

    resolved: List[Edge] = []
    existing: Set[Tuple[str, str, Optional[str]]] = set()

    for edge in workflow.edges:
        key = (edge.from_node, edge.to_node, edge.condition)
        if key in existing:
            continue
        existing.add(key)
        resolved.append(edge)

    node_ids = {n.id for n in workflow.nodes}

    for node in workflow.nodes:
        if node.type == "condition":
            branch_pairs = [
                ("true", node.true_to_node),
                ("false", node.false_to_node),
            ]

            for branch_label, target in branch_pairs:
                if not isinstance(target, str) or target not in node_ids:
                    continue
                key = (node.id, target, branch_label)
                if key in existing:
                    continue
                existing.add(key)
                resolved.append(
                    Edge(from_node=node.id, to_node=target, condition=branch_label)
                )
        if getattr(node, "type", None) == "switch":
            cases = getattr(node, "cases", []) if isinstance(getattr(node, "cases", []), list) else []
            for case in cases:
                if not isinstance(case, Mapping):
                    continue
                target = case.get("to_node")
                if not isinstance(target, str) or target not in node_ids:
                    continue
                label = str(case.get("match")) if "match" in case else str(case.get("value"))
                key = (node.id, target, label)
                if key in existing:
                    continue
                existing.add(key)
                resolved.append(Edge(from_node=node.id, to_node=target, condition=label))
            default_to = getattr(node, "default_to_node", None)
            if isinstance(default_to, str) and default_to in node_ids:
                key = (node.id, default_to, "default")
                if key not in existing:
                    existing.add(key)
                    resolved.append(
                        Edge(from_node=node.id, to_node=default_to, condition="default")
                    )

    return resolved


def _layout_graph(
    workflow: Workflow,
    edges: List[Edge],
    node_size: Tuple[int, int],
    level_gap: int,
    node_gap: int,
    padding: int,
) -> Dict[str, object]:
    levels = _topological_levels(workflow, edges)
    max_level = max(levels.values()) if levels else 0
    nodes_by_level: Dict[int, List[str]] = {}
    predecessors: Dict[str, List[str]] = {n.id: [] for n in workflow.nodes}
    successors: Dict[str, List[str]] = {n.id: [] for n in workflow.nodes}
    for edge in edges:
        nodes_by_level.setdefault(levels[edge.from_node], []).append(edge.from_node)
        nodes_by_level.setdefault(levels[edge.to_node], []).append(edge.to_node)
        successors[edge.from_node].append(edge.to_node)
        predecessors[edge.to_node].append(edge.from_node)
    for nid, lvl in levels.items():
        nodes_by_level.setdefault(lvl, []).append(nid)

    for lvl, nodes in list(nodes_by_level.items()):
        seen = []
        for nid in nodes:
            if nid not in seen:
                seen.append(nid)
        nodes_by_level[lvl] = seen

    ordered_levels = _refine_order(nodes_by_level, predecessors, successors, max_level)

    node_width, node_height = node_size
    max_nodes_in_level = max((len(v) for v in ordered_levels.values()), default=1)
    width = padding * 2 + (max_level + 1) * node_width + max_level * level_gap
    height = padding * 2 + max_nodes_in_level * node_height + max(0, max_nodes_in_level - 1) * node_gap

    positions: Dict[str, Tuple[int, int]] = {}
    for lvl in range(max_level + 1):
        level_nodes = ordered_levels.get(lvl, [])
        for idx, nid in enumerate(level_nodes):
            x = padding + lvl * (node_width + level_gap)
            y = padding + idx * (node_height + node_gap)
            positions[nid] = (x, y)

    return {
        "positions": positions,
        "width": width,
        "height": height,
        "nodes_by_level": ordered_levels,
        "levels": levels,
        "padding": padding,
    }


def _draw_graph(
    canvas: _ImageCanvas,
    workflow: Workflow,
    edges: List[Edge],
    layout: Dict[str, object],
    node_size: Tuple[int, int],
    offset: Tuple[int, int] = (0, 0),
):
    node_width, node_height = node_size
    offset_x, offset_y = offset
    positions: Dict[str, Tuple[int, int]] = layout["positions"]  # type: ignore[assignment]
    levels: Dict[str, int] = layout.get("levels", {})  # type: ignore[assignment]
    padding: int = layout.get("padding", 50)  # type: ignore[assignment]

    incoming, outgoing = _build_edge_maps(workflow.nodes, edges)

    flyover_lanes: Dict[Tuple[int, int], int] = {}
    corridor_lanes: Dict[Tuple[int, int], int] = {}

    for edge in edges:
        start_pos = positions[edge.from_node]
        end_pos = positions[edge.to_node]
        start_x = start_pos[0] + node_width + offset_x
        start_y = start_pos[1] + node_height // 2 + offset_y
        end_x = end_pos[0] + offset_x
        end_y = end_pos[1] + node_height // 2 + offset_y

        level_from = levels.get(edge.from_node, 0)
        level_to = levels.get(edge.to_node, 0)

        is_forward_neighbor = level_to - level_from == 1 and end_x > start_x
        if is_forward_neighbor:
            lane_idx = corridor_lanes.get((level_from, level_to), 0)
            corridor_lanes[(level_from, level_to)] = lane_idx + 1

            gap = max(end_x - start_x, 1)
            margin = min(20, gap // 3)
            lane_spacing = 14
            lane_offset = min(lane_idx * lane_spacing, max(gap - 2 * margin, 0))
            mid_x = start_x + margin + lane_offset
            mid_x = min(mid_x, end_x - margin)

            canvas.draw_line(start_x, start_y, mid_x, start_y, EDGE_COLOR)
            canvas.draw_line(mid_x, start_y, mid_x, end_y, EDGE_COLOR)
            canvas.draw_arrow(mid_x, end_y, end_x, end_y, EDGE_COLOR)
        else:
            lane_idx = flyover_lanes.get((level_from, level_to), 0)
            flyover_lanes[(level_from, level_to)] = lane_idx + 1

            base_y = max(12, padding // 3)
            lane_spacing = 18
            lane_y = min(base_y + lane_idx * lane_spacing, padding - 10)

            escape_x = start_x + 12
            approach_x = end_x - 12

            canvas.draw_line(start_x, start_y, escape_x, start_y, EDGE_COLOR)
            canvas.draw_line(escape_x, start_y, escape_x, lane_y + offset_y, EDGE_COLOR)
            canvas.draw_line(escape_x, lane_y + offset_y, approach_x, lane_y + offset_y, EDGE_COLOR)
            canvas.draw_line(approach_x, lane_y + offset_y, approach_x, end_y, EDGE_COLOR)
            canvas.draw_arrow(approach_x, end_y, end_x, end_y, EDGE_COLOR)

    for node in workflow.nodes:
        x, y = positions[node.id]
        base_x = x + offset_x
        base_y = y + offset_y
        color = NODE_COLORS.get(node.type, (120, 120, 120))
        canvas.draw_rect(base_x, base_y, node_width, node_height, color, fill=True, thickness=3)
        content_width = node_width - 20
        label = f"{node.id} ({node.type})"
        detail = node.display_name or node.action_id or ""
        tool_text, params_text = _action_node_texts(node)
        inputs = incoming.get(node.id) or []
        outputs = outgoing.get(node.id) or []

        inputs_text = "输入: " + (", ".join(inputs) if inputs else "-")
        outputs_text = "输出: " + (", ".join(outputs) if outputs else "-")

        inputs_height = canvas.measure_text_block(inputs_text, max_width=content_width, font=canvas.font_small)
        outputs_height = canvas.measure_text_block(outputs_text, max_width=content_width, font=canvas.font_small)

        bottom_padding = 8
        io_spacing = 6
        bottom_block_height = inputs_height + io_spacing + outputs_height + bottom_padding

        cursor_y = base_y + 8
        available_height = node_height - bottom_block_height - (cursor_y - base_y)

        title_height = canvas.draw_text_block(
            base_x + 10,
            cursor_y,
            label,
            TEXT_COLOR,
            max_width=content_width,
            font=canvas.font_regular,
            max_height=available_height,
        )
        cursor_y += title_height

        if detail:
            cursor_y += 6
            detail_space = node_height - bottom_block_height - (cursor_y - base_y)
            if detail_space > 0:
                detail_height = canvas.draw_text_block(
                    base_x + 10,
                    cursor_y,
                    str(detail),
                    TEXT_COLOR,
                    max_width=content_width,
                    font=canvas.font_small,
                    max_height=detail_space,
                )
                cursor_y += detail_height

        if tool_text:
            cursor_y += 6
            tool_space = node_height - bottom_block_height - (cursor_y - base_y)
            if tool_space > 0:
                tool_height = canvas.draw_text_block(
                    base_x + 10,
                    cursor_y,
                    tool_text,
                    TEXT_COLOR,
                    max_width=content_width,
                    font=canvas.font_small,
                    max_height=tool_space,
                )
                cursor_y += tool_height

        if params_text:
            cursor_y += 6
            params_space = node_height - bottom_block_height - (cursor_y - base_y)
            if params_space > 0:
                params_height = canvas.draw_text_block(
                    base_x + 10,
                    cursor_y,
                    params_text,
                    TEXT_COLOR,
                    max_width=content_width,
                    font=canvas.font_small,
                    max_height=params_space,
                )
                cursor_y += params_height

        inputs_y = base_y + node_height - bottom_block_height
        canvas.draw_text_block(
            base_x + 10,
            inputs_y,
            inputs_text,
            TEXT_COLOR,
            max_width=content_width,
            font=canvas.font_small,
        )
        canvas.draw_text_block(
            base_x + 10,
            inputs_y + inputs_height + io_spacing,
            outputs_text,
            TEXT_COLOR,
            max_width=content_width,
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

    probe_canvas = _ImageCanvas(10, 10)
    edges = _resolve_display_edges(workflow)
    main_node_width = 200
    main_node_height = _compute_uniform_node_height(
        workflow, edges, probe_canvas, main_node_width, min_height=110
    )
    main_node_size = (main_node_width, main_node_height)
    main_layout = _layout_graph(
        workflow,
        edges,
        main_node_size,
        level_gap=170,
        node_gap=36,
        padding=50,
    )

    loop_panels = []
    for node in workflow.nodes:
        if node.type != "loop":
            continue
        body_workflow = _extract_loop_body(node)
        if not body_workflow:
            continue
        body_node_width = 170
        body_edges = _resolve_display_edges(body_workflow)
        body_node_height = _compute_uniform_node_height(
            body_workflow, body_edges, probe_canvas, body_node_width, min_height=100
        )
        body_node_size = (body_node_width, body_node_height)
        body_layout = _layout_graph(
            body_workflow,
            body_edges,
            body_node_size,
            level_gap=130,
            node_gap=28,
            padding=40,
        )
        panel_title = f"循环子图：{node.display_name or node.id}"
        loop_panels.append(
            {
                "loop_id": node.id,
                "workflow": body_workflow,
                "edges": body_edges,
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

    _draw_graph(canvas, workflow, edges, main_layout, main_node_size)

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
            panel["edges"],
            panel["layout"],
            panel["node_size"],
            offset=(offset_x, current_y + title_height),
        )
        current_y += panel_height + panel_gap

    return _save_jpeg(canvas.image, output_path)


__all__ = ["render_workflow_dag"]
