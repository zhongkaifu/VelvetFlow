import { describeNode, summarizeValue, normalizeWorkflow, normalizeEdge, collectRefsFromValue } from "./workflow-utils.js";

export const NODE_WIDTH = 320;

export function wrapText(ctx, text, maxWidth, font = "15px Inter") {
  if (!text) return [];
  ctx.save();
  ctx.font = font;
  const words = String(text).split("");
  const lines = [];
  let line = "";
  words.forEach((ch) => {
    const candidate = line + ch;
    if (ctx.measureText(candidate).width <= maxWidth) {
      line = candidate;
    } else {
      if (line) lines.push(line);
      line = ch;
    }
  });
  if (line) lines.push(line);
  ctx.restore();
  return lines.length ? lines : [text];
}

export function estimateNodeHeight(node, ctx, actionCatalog, lastRunResults) {
  const { toolLabel, runtimeInputs, runtimeOutputs } = describeNode(node, actionCatalog, lastRunResults);
  const contentLines = [];
  if (toolLabel) contentLines.push(`工具: ${toolLabel}`);

  if (runtimeInputs !== undefined) {
    contentLines.push(`运行入参: ${summarizeValue(runtimeInputs)}`);
  }
  if (runtimeOutputs !== undefined && Object.keys(runtimeOutputs).length > 0) {
    contentLines.push(`运行结果: ${summarizeValue(runtimeOutputs)}`);
  }

  const wrappedLines = contentLines.flatMap((line) => wrapText(ctx, line, NODE_WIDTH - 28, "15px Inter"));
  const baseHeight = 90;
  const dynamicHeight = wrappedLines.length * 18;
  return baseHeight + dynamicHeight;
}

export function layoutNodes(workflow, options = {}) {
  const { nodes = [], edges = [] } = workflow;
  if (!nodes.length) return { positions: {}, meta: { height: 0, width: 0 } };

  const width = options.width || 1200;
  const heightHint = options.heightHint || 720;

  const nodeOrder = nodes.map((n) => n.id);
  const indegree = {};
  const level = {};
  const outgoing = {};
  const incoming = {};

  nodeOrder.forEach((id) => {
    indegree[id] = 0;
    level[id] = 0;
    outgoing[id] = [];
    incoming[id] = [];
  });

  edges.forEach((edge) => {
    const from = edge.from_node || edge.from;
    const to = edge.to_node || edge.to;
    if (!from || !to) return;
    if (indegree[to] === undefined) indegree[to] = 0;
    if (level[to] === undefined) level[to] = 0;
    outgoing[from] = outgoing[from] || [];
    outgoing[from].push(to);
    incoming[to] = incoming[to] || [];
    incoming[to].push(from);
    indegree[to] += 1;
  });

  const queue = nodeOrder.filter((id) => indegree[id] === 0);
  const visited = new Set(queue);

  while (queue.length) {
    const current = queue.shift();
    const nextLevel = level[current] + 1;
    (outgoing[current] || []).forEach((neighbor) => {
      level[neighbor] = Math.max(level[neighbor] || 0, nextLevel);
      indegree[neighbor] -= 1;
      if (indegree[neighbor] === 0 && !visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    });
  }

  const maxKnownLevel = Math.max(0, ...Object.values(level));
  nodeOrder.forEach((id) => {
    if (!visited.has(id)) {
      level[id] = maxKnownLevel + 1;
    }
  });

  const layers = {};
  nodeOrder.forEach((id) => {
    const layerIndex = level[id] || 0;
    if (!layers[layerIndex]) layers[layerIndex] = [];
    layers[layerIndex].push(id);
  });

  const layerOrderIndex = {};
  const sortedLayers = {};
  const layerKeys = Object.keys(layers)
    .map((n) => Number(n))
    .sort((a, b) => a - b);

  layerKeys.forEach((layerIndex, idx) => {
    const row = [...(layers[layerIndex] || [])];
    sortedLayers[layerIndex] = row;

    if (idx > 0) {
      const adjusted = sortedLayers[layerIndex]
        .map((id, order) => {
          const children = outgoing[id] || [];
          const childRanks = children.map((c) => layerOrderIndex[c]).filter((v) => v !== undefined);
          const childScore = childRanks.length
            ? childRanks.reduce((a, b) => a + b, 0) / childRanks.length
            : order;
          return { id, score: (order + childScore) / 2 };
        })
        .sort((a, b) => a.score - b.score)
        .map((item, order) => {
          layerOrderIndex[item.id] = order;
          return item.id;
        });

      sortedLayers[layerIndex] = adjusted;
    }
  });

  const topPadding = 80;
  const bottomPadding = 60;
  const minGapX = 48;
  const minGapY = 42;
  const heightById = nodes.reduce((acc, node) => {
    acc[node.id] = estimateNodeHeight(node, options.ctx || options.measureCtx, options.actionCatalog, options.lastRunResults);
    return acc;
  }, {});
  const layerHeights = layerKeys.map((k) => Math.max(...(sortedLayers[k] || layers[k]).map((id) => heightById[id]), 120));
  const safeHeight = Math.max(400, heightHint - topPadding - bottomPadding);
  const gapY = Math.max(minGapY, Math.min(160, safeHeight / Math.max(1, layerKeys.length)));

  const positions = {};
  let currentY = topPadding;
  layerKeys.forEach((layerIndex) => {
    const row = sortedLayers[layerIndex] || layers[layerIndex];
    const rowWidth = row.length * NODE_WIDTH;
    const gapCount = Math.max(0, row.length - 1);
    const gapX = Math.max(minGapX, Math.min(200, (width - rowWidth) / Math.max(1, gapCount) - 12));
    const totalWidth = rowWidth + gapCount * gapX;
    const startX = width / 2 - totalWidth / 2 + NODE_WIDTH / 2;
    const rowHeight = layerHeights[layerKeys.indexOf(layerIndex)] || 140;
    const y = currentY + rowHeight / 2;
    row.forEach((id, idx) => {
      positions[id] = {
        x: startX + idx * (NODE_WIDTH + gapX),
        y,
      };
    });
    currentY += rowHeight + gapY;
  });

  const padding = 48;
  const maxY = Math.max(...Object.entries(positions).map(([id, pos]) => pos.y + (heightById[id] || 0) / 2));
  const minY = Math.min(...Object.entries(positions).map(([id, pos]) => pos.y - (heightById[id] || 0) / 2));
  const meta = {
    width,
    height: maxY - minY + bottomPadding + padding,
    bounds: { minY, maxY },
  };

  return { positions, meta };
}

export function layoutPortAnchors(labels = [], pos, width, height, side) {
  const count = Math.max(1, labels.length || 0);
  const top = pos.y - height / 2 + 64;
  const bottom = pos.y + height / 2 - 30;
  const available = Math.max(32, bottom - top);
  const gap = available / (count + 1);
  const x = side === "left" ? pos.x - width / 2 : pos.x + width / 2;
  const anchors = [];
  for (let i = 0; i < count; i += 1) {
    anchors.push({
      x,
      y: top + gap * (i + 1),
      label: labels[i] || `${side === "left" ? "in" : "out"}${i + 1}`,
      side,
    });
  }
  return anchors;
}

export function measureNode(node, pos, mode, ctx, actionCatalog, lastRunResults) {
  const radius = 16;
  const width = NODE_WIDTH;
  const { inputs, outputs, toolLabel, runtimeInputs, runtimeOutputs } = describeNode(node, actionCatalog, lastRunResults);
  const contentLines = [];
  if (toolLabel) contentLines.push(`工具: ${toolLabel}`);

  if (runtimeInputs !== undefined) {
    contentLines.push(`运行入参: ${summarizeValue(runtimeInputs)}`);
  }
  if (runtimeOutputs !== undefined && Object.keys(runtimeOutputs).length > 0) {
    contentLines.push(`运行结果: ${summarizeValue(runtimeOutputs)}`);
  }

  const wrappedLines = contentLines.flatMap((line) => wrapText(ctx, line, width - 28, "15px Inter"));
  const baseHeight = 90;
  const dynamicHeight = wrappedLines.length * 18;
  const height = baseHeight + dynamicHeight;

  const portAnchors = {
    inputs: layoutPortAnchors(inputs, pos, width, height, "left"),
    outputs: layoutPortAnchors(outputs, pos, width, height, "right"),
  };

  return {
    radius,
    width,
    height,
    inputs,
    outputs,
    toolLabel,
    runtimeInputs,
    runtimeOutputs,
    wrappedLines,
    portAnchors,
  };
}

export function drawPort(ctx, anchor) {
  const bubbleColor = anchor.side === "left" ? "#22d3ee" : "#c084fc";
  ctx.save();
  ctx.fillStyle = "rgba(15, 23, 42, 0.7)";
  ctx.strokeStyle = "rgba(148, 163, 184, 0.7)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(anchor.x, anchor.y, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.beginPath();
  ctx.fillStyle = bubbleColor;
  ctx.arc(anchor.x, anchor.y, 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.font = "13px Inter";
  ctx.fillStyle = "#cbd5e1";
  ctx.textAlign = anchor.side === "left" ? "left" : "right";
  const offset = anchor.side === "left" ? 14 : -14;
  ctx.fillText(anchor.label, anchor.x + offset, anchor.y + 4);
  ctx.restore();
}

export function drawNode(node, pos, mode, geometry, ctx, lastRunResults) {
  const { radius, width, height, portAnchors, wrappedLines } = geometry;
  const runInfo = lastRunResults[node.id];
  const executed = runInfo !== undefined;
  const executionStyle = executed
    ? { fill: "rgba(74, 222, 128, 0.14)", stroke: "rgba(34, 197, 94, 0.9)", badgeBg: "rgba(74, 222, 128, 0.16)", badgeText: "#4ade80", label: "已执行" }
    : { fill: "rgba(255, 255, 255, 0.04)", stroke: "rgba(255, 255, 255, 0.12)", badgeBg: "rgba(148, 163, 184, 0.18)", badgeText: "#cbd5e1", label: "未执行" };

  const typeColors = {
    start: "#22d3ee",
    end: "#34d399",
    condition: "#fbbf24",
    loop: "#a5b4fc",
    switch: "#7dd3fc",
    action: "#c084fc",
  };
  const fill = typeColors[node.type] || "#94a3b8";
  ctx.save();
  ctx.fillStyle = executionStyle.fill;
  ctx.strokeStyle = executionStyle.stroke;
  ctx.lineWidth = 1.2;
  roundedRect(ctx, pos.x - width / 2, pos.y - height / 2, width, height, radius);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = executionStyle.badgeBg;
  ctx.strokeStyle = executionStyle.stroke;
  ctx.lineWidth = 1;
  const badgeWidth = 64;
  const badgeHeight = 22;
  roundedRect(ctx, pos.x + width / 2 - badgeWidth - 12, pos.y - height / 2 + 12, badgeWidth, badgeHeight, 10);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = executionStyle.badgeText;
  ctx.font = "13px Inter";
  ctx.textAlign = "center";
  ctx.fillText(executionStyle.label, pos.x + width / 2 - badgeWidth / 2 - 12, pos.y - height / 2 + 28);

  ctx.fillStyle = fill;
  ctx.font = "14px Inter";
  ctx.textAlign = "center";
  ctx.fillText(node.type.toUpperCase(), pos.x, pos.y - height / 2 + 20);

  ctx.fillStyle = "#e5e7eb";
  ctx.font = mode === "visual" ? "18px Inter" : "17px Inter";
  const label = node.display_name || node.action_id || node.id;
  ctx.fillText(label, pos.x, pos.y - height / 2 + 46);

  ctx.textAlign = "left";
  ctx.font = "15px Inter";
  let offsetY = pos.y - height / 2 + 72;
  wrappedLines.forEach((line) => {
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText(line, pos.x - width / 2 + 14, offsetY);
    offsetY += 18;
  });

  const annotatedInputs = portAnchors.inputs.map((anchor, index) => ({
    ...anchor,
    nodeId: node.id,
    portIndex: index,
    portKey: anchor.label || `in${index + 1}`,
  }));
  const annotatedOutputs = portAnchors.outputs.map((anchor, index) => ({
    ...anchor,
    nodeId: node.id,
    portIndex: index,
    portKey: anchor.label || `out${index + 1}`,
  }));

  annotatedInputs.forEach((anchor) => drawPort(ctx, anchor));
  annotatedOutputs.forEach((anchor) => drawPort(ctx, anchor));
  ctx.restore();

  return {
    id: node.id,
    x: pos.x - width / 2,
    y: pos.y - height / 2,
    width,
    height,
    inputs: annotatedInputs,
    outputs: annotatedOutputs,
    portAnchors,
  };
}

export function drawArrow(ctx, from, to, label) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const angle = Math.atan2(dy, dx);
  const startX = from.x + Math.cos(angle) * 14;
  const startY = from.y + Math.sin(angle) * 14;
  const endX = to.x - Math.cos(angle) * 14;
  const endY = to.y - Math.sin(angle) * 14;

  ctx.save();
  ctx.strokeStyle = "rgba(148, 163, 184, 0.8)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  ctx.beginPath();
  ctx.fillStyle = "rgba(148, 163, 184, 0.9)";
  const arrowSize = 8;
  ctx.translate(endX, endY);
  ctx.rotate(angle);
  ctx.moveTo(0, 0);
  ctx.lineTo(-arrowSize, arrowSize / 1.6);
  ctx.lineTo(-arrowSize, -arrowSize / 1.6);
  ctx.closePath();
  ctx.fill();

  if (label) {
    ctx.rotate(-angle);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px Inter";
    ctx.textAlign = "center";
    ctx.fillText(String(label), 0, -6);
  }
  ctx.restore();
}

export function pickPortAnchor(portList, usageMap, usageKey, fallback) {
  if (!portList || !portList.length) return fallback;
  const index = usageMap[usageKey] || 0;
  usageMap[usageKey] = index + 1;
  return portList[Math.min(index, portList.length - 1)];
}

export function findAnchorByLabel(anchors, label) {
  if (!anchors || !anchors.length || !label) return null;
  return anchors.find((anchor) => anchor.label === label || anchor.portKey === label) || null;
}

export function buildDisplayEdges(graph) {
  const normalized = normalizeWorkflow(graph || {});
  const baseEdges = (normalized.edges || []).map(normalizeEdge);
  const seen = new Set(baseEdges.map((e) => `${e.from_node}->${e.to_node}`));
  const displayEdges = [...baseEdges];

  const addEdge = (from, to, condition) => {
    if (!from || !to) return;
    const key = `${from}->${to}`;
    if (seen.has(key)) return;
    seen.add(key);
    displayEdges.push({ from_node: from, to_node: to, condition });
  };

  (normalized.nodes || []).forEach((node) => {
    if (!node) return;

    if (node.type === "condition") {
      addEdge(node.id, node.true_to_node, "true");
      addEdge(node.id, node.false_to_node, "false");
    }

    const refs = new Set();
    collectRefsFromValue(node.params, refs);
    refs.forEach((fromId) => addEdge(fromId, node.id));
  });

  return displayEdges;
}

export function roundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

export function applyViewTransform(ctx, viewState) {
  const dpr = window.devicePixelRatio || 1;
  const scale = viewState.scale || 1;
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, viewState.offset.x * dpr, viewState.offset.y * dpr);
}

export function estimateWorkflowHeight(workflow, widthHint = 1200, heightHint = 720, options = {}) {
  const nodes = workflow && Array.isArray(workflow.nodes) ? workflow.nodes : [];
  if (!nodes.length) return Math.max(420, heightHint);

  const { meta } = layoutNodes(workflow, {
    width: widthHint,
    heightHint,
    ctx: options.ctx,
    measureCtx: options.measureCtx,
    actionCatalog: options.actionCatalog,
    lastRunResults: options.lastRunResults,
  });
  if (meta && meta.height) return meta.height;

  const columns = Math.max(2, Math.ceil(Math.sqrt(nodes.length)));
  const rows = Math.ceil(nodes.length / columns);
  const tallest = Math.max(...nodes.map((n) => estimateNodeHeight(n, options.ctx, options.actionCatalog, options.lastRunResults)), 120);
  const verticalSpacing = 70;
  const padding = 140;
  return rows * (tallest + verticalSpacing) + padding;
}

export function measureHiddenHeight(element) {
  if (!element) return 0;
  if (!element.classList.contains("tab-content--hidden")) {
    return element.getBoundingClientRect().height;
  }

  const prevDisplay = element.style.display;
  const prevVisibility = element.style.visibility;
  const prevPosition = element.style.position;
  element.style.display = "flex";
  element.style.visibility = "hidden";
  element.style.position = "absolute";
  const height = element.getBoundingClientRect().height;
  element.style.display = prevDisplay;
  element.style.visibility = prevVisibility;
  element.style.position = prevPosition;
  return height;
}

export function resizeCanvas(canvas, panel, graph, state, options = {}) {
  const dpr = window.devicePixelRatio || 1;
  const panelWidth = panel ? panel.getBoundingClientRect().width : canvas.clientWidth;
  const targetWidth = Math.max(480, panelWidth - 16);

  const viewportBase = Math.max(420, window.innerHeight - 260);
  const contentHeight = estimateWorkflowHeight(graph, targetWidth, viewportBase, options);
  const targetHeight = Math.max(viewportBase, contentHeight + 40);

  canvas.style.width = "100%";
  canvas.style.height = `${Math.round(targetHeight)}px`;
  canvas.width = Math.floor(targetWidth * dpr);
  canvas.height = Math.floor(targetHeight * dpr);
  if (state && state.viewState) {
    state.viewState.offset.y = Math.max(state.viewState.offset.y, -(targetHeight / 2));
  }
}

export function findNodeByPoint(point, renderedNodes) {
  return renderedNodes.find(
    (node) => point.x >= node.x && point.x <= node.x + node.width && point.y >= node.y && point.y <= node.y + node.height,
  );
}

export function findPortHit(point, renderedNodes, preferredSide) {
  const radius = 10;
  for (const node of renderedNodes) {
    const candidates =
      preferredSide === "left"
        ? node.inputs || []
        : preferredSide === "right"
          ? node.outputs || []
          : [...(node.inputs || []), ...(node.outputs || [])];
    for (const anchor of candidates) {
      const dx = point.x - anchor.x;
      const dy = point.y - anchor.y;
      if (Math.sqrt(dx * dx + dy * dy) <= radius) {
        return anchor;
      }
    }
  }
  return null;
}
