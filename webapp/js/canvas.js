function estimateNodeHeight(node) {
  const { inputs, outputs, toolLabel, runtimeInputs, runtimeOutputs } = describeNode(node);
  const contentLines = [];
  if (toolLabel) contentLines.push(`工具: ${toolLabel}`);
  contentLines.push(`入参: ${inputs.length ? inputs.join(", ") : "-"}`);
  contentLines.push(`出参: ${outputs.length ? outputs.join(", ") : "-"}`);

  if (runtimeInputs !== undefined) {
    contentLines.push(`运行入参: ${summarizeValue(runtimeInputs)}`);
  }
  if (runtimeOutputs !== undefined && Object.keys(runtimeOutputs).length > 0) {
    contentLines.push(`运行结果: ${summarizeValue(runtimeOutputs)}`);
  }

  const wrappedLines = contentLines.flatMap((line) => wrapText(line, NODE_WIDTH - 28, "15px Inter"));
  const baseHeight = 90;
  const dynamicHeight = wrappedLines.length * 18;
  return baseHeight + dynamicHeight;
}

function estimateWorkflowHeight(workflow, widthHint = 1200, heightHint = 720) {
  const nodes = workflow && Array.isArray(workflow.nodes) ? workflow.nodes : [];
  if (!nodes.length) return Math.max(420, heightHint);

  const { meta } = layoutNodes(workflow, { width: widthHint, heightHint });
  if (meta && meta.height) return meta.height;

  const columns = Math.max(2, Math.ceil(Math.sqrt(nodes.length)));
  const rows = Math.ceil(nodes.length / columns);
  const tallest = Math.max(...nodes.map(estimateNodeHeight), 120);
  const verticalSpacing = 70;
  const padding = 140;
  return rows * (tallest + verticalSpacing) + padding;
}

function measureHiddenHeight(element) {
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

function resizeCanvas(graph) {
  const dpr = window.devicePixelRatio || 1;
  const panelWidth = canvasPanel ? canvasPanel.getBoundingClientRect().width : workflowCanvas.clientWidth;
  const targetWidth = Math.max(480, panelWidth - 16);

  const viewportBase = Math.max(420, window.innerHeight - 260);
  const contentHeight = estimateWorkflowHeight(graph || currentWorkflow, targetWidth, viewportBase);
  const targetHeight = Math.max(viewportBase, contentHeight + 40);

  workflowCanvas.style.width = "100%";
  workflowCanvas.style.height = `${Math.round(targetHeight)}px`;
  workflowCanvas.width = Math.floor(targetWidth * dpr);
  workflowCanvas.height = Math.floor(targetHeight * dpr);
}

function applyViewTransform() {
  const dpr = window.devicePixelRatio || 1;
  const scale = viewState.scale || 1;
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, viewState.offset.x * dpr, viewState.offset.y * dpr);
}

function layoutNodes(workflow, options = {}) {
  const { nodes = [], edges = [] } = workflow;
  if (!nodes.length) return { positions: {}, meta: { height: 0, width: 0 } };

  const width = options.width || (workflowCanvas ? workflowCanvas.clientWidth : 1200);
  const heightHint = options.heightHint || (workflowCanvas ? workflowCanvas.clientHeight : 720);

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

  // Handle any remaining nodes (cycles or disconnected) by assigning them to the last level.
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
    const row = layers[layerIndex];
    const ranked = row
      .map((id, rawIndex) => {
        const parents = incoming[id] || [];
        const parentRanks = parents.map((p) => layerOrderIndex[p]).filter((v) => v !== undefined);
        const barycenter = parentRanks.length ? parentRanks.reduce((a, b) => a + b, 0) / parentRanks.length : rawIndex;
        return { id, score: barycenter };
      })
      .sort((a, b) => a.score - b.score);

    sortedLayers[layerIndex] = ranked.map((item, order) => {
      layerOrderIndex[item.id] = order;
      return item.id;
    });

    // Encourage continuity by re-evaluating based on outgoing neighbors after parents are placed.
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
    acc[node.id] = estimateNodeHeight(node);
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

function positionStoreFor(tabId) {
  if (!nodePositionsByTab[tabId]) {
    nodePositionsByTab[tabId] = {};
  }
  return nodePositionsByTab[tabId];
}

function syncPositions(workflow, tabId) {
  const { positions: auto } = layoutNodes(workflow, {
    width: workflowCanvas ? workflowCanvas.clientWidth : undefined,
    heightHint: workflowCanvas ? workflowCanvas.clientHeight : undefined,
  });
  const store = positionStoreFor(tabId);
  const nextPositions = {};
  (workflow.nodes || []).forEach((node) => {
    nextPositions[node.id] = store[node.id] || auto[node.id];
  });
  nodePositionsByTab[tabId] = nextPositions;
  lastPositionsByTab[tabId] = nextPositions;
  return nextPositions;
}

function deriveExecutionStyle(runInfo) {
  if (!runInfo) {
    return {
      label: "未执行",
      fill: "rgba(255, 255, 255, 0.04)",
      stroke: "rgba(255, 255, 255, 0.12)",
      badgeBg: "rgba(148, 163, 184, 0.18)",
      badgeText: "#cbd5e1",
      hasError: false,
      executed: false,
    };
  }

  const status = typeof runInfo.status === "string" ? runInfo.status.toLowerCase() : "";
  const errorStatuses = new Set([
    "tool_error",
    "tool_not_registered",
    "no_action_impl",
    "no_tool_mapping",
    "forbidden",
    "role_not_allowed",
  ]);
  const statusIndicatesError = status
    ? status.includes("error") || status.includes("fail") || errorStatuses.has(status)
    : false;
  const hasError = Boolean(runInfo.error) || statusIndicatesError;

  if (hasError) {
    return {
      label: "执行错误",
      fill: "rgba(248, 113, 113, 0.12)",
      stroke: "rgba(248, 113, 113, 0.9)",
      badgeBg: "rgba(248, 113, 113, 0.2)",
      badgeText: "#fecdd3",
      hasError: true,
      executed: true,
    };
  }

  return {
    label: "已执行",
    fill: "rgba(74, 222, 128, 0.14)",
    stroke: "rgba(34, 197, 94, 0.9)",
    badgeBg: "rgba(74, 222, 128, 0.16)",
    badgeText: "#4ade80",
    hasError: false,
    executed: true,
  };
}

function drawNode(node, pos, mode) {
  const radius = 16;
  const width = NODE_WIDTH;
  const { inputs, outputs, toolLabel, runtimeInputs, runtimeOutputs } = describeNode(node);
  const runInfo = lastRunResults[node.id];
  const executionStyle = deriveExecutionStyle(runInfo);
  const contentLines = [];
  if (toolLabel) contentLines.push(`工具: ${toolLabel}`);
  contentLines.push(`入参: ${inputs.length ? inputs.join(", ") : "-"}`);
  contentLines.push(`出参: ${outputs.length ? outputs.join(", ") : "-"}`);

  if (runtimeInputs !== undefined) {
    contentLines.push(`运行入参: ${summarizeValue(runtimeInputs)}`);
  }
  if (runtimeOutputs !== undefined && Object.keys(runtimeOutputs).length > 0) {
    contentLines.push(`运行结果: ${summarizeValue(runtimeOutputs)}`);
  }

  const wrappedLines = contentLines.flatMap((line) => wrapText(line, width - 28, "15px Inter"));
  const baseHeight = 90;
  const dynamicHeight = wrappedLines.length * 18;
  const height = baseHeight + dynamicHeight;

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
  roundedRect(pos.x - width / 2, pos.y - height / 2, width, height, radius);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = executionStyle.badgeBg;
  ctx.strokeStyle = executionStyle.stroke;
  ctx.lineWidth = 1;
  const badgeWidth = 64;
  const badgeHeight = 22;
  roundedRect(pos.x + width / 2 - badgeWidth - 12, pos.y - height / 2 + 12, badgeWidth, badgeHeight, 10);
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
  ctx.restore();

  return {
    id: node.id,
    x: pos.x - width / 2,
    y: pos.y - height / 2,
    width,
    height,
  };
}

function drawArrow(from, to, label) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const angle = Math.atan2(dy, dx);
  const startX = from.x + Math.cos(angle) * 80;
  const startY = from.y + Math.sin(angle) * 32;
  const endX = to.x - Math.cos(angle) * 80;
  const endY = to.y - Math.sin(angle) * 32;

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

function buildDisplayEdges(graph) {
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

    const dependsOn = Array.isArray(node.depends_on) ? node.depends_on : [];
    dependsOn.forEach((fromId) => addEdge(fromId, node.id));

    const refs = new Set();
    collectRefsFromValue(node.params, refs);
    refs.forEach((fromId) => addEdge(fromId, node.id));
  });

  return displayEdges;
}

function render(mode = currentTab) {
  const context = getTabContext(mode);
  const graph = context.graph || currentWorkflow;
  const tabKey = context.tabKey || mode;

  if (isCanvasTab(mode)) {
    attachCanvasTo(mode);
  }

  resizeCanvas(graph);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, workflowCanvas.width, workflowCanvas.height);
  applyViewTransform();
  renderedNodes = [];
  if (!graph.nodes) return;
  lastPositions = syncPositions(graph, tabKey);

  const edges = buildDisplayEdges(graph);
  edges.forEach((edge) => {
    const from = lastPositions[edge.from_node];
    const to = lastPositions[edge.to_node];
    if (from && to) {
      drawArrow(from, to, edge.condition);
    }
  });

  graph.nodes.forEach((node) => {
    const pos = lastPositions[node.id];
    if (pos) {
      const box = drawNode(node, pos, mode);
      if (box) renderedNodes.push(box);
    }
  });

  drawWatermark();
}

function drawWatermark() {
  ctx.save();
  ctx.fillStyle = "rgba(124, 58, 237, 0.12)";
  ctx.font = "48px Inter";
  ctx.fillText("VelvetFlow 可视化", 32, workflowCanvas.height - 32);
  ctx.restore();
}

function roundedRect(x, y, width, height, radius) {
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

function autoSizeEditor() {
  if (!workflowEditor) return;
  const isHidden =
    workflowEditor.offsetParent === null || workflowEditor.clientWidth === 0 || workflowEditor.clientHeight === 0;
  if (isHidden) {
    autoSizeEditor._pending = true;
    return;
  }
  autoSizeEditor._pending = false;
  const computed = getComputedStyle(workflowEditor);
  const parentWidth = workflowEditor.parentElement?.clientWidth || window.innerWidth;
  workflowEditor.style.width = "auto";
  const paddingX =
    parseFloat(computed.paddingLeft || 0) +
    parseFloat(computed.paddingRight || 0) +
    parseFloat(computed.borderLeftWidth || 0) +
    parseFloat(computed.borderRightWidth || 0);

  const measureCtx =
    autoSizeEditor._measureCtx || (autoSizeEditor._measureCtx = document.createElement("canvas").getContext("2d"));
  let contentWidth = workflowEditor.scrollWidth;

  if (measureCtx) {
    measureCtx.font = `${computed.fontWeight} ${computed.fontSize} ${computed.fontFamily}`;
    const maxLineWidth = workflowEditor.value
      .split("\n")
      .map((line) => measureCtx.measureText(line || " ").width)
      .reduce((max, width) => Math.max(max, width), 0);
    contentWidth = maxLineWidth;
  }

  const desiredWidth = Math.max(contentWidth + paddingX, parentWidth * 0.6);
  const clampedWidth = Math.min(desiredWidth, parentWidth);
  workflowEditor.style.width = `${clampedWidth}px`;
  workflowEditor.style.maxWidth = "100%";

  const minHeight = parseFloat(computed.minHeight) || 200;
  const maxHeight = Math.max(window.innerHeight * 0.7, minHeight);

  workflowEditor.style.height = "auto";
  const newHeight = Math.min(Math.max(workflowEditor.scrollHeight, minHeight), maxHeight);
  workflowEditor.style.height = `${newHeight}px`;
}

function updateEditor() {
  workflowEditor.value = JSON.stringify(currentWorkflow, null, 2);
  autoSizeEditor();
}
