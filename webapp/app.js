const workflowCanvas = document.getElementById("workflowCanvas");
const ctx = workflowCanvas.getContext("2d");
const chatLog = document.getElementById("chatLog");
const buildLog = document.getElementById("buildLog");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const statusIndicator = document.getElementById("statusIndicator");
const workflowEditor = document.getElementById("workflowEditor");
const runWorkflowBtn = document.getElementById("runWorkflow");
const applyWorkflowBtn = document.getElementById("applyWorkflow");
const resetWorkflowBtn = document.getElementById("resetWorkflow");
const editHelpBtn = document.getElementById("editHelp");
const tabs = document.querySelectorAll(".tab");
const tabContents = document.querySelectorAll(".tab-content");

let currentTab = "json";
let currentWorkflow = createEmptyWorkflow();
let renderedNodes = [];
let lastPositions = {};

function createEmptyWorkflow() {
  return {
    workflow_name: "",
    description: "",
    nodes: [],
    edges: [],
  };
}

function normalizeEdge(edge) {
  if (Array.isArray(edge) && edge.length >= 2) {
    return { from_node: edge[0], to_node: edge[1], condition: edge[2] };
  }
  return {
    from_node: edge.from_node || edge.from,
    to_node: edge.to_node || edge.to,
    condition: edge.condition,
  };
}

function normalizeWorkflow(workflow) {
  const edges = Array.isArray(workflow.edges) ? workflow.edges.map(normalizeEdge) : [];
  return { ...workflow, edges };
}

function addChatMessage(text, role = "agent") {
  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendLog(text) {
  const div = document.createElement("div");
  div.className = "log-entry";
  const now = new Date().toLocaleTimeString();
  div.textContent = `[${now}] ${text}`;
  buildLog.appendChild(div);
  buildLog.scrollTop = buildLog.scrollHeight;
}

function renderLogs(logs = []) {
  logs.forEach((line) => appendLog(line));
}

function setStatus(label, variant = "info") {
  const colors = {
    info: { bg: "rgba(59, 130, 246, 0.15)", color: "#60a5fa" },
    success: { bg: "rgba(52, 211, 153, 0.15)", color: "var(--success)" },
    warning: { bg: "rgba(251, 191, 36, 0.15)", color: "var(--warning)" },
    danger: { bg: "rgba(248, 113, 113, 0.15)", color: "var(--danger)" },
  };
  const theme = colors[variant] || colors.info;
  statusIndicator.textContent = label;
  statusIndicator.style.background = theme.bg;
  statusIndicator.style.color = theme.color;
}

function collectParamKeys(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(String);
  if (typeof value === "object") return Object.keys(value);
  return [String(value)];
}

function describeNode(node) {
  const inputs = collectParamKeys(node.inputs || node.input_params || node.params || node.args);
  const outputs = collectParamKeys(node.outputs || node.output_params || node.output);
  const toolLabel = node.type === "action" ? node.action_id || node.display_name || node.id : null;
  return {
    inputs,
    outputs,
    toolLabel,
  };
}

function wrapText(text, maxWidth, font = "13px Inter") {
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

function layoutNodes(workflow) {
  const { nodes } = workflow;
  const positions = {};
  const columns = Math.max(3, Math.ceil(Math.sqrt(nodes.length)));
  const spacingX = workflowCanvas.width / (columns + 1);
  const spacingY = workflowCanvas.height / (Math.ceil(nodes.length / columns) + 1);

  nodes.forEach((node, index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    positions[node.id] = {
      x: spacingX * (col + 1),
      y: spacingY * (row + 1),
    };
  });
  return positions;
}

function drawNode(node, pos, mode) {
  const radius = 16;
  const width = 240;
  const { inputs, outputs, toolLabel } = describeNode(node);
  const contentLines = [];
  if (toolLabel) contentLines.push(`工具: ${toolLabel}`);
  contentLines.push(`入参: ${inputs.length ? inputs.join(", ") : "-"}`);
  contentLines.push(`出参: ${outputs.length ? outputs.join(", ") : "-"}`);

  const wrappedLines = contentLines.flatMap((line) => wrapText(line, width - 28));
  const baseHeight = 72;
  const dynamicHeight = wrappedLines.length * 16;
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
  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1.2;
  roundedRect(pos.x - width / 2, pos.y - height / 2, width, height, radius);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = fill;
  ctx.font = "12px Inter";
  ctx.textAlign = "center";
  ctx.fillText(node.type.toUpperCase(), pos.x, pos.y - height / 2 + 18);

  ctx.fillStyle = "#e5e7eb";
  ctx.font = mode === "visual" ? "16px Inter" : "15px Inter";
  const label = node.display_name || node.action_id || node.id;
  ctx.fillText(label, pos.x, pos.y - height / 2 + 40);

  ctx.textAlign = "left";
  ctx.font = "13px Inter";
  let offsetY = pos.y - height / 2 + 60;
  wrappedLines.forEach((line) => {
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText(line, pos.x - width / 2 + 12, offsetY);
    offsetY += 16;
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

function render(mode = currentTab) {
  ctx.clearRect(0, 0, workflowCanvas.width, workflowCanvas.height);
  renderedNodes = [];
  if (!currentWorkflow.nodes) return;
  lastPositions = layoutNodes(currentWorkflow);

  const edges = (currentWorkflow.edges || []).map(normalizeEdge);
  edges.forEach((edge) => {
    const from = lastPositions[edge.from_node];
    const to = lastPositions[edge.to_node];
    if (from && to) {
      drawArrow(from, to, edge.condition);
    }
  });

  currentWorkflow.nodes.forEach((node) => {
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

function updateEditor() {
  workflowEditor.value = JSON.stringify(currentWorkflow, null, 2);
}

async function requestPlan(requirement) {
  setStatus("规划中", "warning");
  buildLog.innerHTML = "";
  appendLog(`收到需求：${requirement}`);
  try {
    const existingWorkflow =
      currentWorkflow && Array.isArray(currentWorkflow.nodes) && currentWorkflow.nodes.length > 0
        ? currentWorkflow
        : null;

    const response = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ requirement, existing_workflow: existingWorkflow }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const hint =
        [404, 405, 501].includes(response.status)
          ? "后端 API 未启动，请运行 `python webapp/server.py` 后再试。"
          : "";
      throw new Error(detail.detail || detail.message || hint || response.statusText);
    }

    const payload = await response.json();
    renderLogs(payload.logs);
    currentWorkflow = normalizeWorkflow(payload.workflow);
    updateEditor();
    render(currentTab);
    setStatus("构建完成", "success");
    addChatMessage("已完成 DAG 规划与校验，可在画布上查看并继续修改。", "agent");
  } catch (error) {
    setStatus("构建失败", "danger");
    appendLog(`规划失败: ${error.message}`);
    addChatMessage(`规划失败：${error.message}，请检查 OPENAI_API_KEY 是否已配置。`, "agent");
  }
}

async function requestRun() {
  setStatus("运行中", "warning");
  appendLog("开始执行当前 workflow ...");
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow: currentWorkflow }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const hint =
        [404, 405, 501].includes(response.status)
          ? "后端 API 未启动，请运行 `python webapp/server.py` 后再试。"
          : "";
      throw new Error(detail.detail || detail.message || hint || response.statusText);
    }

    const payload = await response.json();
    renderLogs(payload.logs);
    setStatus(payload.status === "completed" ? "运行完成" : "挂起等待回调", "success");
    appendLog(`运行结果: ${payload.status}`);
    addChatMessage(`执行状态：${payload.status}。结果：${JSON.stringify(payload.result, null, 2)}`, "agent");
  } catch (error) {
    setStatus("运行失败", "danger");
    appendLog(`运行失败: ${error.message}`);
    addChatMessage(`执行失败：${error.message}`, "agent");
  }
}

function handleChatSubmit(event) {
  event.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  addChatMessage(text, "user");
  userInput.value = "";
  requestPlan(text);
}

function applyWorkflowFromEditor() {
  try {
    const parsed = JSON.parse(workflowEditor.value);
    if (!parsed.nodes) {
      throw new Error("workflow 需要包含 nodes 数组");
    }
    currentWorkflow = normalizeWorkflow(parsed);
    render(currentTab);
    appendLog("已应用手动修改并刷新画布");
    addChatMessage("收到您的修改，Canvas 已同步更新。", "agent");
  } catch (error) {
    appendLog(`解析失败：${error.message}`);
  }
}

function resetWorkflow() {
  currentWorkflow = createEmptyWorkflow();
  updateEditor();
  render(currentTab);
  appendLog("已重置为空 workflow");
}

function handleTabClick(event) {
  const tab = event.target;
  currentTab = tab.dataset.tab;
  tabs.forEach((t) => {
    t.classList.toggle("tab--active", t === tab);
    t.setAttribute("aria-selected", t === tab);
  });
  tabContents.forEach((content) => {
    const isActive = content.dataset.view === currentTab;
    content.classList.toggle("tab-content--hidden", !isActive);
  });
  render(currentTab);
}

function canvasPointFromEvent(event) {
  const rect = workflowCanvas.getBoundingClientRect();
  const scaleX = workflowCanvas.width / rect.width;
  const scaleY = workflowCanvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function findNodeByPoint(point) {
  return renderedNodes.find(
    (node) =>
      point.x >= node.x &&
      point.x <= node.x + node.width &&
      point.y >= node.y &&
      point.y <= node.y + node.height,
  );
}

function handleCanvasClick(event) {
  if (currentTab !== "visual") return;
  const point = canvasPointFromEvent(event);
  const hit = findNodeByPoint(point);
  if (!hit) return;
  const target = currentWorkflow.nodes.find((n) => n.id === hit.id);
  if (!target) return;

  const edited = prompt("编辑节点 JSON", JSON.stringify(target, null, 2));
  if (!edited) return;
  try {
    const parsed = JSON.parse(edited);
    currentWorkflow.nodes = currentWorkflow.nodes.map((n) => (n.id === target.id ? parsed : n));
    updateEditor();
    render(currentTab);
    appendLog(`节点 ${target.id} 已更新`);
  } catch (error) {
    appendLog(`节点更新失败：${error.message}`);
  }
}

function showEditHelp() {
  addChatMessage(
    "您可以在左侧 JSON 文本框中编辑节点或边，例如增加节点 {id: 'notify', type: 'action', display_name: '通知'} 并添加 {from_node: 'enable_access', to_node: 'notify'}，点击“应用修改”刷新。",
    "agent",
  );
}

chatForm.addEventListener("submit", handleChatSubmit);
runWorkflowBtn.addEventListener("click", requestRun);
applyWorkflowBtn.addEventListener("click", applyWorkflowFromEditor);
resetWorkflowBtn.addEventListener("click", resetWorkflow);

tabs.forEach((tab) => tab.addEventListener("click", handleTabClick));
workflowCanvas.addEventListener("click", handleCanvasClick);
editHelpBtn.addEventListener("click", showEditHelp);

updateEditor();
render();
appendLog("当前 workflow 为空，请输入需求开始规划或自行编辑 JSON。");
addChatMessage("你好，我是 VelvetFlow Agent，请描述你的业务需求。", "agent");
