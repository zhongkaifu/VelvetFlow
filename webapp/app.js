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

let currentTab = "dag";
let currentWorkflow = createDemoWorkflow();

function createDemoWorkflow() {
  return {
    workflow_name: "employee_onboarding",
    description: "示例: 入职账号与设备申请",
    nodes: [
      { id: "start", type: "start", display_name: "入口" },
      {
        id: "collect_req",
        type: "action",
        display_name: "收集需求",
        action_id: "hr.collect_onboarding_requirements",
      },
      {
        id: "create_account",
        type: "action",
        display_name: "账号申请",
        action_id: "it.provision_account",
      },
      {
        id: "assign_device",
        type: "action",
        display_name: "设备分配",
        action_id: "it.assign_device",
      },
      {
        id: "approval",
        type: "condition",
        display_name: "主管审批",
        params: { condition: { kind: "equals", left: "manager", right: "approved" } },
        true_to_node: "enable_access",
        false_to_node: "end",
      },
      {
        id: "enable_access",
        type: "action",
        display_name: "开通权限",
        action_id: "it.enable_access",
      },
      { id: "end", type: "end", display_name: "完成" },
    ],
    edges: [
      { from_node: "start", to_node: "collect_req" },
      { from_node: "collect_req", to_node: "create_account" },
      { from_node: "create_account", to_node: "assign_device" },
      { from_node: "assign_device", to_node: "approval" },
      { from_node: "approval", to_node: "enable_access", condition: "true" },
      { from_node: "approval", to_node: "end", condition: "false" },
      { from_node: "enable_access", to_node: "end" },
    ],
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
  const width = 190;
  const height = 64;
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
  ctx.lineWidth = 1;
  roundedRect(pos.x - width / 2, pos.y - height / 2, width, height, radius);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = fill;
  ctx.font = "12px Inter";
  ctx.textAlign = "center";
  ctx.fillText(node.type.toUpperCase(), pos.x, pos.y - 16);

  ctx.fillStyle = "#e5e7eb";
  ctx.font = mode === "visual" ? "17px Inter" : "15px Inter";
  const label = node.display_name || node.action_id || node.id;
  ctx.fillText(label, pos.x, pos.y + 8);
  ctx.restore();
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
  if (!currentWorkflow.nodes) return;
  const positions = layoutNodes(currentWorkflow);

  const edges = (currentWorkflow.edges || []).map(normalizeEdge);
  edges.forEach((edge) => {
    const from = positions[edge.from_node];
    const to = positions[edge.to_node];
    if (from && to) {
      drawArrow(from, to, edge.condition);
    }
  });

  currentWorkflow.nodes.forEach((node) => {
    const pos = positions[node.id];
    if (pos) {
      drawNode(node, pos, mode);
    }
  });

  if (mode === "visual") {
    drawWatermark();
  }
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
    const response = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ requirement, existing_workflow: currentWorkflow }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const hint =
        [404, 405, 501].includes(response.status)
          ? "后端 API 未启动或未通过 uvicorn 暴露，请运行 `uvicorn webapp.server:app --reload --port 8000` 后再试。"
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
          ? "后端 API 未启动或未通过 uvicorn 暴露，请运行 `uvicorn webapp.server:app --reload --port 8000` 后再试。"
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
  currentWorkflow = createDemoWorkflow();
  updateEditor();
  render(currentTab);
  appendLog("已恢复示例 workflow");
}

function handleTabClick(event) {
  const tab = event.target;
  currentTab = tab.dataset.tab;
  tabs.forEach((t) => {
    t.classList.toggle("tab--active", t === tab);
    t.setAttribute("aria-selected", t === tab);
  });
  render(currentTab);
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
editHelpBtn.addEventListener("click", showEditHelp);

updateEditor();
render();
appendLog("加载示例 workflow，可直接发送需求或修改 JSON。");
addChatMessage("你好，我是 VelvetFlow Agent，请描述你的业务需求。", "agent");
