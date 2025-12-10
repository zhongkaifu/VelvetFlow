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
let buildStep = 0;

function createDemoWorkflow() {
  return {
    name: "Employee onboarding",
    nodes: [
      { id: "entry", label: "入口", type: "entry" },
      { id: "collect_req", label: "收集需求", type: "action" },
      { id: "create_account", label: "账号申请", type: "action" },
      { id: "assign_device", label: "设备分配", type: "action" },
      { id: "approval", label: "主管审批", type: "condition", trueTo: "enable_access", falseTo: "exit" },
      { id: "enable_access", label: "开通权限", type: "action" },
      { id: "exit", label: "完成", type: "exit" }
    ],
    edges: [
      ["entry", "collect_req"],
      ["collect_req", "create_account"],
      ["create_account", "assign_device"],
      ["assign_device", "approval"],
      ["approval", "enable_access"],
      ["approval", "exit"],
      ["enable_access", "exit"]
    ]
  };
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

function layoutNodes(workflow) {
  const { nodes } = workflow;
  const positions = {};
  const columns = Math.ceil(Math.sqrt(nodes.length));
  const spacingX = workflowCanvas.width / (columns + 1);
  const spacingY = workflowCanvas.height / (Math.ceil(nodes.length / columns) + 1);

  nodes.forEach((node, index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    positions[node.id] = {
      x: spacingX * (col + 1),
      y: spacingY * (row + 1)
    };
  });
  return positions;
}

function drawNode(node, pos, mode) {
  const radius = 16;
  const width = 170;
  const height = 56;
  const typeColors = {
    entry: "#22d3ee",
    exit: "#34d399",
    condition: "#fbbf24",
    action: "#c084fc"
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
  ctx.fillText(node.type.toUpperCase(), pos.x, pos.y - 14);

  ctx.fillStyle = "#e5e7eb";
  ctx.font = mode === "visual" ? "16px Inter" : "15px Inter";
  ctx.fillText(node.label, pos.x, pos.y + 8);
  ctx.restore();
}

function drawArrow(from, to) {
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
  ctx.restore();
}

function render(mode = currentTab) {
  ctx.clearRect(0, 0, workflowCanvas.width, workflowCanvas.height);
  const positions = layoutNodes(currentWorkflow);

  currentWorkflow.edges.forEach(([fromId, toId]) => {
    drawArrow(positions[fromId], positions[toId]);
  });

  currentWorkflow.nodes.forEach((node) => {
    const pos = positions[node.id];
    drawNode(node, pos, mode);
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

function simulateBuild(requirement) {
  statusIndicator.textContent = "规划中";
  statusIndicator.style.background = "rgba(251, 191, 36, 0.2)";
  statusIndicator.style.color = "var(--warning)";
  buildLog.innerHTML = "";
  appendLog(`收到需求：${requirement}`);

  const steps = [
    "解析需求并抽取业务动作",
    "从动作库检索匹配步骤",
    "构建骨架并自动补充参数",
    "运行静态校验与自修复",
    "生成可视化 DAG"
  ];

  steps.forEach((step, index) => {
    setTimeout(() => {
      appendLog(step);
      if (index === steps.length - 1) {
        statusIndicator.textContent = "构建完成";
        statusIndicator.style.background = "rgba(52, 211, 153, 0.15)";
        statusIndicator.style.color = "var(--success)";
      }
    }, 650 * (index + 1));
  });
}

function generateWorkflow(requirement) {
  const normalized = requirement.toLowerCase();
  const needsApproval = /审批|approve|review/.test(normalized);
  const needsNotification = /通知|notify|邮件/.test(normalized);
  const needsLoop = /批量|循环|列表/.test(normalized);

  const nodes = [
    { id: "entry", label: "入口", type: "entry" },
    { id: "intent", label: "需求解析", type: "action" },
    { id: "search_tools", label: "动作检索", type: "action" }
  ];
  const edges = [
    ["entry", "intent"],
    ["intent", "search_tools"]
  ];

  let tail = "search_tools";

  if (needsLoop) {
    nodes.push({ id: "loop", label: "批量处理", type: "condition", trueTo: "fanout", falseTo: "compose" });
    nodes.push({ id: "fanout", label: "分发子任务", type: "action" });
    edges.push([tail, "loop"]);
    edges.push(["loop", "fanout"]);
    edges.push(["loop", "compose"]);
    tail = "fanout";
  }

  nodes.push({ id: "compose", label: "组合节点", type: "action" });
  edges.push([tail, "compose"]);
  tail = "compose";

  if (needsApproval) {
    nodes.push({ id: "approval", label: "审批节点", type: "condition", trueTo: "notify", falseTo: "exit" });
    edges.push([tail, "approval"]);
    edges.push(["approval", "exit"]);
    tail = "approval";
  }

  if (needsNotification) {
    nodes.push({ id: "notify", label: "结果通知", type: "action" });
    edges.push([tail, "notify"]);
    tail = "notify";
  }

  nodes.push({ id: "exit", label: "完成", type: "exit" });
  edges.push([tail, "exit"]);

  return { name: `Workflow for: ${requirement}`, nodes, edges };
}

function handleChatSubmit(event) {
  event.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  addChatMessage(text, "user");
  userInput.value = "";
  simulateBuild(text);

  setTimeout(() => {
    const workflow = generateWorkflow(text);
    currentWorkflow = workflow;
    updateEditor();
    render(currentTab);
    addChatMessage("已完成 DAG 规划，您可以在左侧查看可视化结果并继续编辑。", "agent");
  }, 2000);
}

function handleRunWorkflow() {
  buildStep += 1;
  const label = `运行 #${buildStep}: 已按拓扑顺序执行 ${currentWorkflow.nodes.length} 个节点`;
  appendLog(label);
  addChatMessage(`执行完成，输出已写入模拟存储。节点总数：${currentWorkflow.nodes.length}`, "agent");
}

function applyWorkflowFromEditor() {
  try {
    const parsed = JSON.parse(workflowEditor.value);
    if (!parsed.nodes || !parsed.edges) {
      throw new Error("workflow 需要包含 nodes 与 edges 数组");
    }
    currentWorkflow = parsed;
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
    "您可以在左侧 JSON 文本框中编辑节点或边，例如增加节点 {id: 'notify', label: '通知', type: 'action'} 并在 edges 中添加连接，点击“应用修改”即可刷新。",
    "agent"
  );
}

chatForm.addEventListener("submit", handleChatSubmit);
runWorkflowBtn.addEventListener("click", handleRunWorkflow);
applyWorkflowBtn.addEventListener("click", applyWorkflowFromEditor);
resetWorkflowBtn.addEventListener("click", resetWorkflow);

tabs.forEach((tab) => tab.addEventListener("click", handleTabClick));
editHelpBtn.addEventListener("click", showEditHelp);

updateEditor();
render();
appendLog("加载示例 workflow，可直接发送需求或修改 JSON。");
addChatMessage("你好，我是 VelvetFlow Agent，请描述你的业务需求。", "agent");
