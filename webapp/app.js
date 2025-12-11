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
const tabBar = document.querySelector(".tab-bar");
let tabs = Array.from(document.querySelectorAll(".tab"));
let tabContents = Array.from(document.querySelectorAll(".tab-content"));
const visualTabContent = document.querySelector('[data-view="visual"]');
const jsonTabContent = document.querySelector('[data-view="json"]');
const canvasPanel = document.querySelector(".canvas-panel");
const canvasHost = document.getElementById("canvasHost");

const NODE_WIDTH = 320;

let currentTab = "json";
let currentWorkflow = createEmptyWorkflow();
let lastRunResults = {};
let renderedNodes = [];
let lastPositions = {};
let nodePositionsByTab = {};
let lastPositionsByTab = {};
let tabDirtyFlags = {};
let actionCatalog = {};

let isDragging = false;
let dragNodeId = null;
let dragOffset = { x: 0, y: 0 };
let dragBox = null;
let dragTabKey = null;

async function loadActionCatalog() {
  try {
    const resp = await fetch("/api/actions");
    if (!resp.ok) throw new Error(`加载业务工具失败: ${resp.status}`);
    const data = await resp.json();
    actionCatalog = (data || []).reduce((acc, action) => {
      if (action && action.action_id) acc[action.action_id] = action;
      return acc;
    }, {});
  } catch (error) {
    appendLog(`业务工具清单加载失败：${error.message}`);
  }
}

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

function refreshTabCollections() {
  tabs = Array.from(document.querySelectorAll(".tab"));
  tabContents = Array.from(document.querySelectorAll(".tab-content"));
}

function isCanvasTab(tabId) {
  return tabId === "visual" || tabId.startsWith("loop:");
}

function getTabContent(tabId) {
  return tabContents.find((content) => content.dataset.view === tabId);
}

function attachCanvasTo(tabId) {
  if (!isCanvasTab(tabId) || !canvasHost) return;
  const target = getTabContent(tabId);
  if (target && canvasHost.parentElement !== target) {
    target.appendChild(canvasHost);
  }
}

function markTabDirty(tabId, dirty = true) {
  if (!tabId || tabId === "json" || tabId === "visual") return;
  tabDirtyFlags[tabId] = dirty;
  const tab = tabs.find((t) => t.dataset.tab === tabId);
  if (tab) {
    tab.classList.toggle("tab--dirty", !!dirty);
  }
}

function clearPositionCaches(tabId) {
  if (tabId) {
    delete nodePositionsByTab[tabId];
    delete lastPositionsByTab[tabId];
    return;
  }
  nodePositionsByTab = {};
  lastPositionsByTab = {};
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

function renderLogs(logs = [], echoToChat = false) {
  logs.forEach((line) => {
    appendLog(line);
    if (echoToChat) {
      addChatMessage(`流程更新：${line}`, "agent");
    }
  });
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

function findLoopNode(loopId, workflow = currentWorkflow) {
  if (!workflow || !Array.isArray(workflow.nodes)) return null;
  for (const node of workflow.nodes) {
    if (node && node.id === loopId && node.type === "loop") {
      return { node, container: workflow };
    }
    const params = node && node.params;
    const body = params && params.body_subgraph;
    if (body && body.nodes) {
      const nested = findLoopNode(loopId, body);
      if (nested) return nested;
    }
  }
  return null;
}

function getTabContext(tabId = currentTab) {
  if (tabId && tabId.startsWith("loop:")) {
    const loopId = tabId.replace("loop:", "");
    const found = findLoopNode(loopId);
    const bodyGraph = (found && found.node && found.node.params && found.node.params.body_subgraph) || {
      nodes: [],
      edges: [],
    };
    return {
      kind: "loop",
      tabKey: tabId,
      loopId,
      graph: normalizeWorkflow(bodyGraph),
      saveGraph(updatedGraph) {
        const target = findLoopNode(loopId);
        if (!target) return;
        const params = { ...(target.node.params || {}), body_subgraph: normalizeWorkflow(updatedGraph) };
        target.node.params = params;
        updateEditor();
        markTabDirty(tabId, true);
      },
    };
  }

  return {
    kind: "root",
    tabKey: "root",
    graph: normalizeWorkflow(currentWorkflow),
    saveGraph(updatedGraph) {
      currentWorkflow = normalizeWorkflow(updatedGraph);
      updateEditor();
    },
  };
}

function ensureLoopTab(loopNode) {
  if (!loopNode || loopNode.type !== "loop") return;
  const tabId = `loop:${loopNode.id}`;
  const existing = tabs.find((t) => t.dataset.tab === tabId);
  if (existing) {
    switchToTab(tabId);
    return;
  }

  const tabButton = document.createElement("button");
  tabButton.className = "tab";
  tabButton.dataset.tab = tabId;
  tabButton.setAttribute("role", "tab");
  tabButton.setAttribute("aria-selected", "false");
  tabButton.innerHTML = `子图: ${loopNode.display_name || loopNode.id}<button class="tab__close" aria-label="关闭子图">×</button>`;

  const closeBtn = tabButton.querySelector(".tab__close");
  closeBtn.addEventListener("click", (evt) => {
    evt.stopPropagation();
    closeLoopTab(tabId);
  });
  tabButton.addEventListener("click", handleTabClick);

  const tabContent = document.createElement("div");
  tabContent.className = "tab-content tab-content--hidden";
  tabContent.dataset.view = tabId;
  tabContent.innerHTML = `<p class="muted tab-hint">正在查看循环 ${
    loopNode.display_name || loopNode.id
  } 的子图，双击节点可编辑；关闭前会提示确认已保存修改。</p>`;

  tabBar.appendChild(tabButton);
  canvasPanel.appendChild(tabContent);
  tabDirtyFlags[tabId] = false;
  refreshTabCollections();
  switchToTab(tabId);
}

function closeLoopTab(tabId) {
  if (!tabId || !tabId.startsWith("loop:")) return;
  const dirty = tabDirtyFlags[tabId];
  if (dirty) {
    const confirmClose = window.confirm("子图节点已修改，确认已保存并关闭吗？选择取消可继续编辑。");
    if (!confirmClose) return;
  }

  const button = tabs.find((t) => t.dataset.tab === tabId);
  const content = getTabContent(tabId);
  if (button) button.remove();
  if (content) content.remove();
  clearPositionCaches(tabId);
  delete tabDirtyFlags[tabId];
  refreshTabCollections();
  if (currentTab === tabId) {
    switchToTab("visual");
  }
}

function closeAllLoopTabs(force = false) {
  tabs
    .filter((t) => t.dataset.tab && t.dataset.tab.startsWith("loop:"))
    .forEach((t) => {
      if (force) {
        tabDirtyFlags[t.dataset.tab] = false;
      }
      closeLoopTab(t.dataset.tab);
    });
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
  const runInfo = lastRunResults[node.id];
  const runtimeInputs = runInfo && runInfo.params ? runInfo.params : undefined;
  const runtimeOutputs = runInfo
    ? Object.keys(runInfo)
        .filter((key) => key !== "params")
        .reduce((acc, key) => ({ ...acc, [key]: runInfo[key] }), {})
    : undefined;
  return {
    inputs,
    outputs,
    toolLabel,
    runtimeInputs,
    runtimeOutputs,
  };
}

function summarizeValue(value, limit = 160) {
  try {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    if (text.length > limit) return `${text.slice(0, limit)}…`;
    return text;
  } catch (error) {
    return String(value);
  }
}

function paramsSchemaFor(actionId) {
  const action = actionCatalog[actionId];
  return (action && (action.params_schema || action.arg_schema)) || {};
}

function outputSchemaFor(actionId, node) {
  if (node && node.out_params_schema) return node.out_params_schema;
  const action = actionCatalog[actionId];
  return (action && (action.output_schema || action.out_params_schema)) || {};
}

function extractParamDefs(schema) {
  const properties = schema && typeof schema === "object" ? schema.properties || {} : {};
  const required = Array.isArray(schema && schema.required) ? schema.required : [];
  return Object.entries(properties).map(([name, def]) => ({
    name,
    type: def && def.type ? def.type : "",
    description: def && def.description ? def.description : "",
    required: required.includes(name),
  }));
}

function extractOutputDefs(schema) {
  const properties = schema && typeof schema === "object" ? schema.properties || {} : {};
  const required = Array.isArray(schema && schema.required) ? schema.required : [];
  return Object.entries(properties).map(([name, def]) => ({
    name,
    type: def && def.type ? def.type : "",
    description: def && def.description ? def.description : "",
    required: required.includes(name),
  }));
}

function stringifyBinding(value) {
  if (value && typeof value === "object" && value.__from__) return String(value.__from__);
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch (error) {
    return String(value);
  }
}

function parseBinding(text) {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  if (trimmed.startsWith("result_of.")) {
    return { __from__: trimmed };
  }
  try {
    return JSON.parse(trimmed);
  } catch (error) {
    return trimmed;
  }
}

function collectLoopExportOptions(node) {
  if (!node || node.type !== "loop") return [];
  const params = node.params || {};
  const exportsSpec = params.exports || {};
  const base = `result_of.${node.id}.exports`;
  const options = [];

  const items = exportsSpec.items;
  if (items) {
    options.push(`${base}.items`);
    const fields = Array.isArray(items.fields) ? items.fields : [];
    fields.forEach((field) => {
      const key = String(field);
      options.push(`${base}.items.${key}`);
      options.push(`${base}.items[*].${key}`);
    });
    options.push(`${base}.items.length`);
  }

  const aggregates = Array.isArray(exportsSpec.aggregates) ? exportsSpec.aggregates : [];
  aggregates.forEach((agg) => {
    if (agg && agg.name) {
      options.push(`${base}.aggregates.${agg.name}`);
    }
  });

  return options;
}

function collectReferenceOptions(currentNodeId, graph = currentWorkflow) {
  const options = new Set();
  (graph.nodes || []).forEach((node) => {
    if (!node || node.id === currentNodeId) return;
    const outputs = extractOutputDefs(outputSchemaFor(node.action_id, node));
    outputs.forEach((field) => {
      options.add(`result_of.${node.id}.${field.name}`);
    });
    collectLoopExportOptions(node).forEach((opt) => options.add(opt));
  });
  return Array.from(options);
}

function collectRefsFromValue(value, refs) {
  if (!value) return;
  if (Array.isArray(value)) {
    value.forEach((item) => collectRefsFromValue(item, refs));
    return;
  }
  if (typeof value === "object") {
    if (value.__from__ && typeof value.__from__ === "string" && value.__from__.startsWith("result_of.")) {
      const parts = value.__from__.split(".");
      if (parts.length >= 2) refs.add(parts[1]);
    }
    Object.values(value).forEach((v) => collectRefsFromValue(v, refs));
    return;
  }
  if (typeof value === "string" && value.startsWith("result_of.")) {
    const parts = value.split(".");
    if (parts.length >= 2) refs.add(parts[1]);
  }
}

function rebuildEdgesFromBindings(workflow) {
  const edges = new Set((workflow.edges || []).map((e) => `${e.from_node || e.from}->${e.to_node || e.to}`));
  const newEdges = [];
  (workflow.nodes || []).forEach((node) => {
    if (!node || !node.params) return;
    const refs = new Set();
    Object.values(node.params).forEach((value) => collectRefsFromValue(value, refs));
    refs.forEach((fromId) => {
      const key = `${fromId}->${node.id}`;
      if (!edges.has(key)) {
        newEdges.push({ from_node: fromId, to_node: node.id });
      }
    });
  });
  workflow.edges = normalizeWorkflow({ edges: [...(workflow.edges || []), ...newEdges] }).edges;
}

function openNodeDialog(node, context = getTabContext()) {
  if (!node) return;
  if (node.type !== "action") {
    addChatMessage("当前仅支持编辑 action 节点的入参/出参。", "agent");
    return;
  }

  if (!Object.keys(actionCatalog).length) {
    loadActionCatalog();
  }

  let draftParams = { ...(node.params || {}) };

  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";

  const dialog = document.createElement("div");
  dialog.className = "modal";

  const title = document.createElement("div");
  title.className = "modal__title";
  title.textContent = `编辑节点：${node.display_name || node.action_id || node.id}`;
  dialog.appendChild(title);

  const actionRow = document.createElement("div");
  actionRow.className = "modal__row";
  const actionLabel = document.createElement("label");
  actionLabel.textContent = "业务工具 (action_id)";
  const actionSelect = document.createElement("select");
  actionSelect.className = "modal__select";
  const actions = Object.values(actionCatalog).length
    ? Object.values(actionCatalog)
    : [{ action_id: node.action_id, name: node.display_name }];
  actions.forEach((action) => {
    const option = document.createElement("option");
    option.value = action.action_id;
    option.textContent = `${action.action_id}${action.name ? ` · ${action.name}` : ""}`;
    actionSelect.appendChild(option);
  });
  actionSelect.value = node.action_id || actionSelect.value;
  actionRow.appendChild(actionLabel);
  actionRow.appendChild(actionSelect);
  dialog.appendChild(actionRow);

  const inputsHeader = document.createElement("div");
  inputsHeader.className = "modal__subtitle";
  inputsHeader.textContent = "输入参数 (引用 result_of.* 或常量)";
  dialog.appendChild(inputsHeader);

  const inputsContainer = document.createElement("div");
  inputsContainer.className = "modal__grid";
  dialog.appendChild(inputsContainer);

  const outputsHeader = document.createElement("div");
  outputsHeader.className = "modal__subtitle";
  outputsHeader.textContent = "输出参数";
  dialog.appendChild(outputsHeader);

  const outputsContainer = document.createElement("div");
  outputsContainer.className = "modal__list";
  dialog.appendChild(outputsContainer);

  function refreshIO(actionId) {
    inputsContainer.innerHTML = "";
    outputsContainer.innerHTML = "";

    const paramDefs = extractParamDefs(paramsSchemaFor(actionId));
    const refOptions = collectReferenceOptions(node.id, context.graph || currentWorkflow);
    paramDefs.forEach((param) => {
      const row = document.createElement("label");
      row.className = "modal__field";
      row.innerHTML = `<div class="modal__field-label">${param.name}${param.required ? " *" : ""}<span class="modal__hint">${
        param.description || param.type || ""
      }</span></div>`;
      const input = document.createElement("input");
      input.type = "text";
      input.className = "modal__input";
      input.value = stringifyBinding(draftParams ? draftParams[param.name] : undefined);
      input.dataset.paramName = param.name;
      const datalistId = `refs-${node.id}-${param.name}`;
      if (refOptions.length) {
        input.setAttribute("list", datalistId);
        const datalist = document.createElement("datalist");
        datalist.id = datalistId;
        refOptions.forEach((opt) => {
          const option = document.createElement("option");
          option.value = opt;
          datalist.appendChild(option);
        });
        row.appendChild(datalist);
      }
      row.appendChild(input);
      inputsContainer.appendChild(row);
    });

    const outputDefs = extractOutputDefs(outputSchemaFor(actionId, node));
    outputDefs.forEach((out) => {
      const item = document.createElement("div");
      item.className = "modal__output";
      item.innerHTML = `<div class="modal__output-name">${out.name}${out.required ? " *" : ""}</div><div class="modal__hint">${
        out.description || out.type || ""
      }</div>`;
      outputsContainer.appendChild(item);
    });
  }

  const actionsRow = document.createElement("div");
  actionsRow.className = "modal__actions";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "button button--ghost";
  cancelBtn.textContent = "取消";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "button button--primary";
  saveBtn.textContent = "保存并刷新";
  actionsRow.appendChild(cancelBtn);
  actionsRow.appendChild(saveBtn);
  dialog.appendChild(actionsRow);

  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  cancelBtn.addEventListener("click", close);
  overlay.addEventListener("click", (evt) => {
    if (evt.target === overlay) close();
  });

  actionSelect.addEventListener("change", () => {
    draftParams = {};
    refreshIO(actionSelect.value);
  });

  saveBtn.addEventListener("click", () => {
    const updatedParams = {};
    const fields = inputsContainer.querySelectorAll("input[data-param-name]");
    fields.forEach((input) => {
      const name = input.dataset.paramName;
      const value = parseBinding(input.value);
      if (value !== undefined) {
        updatedParams[name] = value;
      }
    });

    const updatedGraph = {
      ...(context.graph || currentWorkflow),
      nodes: (context.graph.nodes || []).map((n) => (n.id === node.id ? { ...n, action_id: actionSelect.value, params: updatedParams } : n)),
    };
    rebuildEdgesFromBindings(updatedGraph);
    context.saveGraph(updatedGraph);
    render(currentTab);
    appendLog(`节点 ${node.id} 已更新并重新绘制`);
    close();
  });

  refreshIO(actionSelect.value);
}

function wrapText(text, maxWidth, font = "15px Inter") {
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

  const jsonHeight = measureHiddenHeight(jsonTabContent);
  const activeContent = getTabContent(currentTab) || visualTabContent;
  const visualHeight = Math.max(measureHiddenHeight(visualTabContent), measureHiddenHeight(activeContent));
  const viewportBase = Math.max(420, window.innerHeight - 260);
  const contentHeight = estimateWorkflowHeight(graph || currentWorkflow, targetWidth, viewportBase);
  const targetHeight = Math.max(contentHeight, jsonHeight, visualHeight, viewportBase);

  workflowCanvas.style.width = "100%";
  workflowCanvas.style.height = `${Math.round(targetHeight)}px`;
  workflowCanvas.width = Math.floor(targetWidth * dpr);
  workflowCanvas.height = Math.floor(targetHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function layoutNodes(workflow, options = {}) {
  const { nodes = [], edges = [] } = workflow;
  if (!nodes.length) return { positions: {}, meta: { height: 0, width: 0 } };

  const width = options.width || (workflowCanvas ? workflowCanvas.width : 1200);
  const heightHint = options.heightHint || (workflowCanvas ? workflowCanvas.height : 720);

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
    width: workflowCanvas ? workflowCanvas.width : undefined,
    heightHint: workflowCanvas ? workflowCanvas.height : undefined,
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

function drawNode(node, pos, mode) {
  const radius = 16;
  const width = NODE_WIDTH;
  const { inputs, outputs, toolLabel, runtimeInputs, runtimeOutputs } = describeNode(node);
  const runInfo = lastRunResults[node.id];
  const executed = runInfo !== undefined;
  const executionStyle = executed
    ? { fill: "rgba(74, 222, 128, 0.14)", stroke: "rgba(34, 197, 94, 0.9)", badgeBg: "rgba(74, 222, 128, 0.16)", badgeText: "#4ade80", label: "已执行" }
    : { fill: "rgba(255, 255, 255, 0.04)", stroke: "rgba(255, 255, 255, 0.12)", badgeBg: "rgba(148, 163, 184, 0.18)", badgeText: "#cbd5e1", label: "未执行" };
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
  ctx.clearRect(0, 0, workflowCanvas.width, workflowCanvas.height);
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
  workflowEditor.style.height = "auto";
  const newHeight = Math.max(workflowEditor.scrollHeight, 200);
  workflowEditor.style.height = `${newHeight}px`;
}

function updateEditor() {
  workflowEditor.value = JSON.stringify(currentWorkflow, null, 2);
  autoSizeEditor();
}

async function streamEvents(url, options, onEvent) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const hint =
      [404, 405, 501].includes(response.status)
        ? "后端 API 未启动，请运行 `python webapp/server.py` 后再试。"
        : "";
    throw new Error(detail.detail || detail.message || hint || response.statusText);
  }

  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      const content = part.trim();
      if (!content) continue;
      const payload = content.startsWith("data:") ? content.slice(5).trim() : content;
      try {
        onEvent(JSON.parse(payload));
      } catch (error) {
        console.warn("无法解析流式消息", payload, error);
      }
    }
  }
}

async function requestPlan(requirement) {
  setStatus("规划中", "warning");
  buildLog.innerHTML = "";
  appendLog(`收到需求：${requirement}`);
  addChatMessage(`已收到需求：“${requirement}”，开始规划/校验/修复。`, "agent");
  try {
    const existingWorkflow =
      currentWorkflow && Array.isArray(currentWorkflow.nodes) && currentWorkflow.nodes.length > 0
        ? currentWorkflow
        : null;

    if (existingWorkflow) {
      addChatMessage("将基于当前流程进行更新与自修复，保持节点和连线同步。", "agent");
    } else {
      addChatMessage("将从零开始构建全新 workflow，并进行自动校验与修复。", "agent");
    }
    addChatMessage("正在调用 VelvetFlow Planner，请稍候，处理中间状态……", "agent");

    let streamError = null;
    let finalWorkflow = null;

    await streamEvents(
      "/api/plan/stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requirement, existing_workflow: existingWorkflow }),
      },
      (event) => {
        if (!event || !event.type) return;
        if (event.type === "log" && event.message) {
          appendLog(event.message);
          addChatMessage(`流程更新：${event.message}`, "agent");
        } else if (event.type === "result" && event.workflow) {
          finalWorkflow = normalizeWorkflow(event.workflow);
        } else if (event.type === "error") {
          streamError = event.message || "未知错误";
        }
      },
    );

    if (streamError) {
      throw new Error(streamError);
    }
    if (!finalWorkflow) {
      throw new Error("未收到构建结果");
    }

    lastRunResults = {};
    clearPositionCaches();
    closeAllLoopTabs(true);
    currentWorkflow = finalWorkflow;
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
  addChatMessage("开始执行 workflow，实时同步运行日志。", "agent");
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
    renderLogs(payload.logs, true);
    setStatus(payload.status === "completed" ? "运行完成" : "挂起等待回调", "success");
    lastRunResults = payload.result || {};
    render(currentTab);
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
    lastRunResults = {};
    clearPositionCaches();
    closeAllLoopTabs(true);
    render(currentTab);
    appendLog("已应用手动修改并刷新画布");
    addChatMessage("收到您的修改，Canvas 已同步更新。", "agent");
  } catch (error) {
    appendLog(`解析失败：${error.message}`);
  }
}

function resetWorkflow() {
  currentWorkflow = createEmptyWorkflow();
  clearPositionCaches();
  closeAllLoopTabs(true);
  lastRunResults = {};
  updateEditor();
  render(currentTab);
  appendLog("已重置为空 workflow");
}

function switchToTab(tabId) {
  const tab = tabs.find((t) => t.dataset.tab === tabId);
  if (!tab) return;
  currentTab = tabId;
  attachCanvasTo(tabId);
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

function handleTabClick(event) {
  const tab = event.currentTarget || event.target;
  if (!tab || !tab.dataset.tab) return;
  switchToTab(tab.dataset.tab);
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

function handleCanvasDoubleClick(event) {
  if (!isCanvasTab(currentTab) || isDragging) return;
  const context = getTabContext(currentTab);
  const point = canvasPointFromEvent(event);
  const hit = findNodeByPoint(point);
  if (!hit) return;
  const target = (context.graph.nodes || []).find((n) => n.id === hit.id);
  if (!target) return;

  if (target.type === "loop") {
    ensureLoopTab(target);
    return;
  }

  openNodeDialog(target, context);
}

function handleCanvasMouseDown(event) {
  if (!isCanvasTab(currentTab)) return;
  const point = canvasPointFromEvent(event);
  const hit = findNodeByPoint(point);
  if (!hit) return;
  isDragging = true;
  dragNodeId = hit.id;
  dragBox = hit;
  dragOffset = { x: point.x - hit.x, y: point.y - hit.y };
  const ctxInfo = getTabContext(currentTab);
  dragTabKey = ctxInfo.tabKey || currentTab;
  workflowCanvas.style.cursor = "grabbing";
}

function handleCanvasMouseMove(event) {
  if (!isDragging || !isCanvasTab(currentTab) || !dragNodeId || !dragBox) return;
  const point = canvasPointFromEvent(event);
  const topLeftX = point.x - dragOffset.x;
  const topLeftY = point.y - dragOffset.y;
  const ctxInfo = getTabContext(currentTab);
  const store = positionStoreFor(dragTabKey || ctxInfo.tabKey || currentTab);
  store[dragNodeId] = {
    x: topLeftX + dragBox.width / 2,
    y: topLeftY + dragBox.height / 2,
  };
  render(currentTab);
}

function stopDragging() {
  if (!isDragging) return;
  isDragging = false;
  dragNodeId = null;
  dragBox = null;
  dragTabKey = null;
  workflowCanvas.style.cursor = "grab";
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
workflowEditor.addEventListener("input", autoSizeEditor);

tabs.forEach((tab) => tab.addEventListener("click", handleTabClick));
workflowCanvas.addEventListener("dblclick", handleCanvasDoubleClick);
workflowCanvas.addEventListener("mousedown", handleCanvasMouseDown);
workflowCanvas.addEventListener("mousemove", handleCanvasMouseMove);
workflowCanvas.addEventListener("mouseup", stopDragging);
workflowCanvas.addEventListener("mouseleave", stopDragging);
editHelpBtn.addEventListener("click", showEditHelp);

const editorResizeObserver = new ResizeObserver(() => render(currentTab));
editorResizeObserver.observe(workflowEditor);
window.addEventListener("resize", () => render(currentTab));

loadActionCatalog();
updateEditor();
render();
appendLog("当前 workflow 为空，请输入需求开始规划或自行编辑 JSON。");
addChatMessage("你好，我是 VelvetFlow Agent，请描述你的业务需求。", "agent");
