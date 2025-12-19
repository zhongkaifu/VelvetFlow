const workflowCanvas = document.getElementById("workflowCanvas");
const ctx = workflowCanvas.getContext("2d");
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const statusIndicator = document.getElementById("statusIndicator");
const workflowEditor = document.getElementById("workflowEditor");
const runWorkflowBtn = document.getElementById("runWorkflow");
const applyWorkflowBtn = document.getElementById("applyWorkflow");
const resetWorkflowBtn = document.getElementById("resetWorkflow");
const loadWorkflowBtn = document.getElementById("loadWorkflow");
const loadWorkflowInput = document.getElementById("loadWorkflowFile");
const saveWorkflowBtn = document.getElementById("saveWorkflow");
const editHelpBtn = document.getElementById("editHelp");
const tabBar = document.querySelector(".tab-bar");
let tabs = Array.from(document.querySelectorAll(".tab"));
let tabContents = Array.from(document.querySelectorAll(".tab-content"));
const visualTabContent = document.querySelector('[data-view="visual"]');
const jsonTabContent = document.querySelector('[data-view="json"]');
const canvasPanel = document.querySelector(".canvas-panel");
const canvasHost = document.getElementById("canvasHost");
const addNodeMenu = document.getElementById("addNodeMenu");
const addNodeTypeSelect = document.getElementById("addNodeType");
const addNodeNameInput = document.getElementById("addNodeName");
const addNodeActionSelect = document.getElementById("addNodeActionSelect");
const addNodeActionRow = document.getElementById("addNodeActionRow");
const confirmAddNodeBtn = document.getElementById("confirmAddNode");
const cancelAddNodeBtn = document.getElementById("cancelAddNode");
const closeAddNodeMenuBtn = document.getElementById("closeAddNodeMenu");
const zoomInBtn = document.getElementById("zoomIn");
const zoomOutBtn = document.getElementById("zoomOut");
const resetViewBtn = document.getElementById("resetView");
const zoomValue = document.getElementById("zoomValue");

const NODE_WIDTH = 320;
const MIN_SCALE = 0.5;
const MAX_SCALE = 2.5;
const SCALE_STEP = 0.1;

let currentTab = "visual";
let currentWorkflow = createEmptyWorkflow();
let lastRunResults = {};
let renderedNodes = [];
let lastPositions = {};
let nodePositionsByTab = {};
let lastPositionsByTab = {};
let tabDirtyFlags = {};
let actionCatalog = {};
let selectedNodeType = "action";

let isDragging = false;
let dragNodeId = null;
let dragOffset = { x: 0, y: 0 };
let dragBox = null;
let dragTabKey = null;
let isPanning = false;
let panStart = { x: 0, y: 0 };
let panOrigin = { x: 0, y: 0 };
let viewState = { scale: 1, offset: { x: 0, y: 0 } };

async function loadActionCatalog() {
  try {
    const resp = await fetch("/api/actions");
    if (!resp.ok) throw new Error(`加载业务工具失败: ${resp.status}`);
    const data = await resp.json();
    actionCatalog = (data || []).reduce((acc, action) => {
      if (action && action.action_id) acc[action.action_id] = action;
      return acc;
    }, {});
    populateActionSelect();
  } catch (error) {
    appendLog(`业务工具清单加载失败：${error.message}`);
  }
}

function populateActionSelect(target = addNodeActionSelect) {
  if (!target) return;
  const actions = Object.values(actionCatalog);
  target.innerHTML = "";

  if (!actions.length) {
    const placeholder = document.createElement("option");
    placeholder.value = "custom_action";
    placeholder.textContent = "custom_action";
    target.appendChild(placeholder);
    return;
  }

  actions.forEach((action) => {
    const option = document.createElement("option");
    option.value = action.action_id;
    option.textContent = `${action.action_id}${action.name ? ` · ${action.name}` : ""}`;
    target.appendChild(option);
  });
}

function createEmptyWorkflow() {
  return {
    workflow_name: "",
    description: "",
    nodes: [],
    edges: [],
  };
}

function generateNodeId(prefix, graph = currentWorkflow) {
  const existing = new Set((graph.nodes || []).map((n) => n.id));
  const base = prefix || "node";
  let counter = 1;
  let candidate = `${base}_${counter}`;
  while (existing.has(candidate)) {
    counter += 1;
    candidate = `${base}_${counter}`;
  }
  return candidate;
}

function createDefaultNode(type, graph = currentWorkflow, options = {}) {
  const id = generateNodeId(type, graph);
  const display = options.display_name || `${type} ${id}`;

  if (type === "action") {
    const defaultAction = options.action_id || Object.keys(actionCatalog)[0] || "custom_action";
    return {
      id,
      type: "action",
      action_id: defaultAction,
      display_name: display,
      params: {},
    };
  }

  if (type === "condition") {
    return {
      id,
      type: "condition",
      display_name: display,
      params: {},
      true_to_node: "",
      false_to_node: "",
    };
  }

  if (type === "loop") {
    return {
      id,
      type: "loop",
      display_name: display,
      params: {
        body_subgraph: { nodes: [], edges: [] },
        exports: {},
      },
    };
  }

  return { id, type, display_name: display };
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
  if (!target) return;
  const slot = target.querySelector('[data-canvas-slot]');
  const desiredParent = slot || target;
  if (canvasHost.parentElement !== desiredParent) {
    desiredParent.appendChild(canvasHost);
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
  scrollChatToBottom();
}

function scrollChatToBottom() {
  requestAnimationFrame(() => {
    chatLog.scrollTop = chatLog.scrollHeight;
  });
}

function appendLog(text) {
  const div = document.createElement("div");
  div.className = "log-entry";
  const now = new Date().toLocaleTimeString();
  div.textContent = `[${now}] ${text}`;
  chatLog.appendChild(div);
  scrollChatToBottom();
}

function logWorkflowSnapshot(workflow = currentWorkflow, reason = "最新") {
  if (!workflow) return;
  try {
    const payload = JSON.stringify(normalizeWorkflow(workflow), null, 2);
    appendLog(`${reason} workflow DAG：\n${payload}`);
  } catch (error) {
    appendLog(`workflow 序列化失败：${error.message}`);
  }
}

function renderLogs(logs = [], echoToChat = false) {
  logs.forEach((line) => {
    appendLog(line);
    if (echoToChat) {
      addChatMessage(`流程更新：${line}`, "agent");
    }
  });
}

function setRunWorkflowEnabled(enabled, message = "") {
  if (!runWorkflowBtn) return;
  const disabled = !enabled;
  runWorkflowBtn.disabled = disabled;
  runWorkflowBtn.setAttribute("aria-disabled", String(disabled));
  if (message) {
    runWorkflowBtn.title = message;
  } else if (disabled) {
    runWorkflowBtn.title = "workflow 构建成功后可运行";
  } else {
    runWorkflowBtn.removeAttribute("title");
  }
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

setRunWorkflowEnabled(false);

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

function refreshLoopTabLabel(tabId, loopNode) {
  const tab = tabs.find((t) => t.dataset.tab === tabId);
  if (!tab || !loopNode) return;
  const display = loopNode.display_name || loopNode.id;
  tab.innerHTML = `子图: ${display}<button class="tab__close" aria-label="关闭子图">×</button>`;
  const closeBtn = tab.querySelector(".tab__close");
  if (closeBtn) {
    closeBtn.addEventListener("click", (evt) => {
      evt.stopPropagation();
      closeLoopTab(tabId);
    });
  }
}

function renderLoopInspector(tabId) {
  if (!tabId || !tabId.startsWith("loop:")) return;
  const content = getTabContent(tabId);
  const inspector = content && content.querySelector("[data-loop-inspector]");
  if (!inspector) return;

  const found = findLoopNode(tabId.replace("loop:", ""));
  if (!found || !found.node) {
    inspector.innerHTML = "<p class=\"muted\">未找到对应的循环节点，可能已被删除。</p>";
    return;
  }

  const loopNode = found.node;
  const context = getTabContext(tabId);
  const { body_subgraph, ...restParams } = loopNode.params || {};
  const loopKind = restParams.loop_kind || restParams.kind || "";
  const iterPath = restParams.iter || restParams.source || "";
  const conditionPath = restParams.condition || "";
  const itemAlias = restParams.item_alias || "";
  const exports = restParams.exports || {};
  const exportItems = Array.isArray(exports.items)
    ? exports.items
    : exports.items
      ? [exports.items]
      : [];
  const exportAggregates = Array.isArray(exports.aggregates)
    ? exports.aggregates
    : exports.aggregates
      ? [exports.aggregates]
      : [];

  inspector.innerHTML = "";

  const title = document.createElement("div");
  title.className = "loop-inspector__title";
  title.textContent = `循环节点：${loopNode.display_name || loopNode.id}`;
  inspector.appendChild(title);

  const helper = document.createElement("p");
  helper.className = "muted";
  helper.textContent = "上方为循环子图，您可以在此编辑节点；下方可调整循环节点的名称、循环来源与导出字段。";
  inspector.appendChild(helper);

  const idField = document.createElement("label");
  idField.className = "loop-inspector__field";
  idField.innerHTML = '<span>节点 ID</span>';
  const idInput = document.createElement("input");
  idInput.type = "text";
  idInput.className = "modal__input";
  idInput.value = loopNode.id || "";
  idInput.readOnly = true;
  idInput.setAttribute("aria-readonly", "true");
  idField.appendChild(idInput);
  inspector.appendChild(idField);

  const nameField = document.createElement("label");
  nameField.className = "loop-inspector__field";
  nameField.innerHTML = '<span>显示名称</span>';
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "modal__input";
  nameInput.value = loopNode.display_name || "";
  nameField.appendChild(nameInput);
  inspector.appendChild(nameField);

  const paramsSection = document.createElement("div");
  paramsSection.className = "loop-inspector__section";

  const paramsGrid = document.createElement("div");
  paramsGrid.className = "loop-inspector__grid";

  const createField = (label, value, placeholder, name) => {
    const field = document.createElement("label");
    field.className = "loop-inspector__field";
    field.innerHTML = `<span>${label}</span>`;
    const input = document.createElement("input");
    input.type = "text";
    input.name = name;
    input.placeholder = placeholder || "";
    input.className = "modal__input";
    input.value = stringifyBinding(value) || "";
    field.appendChild(input);
    return field;
  };

  paramsGrid.appendChild(createField("循环类型 (loop_kind)", loopKind, "例如：for_each", "loop_kind"));
  paramsGrid.appendChild(createField("循环数据来源 (iter/source)", iterPath, "如 result_of.fetch.items", "iter"));
  paramsGrid.appendChild(createField("条件字段 (condition，可选)", conditionPath, "可选：用于 while/条件循环", "condition"));
  paramsGrid.appendChild(createField("循环项别名 (item_alias)", itemAlias, "如 item/user", "item_alias"));

  paramsSection.appendChild(paramsGrid);
  inspector.appendChild(paramsSection);

  const exportsSection = document.createElement("div");
  exportsSection.className = "loop-inspector__section";

  const exportsTitle = document.createElement("div");
  exportsTitle.className = "loop-inspector__title";
  exportsTitle.textContent = "循环输出 (exports)";
  exportsSection.appendChild(exportsTitle);

  const itemsHeader = document.createElement("div");
  itemsHeader.className = "loop-inspector__subtitle";
  itemsHeader.textContent = "逐项输出 (items)";
  exportsSection.appendChild(itemsHeader);

  const itemsList = document.createElement("div");
  itemsList.className = "loop-exports";

  const createItemRow = (item = {}) => {
    const row = document.createElement("div");
    row.className = "loop-export-card";
    row.dataset.exportItem = "true";

    const header = document.createElement("div");
    header.className = "loop-export-card__header";
    const titleLabel = document.createElement("span");
    titleLabel.textContent = "采集条目";
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "icon-button";
    removeBtn.textContent = "×";
    removeBtn.title = "移除该条目";
    removeBtn.addEventListener("click", () => row.remove());
    header.appendChild(titleLabel);
    header.appendChild(removeBtn);
    row.appendChild(header);

    const fields = document.createElement("div");
    fields.className = "loop-export-card__fields";

    const fromField = createField("来自节点 (from_node)", item.from_node || "", "body_subgraph 中的节点 id", "from_node");
    const fieldsField = createField(
      "导出字段 (fields)",
      Array.isArray(item.fields) ? item.fields.join(", ") : item.fields || "",
      "以逗号分隔的字段名",
      "fields",
    );
    const modeField = createField("收集方式 (mode)", item.mode || "collect", "collect/first/last 等", "mode");

    fieldsField.querySelector("input").dataset.commaList = "true";
    modeField.querySelector("input").dataset.modeField = "true";

    fields.appendChild(fromField);
    fields.appendChild(fieldsField);
    fields.appendChild(modeField);
    row.appendChild(fields);

    return row;
  };

  exportItems.forEach((item) => itemsList.appendChild(createItemRow(item)));

  const addItemBtn = document.createElement("button");
  addItemBtn.type = "button";
  addItemBtn.className = "button button--ghost";
  addItemBtn.textContent = "添加 items 输出";
  addItemBtn.addEventListener("click", () => itemsList.appendChild(createItemRow()));

  exportsSection.appendChild(itemsList);
  exportsSection.appendChild(addItemBtn);

  const aggregatesHeader = document.createElement("div");
  aggregatesHeader.className = "loop-inspector__subtitle";
  aggregatesHeader.textContent = "聚合输出 (aggregates)";
  exportsSection.appendChild(aggregatesHeader);

  const aggregatesList = document.createElement("div");
  aggregatesList.className = "loop-exports";

  const createAggregateRow = (agg = {}) => {
    const row = document.createElement("div");
    row.className = "loop-export-card";
    row.dataset.exportAggregate = "true";

    const header = document.createElement("div");
    header.className = "loop-export-card__header";
    const titleLabel = document.createElement("span");
    titleLabel.textContent = "聚合规则";
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "icon-button";
    removeBtn.textContent = "×";
    removeBtn.title = "移除聚合";
    removeBtn.addEventListener("click", () => row.remove());
    header.appendChild(titleLabel);
    header.appendChild(removeBtn);
    row.appendChild(header);

    const fields = document.createElement("div");
    fields.className = "loop-export-card__fields";
    fields.appendChild(createField("来自节点 (from_node)", agg.from_node || "", "body_subgraph 中的节点 id", "from_node"));
    fields.appendChild(createField("字段/表达式 (field)", agg.field || "", "如 total/score", "field"));
    fields.appendChild(createField("聚合操作 (op)", agg.op || "", "sum/avg/count 等", "op"));
    fields.appendChild(createField("输出别名 (alias)", agg.alias || "", "聚合结果名称", "alias"));
    row.appendChild(fields);

    return row;
  };

  exportAggregates.forEach((agg) => aggregatesList.appendChild(createAggregateRow(agg)));

  const addAggregateBtn = document.createElement("button");
  addAggregateBtn.type = "button";
  addAggregateBtn.className = "button button--ghost";
  addAggregateBtn.textContent = "添加聚合输出";
  addAggregateBtn.addEventListener("click", () => aggregatesList.appendChild(createAggregateRow()));

  exportsSection.appendChild(aggregatesList);
  exportsSection.appendChild(addAggregateBtn);

  const hint = document.createElement("div");
  hint.className = "modal__hint";
  hint.textContent = "提示：body_subgraph 会与上方子图保持同步；exports.items/aggregates 将用于循环外部的数据引用。";
  exportsSection.appendChild(hint);

  inspector.appendChild(exportsSection);

  const actions = document.createElement("div");
  actions.className = "modal__actions";
  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "button button--danger";
  deleteBtn.textContent = "删除循环节点";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "button button--primary";
  saveBtn.textContent = "保存循环参数";
  actions.appendChild(deleteBtn);
  actions.appendChild(saveBtn);
  inspector.appendChild(actions);

  deleteBtn.addEventListener("click", () => {
    const confirmed = window.confirm("确认删除该循环节点及其子图？此操作不可恢复。");
    if (!confirmed) return;
    deleteLoopNode(loopNode.id);
  });

  saveBtn.addEventListener("click", () => {
    try {
      const formData = new FormData();
      Array.from(paramsGrid.querySelectorAll("input")).forEach((input) => {
        formData.set(input.name, input.value.trim());
      });

      const updatedParams = { ...restParams };

      const loopKindVal = formData.get("loop_kind");
      const iterVal = formData.get("iter");
      const conditionVal = formData.get("condition");
      const aliasVal = formData.get("item_alias");

      if (loopKindVal) updatedParams.loop_kind = loopKindVal;
      else delete updatedParams.loop_kind;

      if (iterVal) updatedParams.iter = iterVal;
      else delete updatedParams.iter;

      if (conditionVal) updatedParams.condition = conditionVal;
      else delete updatedParams.condition;

      if (aliasVal) updatedParams.item_alias = aliasVal;
      else delete updatedParams.item_alias;

      const exportsPayload = {};

      const itemsPayload = Array.from(itemsList.querySelectorAll("[data-export-item]"))
        .map((row) => {
          const rowData = {};
          row.querySelectorAll("input").forEach((input) => {
            const val = input.value.trim();
            if (!val) return;
            if (input.dataset.commaList) {
              rowData[input.name] = val
                .split(",")
                .map((v) => v.trim())
                .filter(Boolean);
            } else {
              rowData[input.name] = val;
            }
          });
          return Object.keys(rowData).length ? rowData : null;
        })
        .filter(Boolean);

      if (itemsPayload.length) {
        exportsPayload.items = itemsPayload;
      }

      const aggregatesPayload = Array.from(aggregatesList.querySelectorAll("[data-export-aggregate]"))
        .map((row) => {
          const rowData = {};
          row.querySelectorAll("input").forEach((input) => {
            const val = input.value.trim();
            if (val) rowData[input.name] = val;
          });
          return Object.keys(rowData).length ? rowData : null;
        })
        .filter(Boolean);

      if (aggregatesPayload.length) {
        exportsPayload.aggregates = aggregatesPayload;
      }

      if (Object.keys(exportsPayload).length) {
        updatedParams.exports = exportsPayload;
      } else {
        delete updatedParams.exports;
      }

      if (iterVal) {
        updatedParams.source = iterVal;
      } else if (restParams.source) {
        updatedParams.source = restParams.source;
      } else {
        delete updatedParams.source;
      }

      updatedParams.body_subgraph = normalizeWorkflow(context.graph || body_subgraph || { nodes: [], edges: [] });

      found.node.display_name = nameInput.value.trim() || loopNode.id;
      found.node.params = updatedParams;
      refreshLoopTabLabel(tabId, found.node);
      updateEditor();
      markTabDirty(tabId, false);
      render(tabId);
      appendLog(`已保存循环 ${found.node.id} 的参数并同步到 workflow JSON`);
    } catch (error) {
      window.alert(`保存失败：${error.message}`);
    }
  });
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
  tabButton.addEventListener("click", handleTabClick);

  const tabContent = document.createElement("div");
  tabContent.className = "tab-content tab-content--hidden";
  tabContent.dataset.view = tabId;
  const loopLayout = document.createElement("div");
  loopLayout.className = "loop-tab";

  const loopHeader = document.createElement("div");
  loopHeader.className = "loop-tab__header";
  loopHeader.innerHTML = `<div><h3>循环子图：${loopNode.display_name || loopNode.id}</h3><p class="muted">上半部分为循环子图，双击子节点可打开对应编辑弹窗。</p></div>`;

  const loopBody = document.createElement("div");
  loopBody.className = "loop-tab__body";

  const canvasSlot = document.createElement("div");
  canvasSlot.className = "loop-tab__canvas";
  canvasSlot.dataset.canvasSlot = "true";
  canvasSlot.innerHTML =
    '<p class="muted tab-hint">提示：在子图画布空白处右键添加节点，双击节点可编辑或继续打开子循环。</p>';

  const inspector = document.createElement("div");
  inspector.className = "loop-tab__inspector";
  inspector.dataset.loopInspector = "true";

  loopBody.appendChild(canvasSlot);
  loopBody.appendChild(inspector);
  loopLayout.appendChild(loopHeader);
  loopLayout.appendChild(loopBody);
  tabContent.appendChild(loopLayout);

  tabBar.appendChild(tabButton);
  canvasPanel.appendChild(tabContent);
  tabDirtyFlags[tabId] = false;
  refreshTabCollections();
  refreshLoopTabLabel(tabId, loopNode);
  renderLoopInspector(tabId);
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

function setSelectedNodeType(type) {
  if (!type) return;
  selectedNodeType = type;
  if (addNodeTypeSelect && addNodeTypeSelect.value !== type) {
    addNodeTypeSelect.value = type;
  }
  if (addNodeActionRow) {
    addNodeActionRow.style.display = type === "action" ? "flex" : "none";
  }
}

function addNodeToCurrentGraph(options = {}) {
  if (!isCanvasTab(currentTab)) {
    switchToTab("visual");
  }
  const context = getTabContext(currentTab);
  const graph = context.graph || currentWorkflow;
  const type = options.type || (addNodeTypeSelect ? addNodeTypeSelect.value : selectedNodeType);
  setSelectedNodeType(type);
  const displayName = options.displayName || (addNodeNameInput ? addNodeNameInput.value.trim() : "");
  const actionId = options.actionId || (addNodeActionSelect ? addNodeActionSelect.value : undefined);
  const newNode = createDefaultNode(type, graph, {
    display_name: displayName,
    action_id: actionId,
  });

  const updatedGraph = { ...graph, nodes: [...(graph.nodes || []), newNode] };
  context.saveGraph(updatedGraph);
  clearPositionCaches(context.tabKey || currentTab);
  render(currentTab);
  if (addNodeNameInput) addNodeNameInput.value = "";
  appendLog(`已添加 ${type} 节点：${newNode.id}`);
  logWorkflowSnapshot(currentWorkflow, "新增节点后");
}

function positionAddMenu(event) {
  if (!addNodeMenu) return;
  const wrapperRect = (canvasHost || workflowCanvas).getBoundingClientRect();
  const menuWidth = addNodeMenu.offsetWidth || 320;
  const menuHeight = addNodeMenu.offsetHeight || 260;
  const relativeX = event.clientX - wrapperRect.left;
  const relativeY = event.clientY - wrapperRect.top;
  const left = Math.min(Math.max(12, relativeX), wrapperRect.width - menuWidth - 12);
  const top = Math.min(Math.max(12, relativeY), wrapperRect.height - menuHeight - 12);
  addNodeMenu.style.left = `${left}px`;
  addNodeMenu.style.top = `${top}px`;
}

function showAddNodeMenu(event) {
  if (!addNodeMenu) return;
  event.preventDefault();
  if (!isCanvasTab(currentTab)) {
    switchToTab("visual");
  }
  if (!Object.keys(actionCatalog).length) {
    loadActionCatalog();
  }
  populateActionSelect(addNodeActionSelect);
  addNodeMenu.classList.remove("hidden");
  setSelectedNodeType(addNodeTypeSelect ? addNodeTypeSelect.value : selectedNodeType);
  positionAddMenu(event);
}

function hideAddNodeMenu() {
  if (addNodeMenu) {
    addNodeMenu.classList.add("hidden");
  }
}

function removeNodeFromGraph(nodeId, context = getTabContext(currentTab)) {
  if (!nodeId) return;
  const graph = context.graph || currentWorkflow;
  if (!graph.nodes || !graph.nodes.length) return;

  const targetNode = (graph.nodes || []).find((n) => n && n.id === nodeId);
  const updatedNodes = graph.nodes.filter((n) => n && n.id !== nodeId);
  if (updatedNodes.length === graph.nodes.length) return;

  const updatedEdges = (graph.edges || []).filter((edge) => {
    const from = edge.from_node || edge.from;
    const to = edge.to_node || edge.to;
    return from !== nodeId && to !== nodeId;
  });

  const updatedGraph = { ...graph, nodes: updatedNodes, edges: updatedEdges };
  context.saveGraph(updatedGraph);
  clearPositionCaches(context.tabKey || currentTab);
  if (context.kind === "root" && targetNode && targetNode.type === "loop") {
    closeLoopTab(`loop:${nodeId}`);
  }
  render(currentTab);
  appendLog(`已删除节点：${nodeId}`);
  logWorkflowSnapshot(currentWorkflow, "删除节点后");
}

function deleteLoopNode(loopId) {
  if (!loopId) return;
  const found = findLoopNode(loopId);
  if (!found || !found.container) return;
  const container = found.container;
  container.nodes = (container.nodes || []).filter((n) => n && n.id !== loopId);
  container.edges = (container.edges || []).filter((edge) => {
    const from = edge.from_node || edge.from;
    const to = edge.to_node || edge.to;
    return from !== loopId && to !== loopId;
  });

  clearPositionCaches();
  tabDirtyFlags[`loop:${loopId}`] = false;
  closeLoopTab(`loop:${loopId}`);
  updateEditor();
  render(currentTab);
  appendLog(`已删除循环节点：${loopId}`);
  logWorkflowSnapshot(currentWorkflow, "删除循环节点后");
}
