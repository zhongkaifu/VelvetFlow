import {
  createEmptyWorkflow,
  generateNodeId,
  createDefaultNode,
  normalizeEdge,
  normalizeWorkflow,
  describeNode,
  summarizeValue,
  paramsSchemaFor,
  outputSchemaFor,
  extractParamDefs,
  extractOutputDefs,
  parseBinding,
  stringifyBinding,
  collectLoopExportOptions,
  collectRefsFromValue,
} from "./workflow-utils.js";
import {
  resizeCanvas,
  applyViewTransform,
  layoutNodes,
  measureNode,
  drawNode,
  drawArrow,
  pickPortAnchor,
  buildDisplayEdges,
  findNodeByPoint,
  findPortHit,
  findAnchorByLabel,
} from "./canvas-renderer.js";

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
let linkingEdge = null;

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

function collectReferenceOptions(currentNodeId, graph = currentWorkflow) {
  const options = new Set();
  (graph.nodes || []).forEach((node) => {
    if (!node || node.id === currentNodeId) return;
    const outputs = extractOutputDefs(outputSchemaFor(node.action_id, node, actionCatalog));
    outputs.forEach((field) => {
      options.add(`result_of.${node.id}.${field.name}`);
    });
    collectLoopExportOptions(node).forEach((opt) => options.add(opt));
  });
  return Array.from(options);
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
  }, actionCatalog);

  const updatedGraph = { ...graph, nodes: [...(graph.nodes || []), newNode] };
  context.saveGraph(updatedGraph);
  clearPositionCaches(context.tabKey || currentTab);
  render(currentTab);
  if (addNodeNameInput) addNodeNameInput.value = "";
  appendLog(`已添加 ${type} 节点：${newNode.id}`);
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
}

function openNodeDialog(node, context = getTabContext()) {
  if (!node) return;
  if (node.type === "condition") {
    openConditionDialog(node, context);
    return;
  }
  if (node.type !== "action") {
    addChatMessage("当前仅支持编辑 action 或 condition 节点的参数。", "agent");
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

  const idRow = document.createElement("div");
  idRow.className = "modal__row";
  const idLabel = document.createElement("label");
  idLabel.textContent = "节点 ID";
  const idInput = document.createElement("input");
  idInput.className = "modal__input";
  idInput.value = node.id || "";
  idInput.readOnly = true;
  idInput.setAttribute("aria-readonly", "true");
  idRow.appendChild(idLabel);
  idRow.appendChild(idInput);
  dialog.appendChild(idRow);

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

    const paramDefs = extractParamDefs(paramsSchemaFor(actionId, actionCatalog));
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

    const outputDefs = extractOutputDefs(outputSchemaFor(actionId, node, actionCatalog));
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
  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "button button--danger";
  deleteBtn.textContent = "删除节点";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "button button--ghost";
  cancelBtn.textContent = "取消";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "button button--primary";
  saveBtn.textContent = "保存并刷新";
  actionsRow.appendChild(deleteBtn);
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

  deleteBtn.addEventListener("click", () => {
    const confirmed = window.confirm(`确认删除节点 ${node.display_name || node.id} 吗？`);
    if (!confirmed) return;
    removeNodeFromGraph(node.id, context);
    close();
  });

  refreshIO(actionSelect.value);
}

function openConditionDialog(node, context = getTabContext()) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";

  const dialog = document.createElement("div");
  dialog.className = "modal";

  const title = document.createElement("div");
  title.className = "modal__title";
  title.textContent = `编辑条件节点：${node.display_name || node.id}`;
  dialog.appendChild(title);

  const idRow = document.createElement("div");
  idRow.className = "modal__row";
  const idLabel = document.createElement("label");
  idLabel.textContent = "节点 ID";
  const idInput = document.createElement("input");
  idInput.className = "modal__input";
  idInput.value = node.id || "";
  idInput.readOnly = true;
  idInput.setAttribute("aria-readonly", "true");
  idRow.appendChild(idLabel);
  idRow.appendChild(idInput);
  dialog.appendChild(idRow);

  const nameRow = document.createElement("div");
  nameRow.className = "modal__row";
  const nameLabel = document.createElement("label");
  nameLabel.textContent = "显示名称";
  const nameInput = document.createElement("input");
  nameInput.className = "modal__input";
  nameInput.value = node.display_name || "";
  nameRow.appendChild(nameLabel);
  nameRow.appendChild(nameInput);
  dialog.appendChild(nameRow);

  const branchHeader = document.createElement("div");
  branchHeader.className = "modal__subtitle";
  branchHeader.textContent = "条件分支指向 (true/false_to_node)";
  dialog.appendChild(branchHeader);

  const branchGrid = document.createElement("div");
  branchGrid.className = "modal__grid";
  const branchInputs = {};

  const nodeOptions = (context.graph.nodes || [])
    .map((n) => n && n.id)
    .filter((id) => id && id !== node.id);
  const branchDatalistId = `branch-targets-${node.id}`;

  const createBranchField = (label, value, name) => {
    const field = document.createElement("label");
    field.className = "modal__field";
    field.innerHTML = `<div class="modal__field-label">${label}<span class="modal__hint">可填节点 id 或留空表示结束</span></div>`;
    const input = document.createElement("input");
    input.className = "modal__input";
    input.name = name;
    input.value = value || "";
    input.setAttribute("list", branchDatalistId);
    branchInputs[name] = input;
    field.appendChild(input);
    return field;
  };

  branchGrid.appendChild(createBranchField("true_to_node", node.true_to_node, "true_to_node"));
  branchGrid.appendChild(createBranchField("false_to_node", node.false_to_node, "false_to_node"));
  dialog.appendChild(branchGrid);

  if (nodeOptions.length) {
    const datalist = document.createElement("datalist");
    datalist.id = branchDatalistId;
    nodeOptions.forEach((id) => {
      const option = document.createElement("option");
      option.value = id;
      datalist.appendChild(option);
    });
    dialog.appendChild(datalist);
  }

  const paramsHeader = document.createElement("div");
  paramsHeader.className = "modal__subtitle";
  paramsHeader.textContent = "条件参数 (支持 result_of.* 引用或常量)";
  dialog.appendChild(paramsHeader);

  const paramsContainer = document.createElement("div");
  paramsContainer.className = "modal__grid";
  dialog.appendChild(paramsContainer);

  const addParamBtn = document.createElement("button");
  addParamBtn.type = "button";
  addParamBtn.className = "button button--ghost";
  addParamBtn.textContent = "新增参数";

  const paramRows = [];

  function addParamRow(key = "", value = "") {
    const row = document.createElement("div");
    row.className = "modal__field";

    const keyInput = document.createElement("input");
    keyInput.className = "modal__input";
    keyInput.placeholder = "参数名";
    keyInput.value = key;

    const valueInput = document.createElement("textarea");
    valueInput.className = "modal__input modal__textarea";
    valueInput.placeholder = "参数值，支持 JSON 或 result_of.* 引用";
    valueInput.value = value;

    row.appendChild(keyInput);
    row.appendChild(valueInput);
    paramsContainer.appendChild(row);
    paramRows.push({ keyInput, valueInput });
  }

  const params = node.params || {};
  const entries = Object.keys(params).length ? Object.entries(params) : [["", ""]];
  entries.forEach(([key, value]) => addParamRow(key, stringifyBinding(value)));

  addParamBtn.addEventListener("click", () => addParamRow());
  dialog.appendChild(addParamBtn);

  const actionsRow = document.createElement("div");
  actionsRow.className = "modal__actions";
  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "button button--danger";
  deleteBtn.textContent = "删除节点";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "button button--ghost";
  cancelBtn.textContent = "取消";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "button button--primary";
  saveBtn.textContent = "保存条件";
  actionsRow.appendChild(deleteBtn);
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

  saveBtn.addEventListener("click", () => {
    const updatedParams = {};
    paramRows.forEach(({ keyInput, valueInput }) => {
      const name = keyInput.value.trim();
      if (!name) return;
      const parsed = parseBinding(valueInput.value);
      if (parsed !== undefined) {
        updatedParams[name] = parsed;
      }
    });

    const normalizeTarget = (text) => {
      const trimmed = text.trim();
      if (!trimmed) return null;
      if (trimmed.toLowerCase() === "null") return null;
      return trimmed;
    };

    const trueTarget = normalizeTarget(branchInputs.true_to_node ? branchInputs.true_to_node.value : "");
    const falseTarget = normalizeTarget(branchInputs.false_to_node ? branchInputs.false_to_node.value : "");

    const updatedGraph = {
      ...(context.graph || currentWorkflow),
      nodes: (context.graph.nodes || []).map((n) =>
        n.id === node.id
          ? {
              ...n,
              display_name: nameInput.value.trim() || n.display_name,
              params: updatedParams,
              true_to_node: trueTarget,
              false_to_node: falseTarget,
            }
          : n,
      ),
    };
    rebuildEdgesFromBindings(updatedGraph);
    context.saveGraph(updatedGraph);
    render(currentTab);
    appendLog(`条件节点 ${node.id} 已更新`);
    close();
  });

  deleteBtn.addEventListener("click", () => {
    const confirmed = window.confirm(`确认删除条件节点 ${node.display_name || node.id} 吗？`);
    if (!confirmed) return;
    removeNodeFromGraph(node.id, context);
    close();
  });
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
    ctx,
    measureCtx: ctx,
    actionCatalog,
    lastRunResults,
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

function render(mode = currentTab) {
  const context = getTabContext(mode);
  const graph = context.graph || currentWorkflow;
  const tabKey = context.tabKey || mode;

  if (isCanvasTab(mode)) {
    attachCanvasTo(mode);
  }

  resizeCanvas(workflowCanvas, canvasPanel, graph, { viewState }, {
    ctx,
    measureCtx: ctx,
    actionCatalog,
    lastRunResults,
  });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, workflowCanvas.width, workflowCanvas.height);
  applyViewTransform(ctx, viewState);
  renderedNodes = [];
  if (!graph.nodes) return;
  lastPositions = syncPositions(graph, tabKey);

  const nodeGeometries = {};
  (graph.nodes || []).forEach((node) => {
    const pos = lastPositions[node.id];
    if (pos) {
      nodeGeometries[node.id] = measureNode(node, pos, mode, ctx, actionCatalog, lastRunResults);
    }
  });

  const edges = buildDisplayEdges(graph);
  const portUsage = { inputs: {}, outputs: {} };
  edges.forEach((edge) => {
    const from = lastPositions[edge.from_node];
    const to = lastPositions[edge.to_node];
    if (from && to) {
      const fromGeom = nodeGeometries[edge.from_node];
      const toGeom = nodeGeometries[edge.to_node];
      const fromPorts = fromGeom && fromGeom.portAnchors ? fromGeom.portAnchors.outputs : null;
      const toPorts = toGeom && toGeom.portAnchors ? toGeom.portAnchors.inputs : null;
      const fromAnchor =
        findAnchorByLabel(fromPorts, edge.from_field) ||
        pickPortAnchor(fromPorts, portUsage.outputs, edge.from_node, from);
      const toAnchor =
        findAnchorByLabel(toPorts, edge.to_field) || pickPortAnchor(toPorts, portUsage.inputs, edge.to_node, to);
      drawArrow(ctx, fromAnchor, toAnchor, edge.condition);
    }
  });

  graph.nodes.forEach((node) => {
    const pos = lastPositions[node.id];
    if (pos) {
      const geometry = nodeGeometries[node.id];
      const box = drawNode(node, pos, mode, geometry, ctx, lastRunResults);
      if (box) renderedNodes.push(box);
    }
  });

  if (linkingEdge && linkingEdge.fromAnchor) {
    const target = linkingEdge.toPoint || linkingEdge.fromAnchor;
    drawArrow(ctx, linkingEdge.fromAnchor, target);
  }

  drawWatermark();
}

function drawWatermark() {
  ctx.save();
  ctx.fillStyle = "rgba(124, 58, 237, 0.12)";
  ctx.font = "48px Inter";
  ctx.fillText("VelvetFlow 可视化", 32, workflowCanvas.height - 32);
  ctx.restore();
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
  if (tabId.startsWith("loop:")) {
    renderLoopInspector(tabId);
  }
  tabs.forEach((t) => {
    t.classList.toggle("tab--active", t === tab);
    t.setAttribute("aria-selected", t === tab);
  });
  tabContents.forEach((content) => {
    const isActive = content.dataset.view === currentTab;
    content.classList.toggle("tab-content--hidden", !isActive);
  });
  if (tabId === "json") {
    requestAnimationFrame(() => {
      if (autoSizeEditor._pending) autoSizeEditor();
      autoSizeEditor();
    });
  }
  render(currentTab);
}

function handleTabClick(event) {
  const tab = event.currentTarget || event.target;
  if (!tab || !tab.dataset.tab) return;
  switchToTab(tab.dataset.tab);
}

function screenPointToWorld(point) {
  const scale = viewState.scale || 1;
  const offset = viewState.offset || { x: 0, y: 0 };
  return {
    x: (point.x - offset.x) / scale,
    y: (point.y - offset.y) / scale,
  };
}

function canvasPointFromEvent(event) {
  const rect = workflowCanvas.getBoundingClientRect();
  const point = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
  return screenPointToWorld(point);
}

function updateZoomLabel() {
  if (zoomValue) {
    zoomValue.textContent = `${Math.round((viewState.scale || 1) * 100)}%`;
  }
}

function setViewScale(nextScale, focalPoint) {
  const clamped = Math.min(MAX_SCALE, Math.max(MIN_SCALE, nextScale));
  const rect = workflowCanvas.getBoundingClientRect();
  const focus = focalPoint || { x: rect.width / 2, y: rect.height / 2 };
  const worldFocus = screenPointToWorld(focus);

  viewState = {
    scale: clamped,
    offset: {
      x: focus.x - worldFocus.x * clamped,
      y: focus.y - worldFocus.y * clamped,
    },
  };

  updateZoomLabel();
  render(currentTab);
}

function resetView() {
  viewState = { scale: 1, offset: { x: 0, y: 0 } };
  updateZoomLabel();
  render(currentTab);
}

function handleCanvasDoubleClick(event) {
  if (!isCanvasTab(currentTab) || isDragging) return;
  const context = getTabContext(currentTab);
  const point = canvasPointFromEvent(event);
  const hit = findNodeByPoint(point, renderedNodes);
  if (!hit) return;
  const target = (context.graph.nodes || []).find((n) => n.id === hit.id);
  if (!target) return;

  if (target.type === "loop") {
    ensureLoopTab(target);
    return;
  }

  openNodeDialog(target, context);
}

function handleCanvasContextMenu(event) {
  if (!isCanvasTab(currentTab)) return;
  const point = canvasPointFromEvent(event);
  const hit = findNodeByPoint(point, renderedNodes);
  event.preventDefault();
  if (hit) return;
  showAddNodeMenu(event);
}

function handleCanvasMouseDown(event) {
  if (!isCanvasTab(currentTab)) return;
  hideAddNodeMenu();
  if (event.button !== 0) return;
  const point = canvasPointFromEvent(event);
  const portHit = findPortHit(point, renderedNodes, "right");
  if (portHit) {
    linkingEdge = { fromNodeId: portHit.nodeId, fromAnchor: portHit, toPoint: portHit };
    workflowCanvas.style.cursor = "crosshair";
    render(currentTab);
    return;
  }
  const hit = findNodeByPoint(point, renderedNodes);
  if (!hit) {
    isPanning = true;
    panStart = { x: event.clientX, y: event.clientY };
    panOrigin = { ...viewState.offset };
    workflowCanvas.style.cursor = "grabbing";
    return;
  }
  isDragging = true;
  dragNodeId = hit.id;
  dragBox = hit;
  dragOffset = { x: point.x - hit.x, y: point.y - hit.y };
  const ctxInfo = getTabContext(currentTab);
  dragTabKey = ctxInfo.tabKey || currentTab;
  workflowCanvas.style.cursor = "grabbing";
}

function handleCanvasMouseMove(event) {
  if (!isCanvasTab(currentTab)) return;
  if (linkingEdge) {
    linkingEdge.toPoint = canvasPointFromEvent(event);
    render(currentTab);
    return;
  }
  if (isPanning) {
    const dx = event.clientX - panStart.x;
    const dy = event.clientY - panStart.y;
    const scale = viewState.scale || 1;
    viewState.offset.x = panOrigin.x + dx / scale;
    viewState.offset.y = panOrigin.y + dy / scale;
    render(currentTab);
    return;
  }
  if (!isDragging || !dragNodeId || !dragBox) return;
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

function connectNodes(fromId, toId, context = getTabContext(currentTab), fromAnchor, toAnchor) {
  if (!fromId || !toId || fromId === toId) return false;
  const graph = context.graph || currentWorkflow;
  const normalized = normalizeWorkflow(graph);
  const fromField = fromAnchor && (fromAnchor.portKey || fromAnchor.label);
  const toField = toAnchor && (toAnchor.portKey || toAnchor.label);

  const edges = normalized.edges || [];
  const hasSameLink = edges.some((edge) => {
    const from = edge.from_node || edge.from;
    const to = edge.to_node || edge.to;
    const edgeFromField = edge.from_field || edge.fromField;
    const edgeToField = edge.to_field || edge.toField;
    return from === fromId && to === toId && edgeFromField === fromField && edgeToField === toField;
  });

  const nextEdges = hasSameLink
    ? edges
    : [
        ...edges,
        {
          from_node: fromId,
          to_node: toId,
          from_field: fromField,
          to_field: toField,
        },
      ];

  const graphWithEdges = { ...normalized, edges: nextEdges };
  const graphWithBindings = autoBindEdgeParams(graphWithEdges, fromId, toId, { fromField, toField });
  context.saveGraph(graphWithBindings);
  const labelHint = fromField && toField ? ` (${fromField} → ${toField})` : "";
  appendLog(`已连接 ${fromId} → ${toId}${labelHint}`);
  return true;
}

function autoBindEdgeParams(graph, fromId, toId, options = {}) {
  if (!graph || !Array.isArray(graph.nodes)) return graph;
  const nodes = [...graph.nodes];
  const fromIndex = nodes.findIndex((n) => n && n.id === fromId);
  const toIndex = nodes.findIndex((n) => n && n.id === toId);
  if (fromIndex === -1 || toIndex === -1) return graph;

  const source = nodes[fromIndex];
  const target = nodes[toIndex];
  const paramDefs = extractParamDefs(paramsSchemaFor(target.action_id, actionCatalog));
  if (!paramDefs.length) return graph;

  const outputDefs = extractOutputDefs(outputSchemaFor(source.action_id, source, actionCatalog));
  const availableOutputs = outputDefs.length
    ? outputDefs
    : (describeNode(source, actionCatalog, lastRunResults).outputs || []).map((name) => ({ name }));
  if (!availableOutputs.length) return graph;

  const nextParams = { ...(target.params || {}) };
  const bindings = [];

  const { fromField, toField } = options;
  if (fromField && toField) {
    const current = nextParams[toField];
    const matchedExplicit =
      availableOutputs.find((out) => out && out.name === fromField) ||
      availableOutputs.find((out) => out && out.name === toField);
    if ((current === undefined || current === "") && matchedExplicit && matchedExplicit.name) {
      nextParams[toField] = { __from__: `result_of.${fromId}.${matchedExplicit.name}` };
      bindings.push(`${toField}←${matchedExplicit.name}`);
    }
  }

  paramDefs.forEach((param) => {
    if (toField && param.name === toField && nextParams[toField] !== undefined && nextParams[toField] !== "") return;
    const current = nextParams[param.name];
    if (current !== undefined && current !== "") return;
    const matched = availableOutputs.find((out) => out && out.name === param.name) || availableOutputs[0];
    if (!matched || !matched.name) return;
    nextParams[param.name] = { __from__: `result_of.${fromId}.${matched.name}` };
    bindings.push(`${param.name}←${matched.name}`);
  });

  if (!bindings.length) return graph;

  nodes[toIndex] = { ...target, params: nextParams };
  appendLog(`已自动绑定 ${fromId} 输出到 ${toId} 输入：${bindings.join(", ")}`);
  return { ...graph, nodes };
}

function handleCanvasMouseUp(event) {
  if (!isCanvasTab(currentTab)) return;
  if (linkingEdge) {
    const point = canvasPointFromEvent(event);
    const targetPort = findPortHit(point, renderedNodes, "left");
    if (targetPort && targetPort.nodeId !== linkingEdge.fromNodeId) {
      const context = getTabContext(currentTab);
      connectNodes(linkingEdge.fromNodeId, targetPort.nodeId, context, linkingEdge.fromAnchor, targetPort);
      if (context.kind === "loop") {
        markTabDirty(context.tabKey, true);
      }
      render(currentTab);
    }
  }
  stopDragging();
}

function handleCanvasWheel(event) {
  if (!isCanvasTab(currentTab)) return;
  event.preventDefault();
  const rect = workflowCanvas.getBoundingClientRect();
  const focus = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  const delta = event.deltaY < 0 ? SCALE_STEP : -SCALE_STEP;
  setViewScale((viewState.scale || 1) + delta, focus);
}

function stopDragging() {
  if (!isDragging && !isPanning && !linkingEdge) return;
  isDragging = false;
  dragNodeId = null;
  dragBox = null;
  dragTabKey = null;
  isPanning = false;
  linkingEdge = null;
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
if (addNodeTypeSelect)
  addNodeTypeSelect.addEventListener("change", () => setSelectedNodeType(addNodeTypeSelect.value));
if (confirmAddNodeBtn)
  confirmAddNodeBtn.addEventListener("click", () => {
    addNodeToCurrentGraph();
    hideAddNodeMenu();
  });
if (cancelAddNodeBtn) cancelAddNodeBtn.addEventListener("click", hideAddNodeMenu);
if (closeAddNodeMenuBtn) closeAddNodeMenuBtn.addEventListener("click", hideAddNodeMenu);
document.addEventListener("click", (event) => {
  if (!addNodeMenu || addNodeMenu.classList.contains("hidden")) return;
  if (addNodeMenu.contains(event.target)) return;
  hideAddNodeMenu();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideAddNodeMenu();
});

tabs.forEach((tab) => tab.addEventListener("click", handleTabClick));
workflowCanvas.addEventListener("dblclick", handleCanvasDoubleClick);
workflowCanvas.addEventListener("contextmenu", handleCanvasContextMenu);
workflowCanvas.addEventListener("mousedown", handleCanvasMouseDown);
workflowCanvas.addEventListener("mousemove", handleCanvasMouseMove);
workflowCanvas.addEventListener("mouseup", handleCanvasMouseUp);
workflowCanvas.addEventListener("mouseleave", stopDragging);
workflowCanvas.addEventListener("wheel", handleCanvasWheel, { passive: false });
editHelpBtn.addEventListener("click", showEditHelp);

if (zoomInBtn) {
  zoomInBtn.addEventListener("click", () => setViewScale((viewState.scale || 1) + SCALE_STEP));
}
if (zoomOutBtn) {
  zoomOutBtn.addEventListener("click", () => setViewScale((viewState.scale || 1) - SCALE_STEP));
}
if (resetViewBtn) {
  resetViewBtn.addEventListener("click", resetView);
}

const editorResizeObserver = new ResizeObserver(() => render(currentTab));
editorResizeObserver.observe(workflowEditor);
window.addEventListener("resize", () => render(currentTab));
window.addEventListener("resize", autoSizeEditor);

populateActionSelect();
setSelectedNodeType(selectedNodeType);
loadActionCatalog();
updateEditor();
updateZoomLabel();
render();
appendLog("当前 workflow 为空，请输入需求开始规划或自行编辑 JSON。");
addChatMessage("你好，我是 VelvetFlow Agent，请描述你的业务需求。", "agent");
