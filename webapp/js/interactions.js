
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

function handleCanvasContextMenu(event) {
  if (!isCanvasTab(currentTab)) return;
  const point = canvasPointFromEvent(event);
  const hit = findNodeByPoint(point);
  event.preventDefault();
  if (hit) return;
  showAddNodeMenu(event);
}

function handleCanvasMouseDown(event) {
  if (!isCanvasTab(currentTab)) return;
  hideAddNodeMenu();
  if (event.button !== 0) return;
  const point = canvasPointFromEvent(event);
  const hit = findNodeByPoint(point);
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

function handleCanvasWheel(event) {
  if (!isCanvasTab(currentTab)) return;
  event.preventDefault();
  const rect = workflowCanvas.getBoundingClientRect();
  const focus = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  const delta = event.deltaY < 0 ? SCALE_STEP : -SCALE_STEP;
  setViewScale((viewState.scale || 1) + delta, focus);
}

function stopDragging() {
  if (!isDragging && !isPanning) return;
  isDragging = false;
  dragNodeId = null;
  dragBox = null;
  dragTabKey = null;
  isPanning = false;
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
workflowCanvas.addEventListener("mouseup", stopDragging);
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
