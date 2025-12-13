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

function maybeApplyWorkflowFromEvent(workflow, sourceLabel) {
  if (!workflow) return false;
  const normalized = normalizeWorkflow(workflow);
  const previous = currentWorkflow ? JSON.stringify(currentWorkflow) : "";
  const next = JSON.stringify(normalized);
  if (previous === next) return false;

  currentWorkflow = normalized;
  updateEditor();
  refreshWorkflowCanvases();
  if (sourceLabel) {
    appendLog(`${sourceLabel} 更新了 workflow，画布已刷新`);
  }
  return true;
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
          if (event.workflow) {
            maybeApplyWorkflowFromEvent(event.workflow);
          }
        } else if ((event.type === "snapshot" || event.type === "partial") && event.workflow) {
          maybeApplyWorkflowFromEvent(event.workflow);
          if (typeof event.progress === "number") {
            const percent = Math.round(event.progress * 100);
            setStatus(`构建中 ${percent}%`, "warning");
          }
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
    refreshWorkflowCanvases();
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
    let streamError = null;
    let finalResult = null;
    let finalStatus = "completed";

    await streamEvents(
      "/api/run/stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: currentWorkflow }),
      },
      (event) => {
        if (!event || !event.type) return;
        if (event.type === "log" && event.message) {
          appendLog(event.message);
          if (event.workflow) {
            maybeApplyWorkflowFromEvent(event.workflow, "运行日志");
          }
        } else if (event.type === "result") {
          finalResult = event.result || {};
          finalStatus = event.status || "completed";
          if (event.workflow) {
            maybeApplyWorkflowFromEvent(event.workflow, "运行结果");
          }
        } else if (event.type === "error") {
          streamError = event.message || "未知错误";
        }
      },
    );

    if (streamError) {
      throw new Error(streamError);
    }
    if (!finalResult) {
      throw new Error("未收到运行结果");
    }

    setStatus(finalStatus === "completed" ? "运行完成" : "挂起等待回调", "success");
    lastRunResults = finalResult;
    render(currentTab);
    appendLog(`运行结果: ${finalStatus}`);
    addChatMessage(`执行状态：${finalStatus}。结果：${JSON.stringify(finalResult, null, 2)}`, "agent");
  } catch (error) {
    setStatus("运行失败", "danger");
    appendLog(`运行失败: ${error.message}`);
    addChatMessage(`执行失败：${error.message}`, "agent");
  }
}
