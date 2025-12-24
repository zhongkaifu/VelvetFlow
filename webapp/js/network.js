async function streamEvents(url, options, onEvent) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const hint =
      [404, 405, 501].includes(response.status)
        ? "The backend API is not running. Please run `python webapp/server.py` and try again."
        : "";
    throw new Error(detail.detail || detail.message || hint || response.statusText);
  }

  if (!response.body) {
    throw new Error("This browser does not support streaming responses");
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
        console.warn("Failed to parse streaming message", payload, error);
      }
    }
  }
}

function extractWorkflowFromEvent(event) {
  if (!event) return null;
  if (event.workflow) return event.workflow;
  if (event.full_workflow) return event.full_workflow;
  if (event.visible_subgraph && event.visible_subgraph.workflow) return event.visible_subgraph.workflow;
  if (event.visible_subgraph) return event.visible_subgraph;
  return null;
}

function maybeApplyWorkflowFromEvent(workflow, sourceLabel, logSnapshot = false) {
  if (!workflow) return false;
  const normalized = normalizeWorkflow(workflow);
  const previous = currentWorkflow ? JSON.stringify(currentWorkflow) : "";
  const next = JSON.stringify(normalized);
  if (previous === next) return false;

  currentWorkflow = normalized;
  updateEditor();
  refreshWorkflowCanvases();
  if (sourceLabel) {
    appendLog(`${sourceLabel} updated the workflow; canvas refreshed`);
  }
  if (logSnapshot) {
    logWorkflowSnapshot(currentWorkflow, `${sourceLabel || "live"} workflow DAG`);
  }
  return true;
}

function isEmptyWorkflow(workflow) {
  const nodes = workflow && workflow.nodes;
  return !nodes || nodes.length === 0;
}

async function requestPlan(requirement) {
  setStatus("Planning", "warning");
  appendLog(`Received requirement: ${requirement}`);
  addChatMessage(`Received requirement: “${requirement}”. Starting planning/validation/repair.`, "agent");
  try {
    const existingWorkflow =
      currentWorkflow && Array.isArray(currentWorkflow.nodes) && currentWorkflow.nodes.length > 0
        ? currentWorkflow
        : null;

    if (existingWorkflow) {
      addChatMessage("Will update and self-repair based on the current workflow while keeping nodes and edges aligned.", "agent");
    } else {
      addChatMessage("Will build a brand-new workflow from scratch with automatic validation and repair.", "agent");
    }
    addChatMessage("Calling the VelvetFlow Planner. Please wait while intermediate states are processed...", "agent");

    let streamError = null;
    let finalWorkflow = null;
    let finalSuggestions = [];
    let finalToolGapMessage = null;
    let finalToolGapSuggestions = [];
    let needsMoreDetail = false;

    await streamEvents(
      "/api/plan/stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requirement, existing_workflow: existingWorkflow }),
      },
      (event) => {
        if (!event || !event.type) return;
        const latestWorkflow = extractWorkflowFromEvent(event);
        if (event.type === "log" && event.message) {
          appendLog(event.message);
          addChatMessage(`Workflow update: ${event.message}`, "agent");
          if (latestWorkflow) {
            maybeApplyWorkflowFromEvent(latestWorkflow, "Planning log", true);
          }
        } else if (event.type === "snapshot" || event.type === "partial") {
          if (latestWorkflow) {
            const sourceLabel = event.stage ? `Planning stage: ${event.stage}` : "Planning snapshot";
            maybeApplyWorkflowFromEvent(latestWorkflow, sourceLabel, true);
          }
          if (typeof event.progress === "number") {
            const percent = Math.round(event.progress * 100);
            setStatus(`Building ${percent}%`, "warning");
          }
        } else if (event.type === "result" && latestWorkflow) {
          finalWorkflow = normalizeWorkflow(latestWorkflow);
          finalSuggestions = Array.isArray(event.suggestions) ? event.suggestions : [];
          finalToolGapMessage = typeof event.tool_gap_message === "string" ? event.tool_gap_message : null;
          finalToolGapSuggestions = Array.isArray(event.tool_gap_suggestions) ? event.tool_gap_suggestions : [];
          needsMoreDetail = Boolean(event.needs_more_detail);
        } else if (event.type === "error") {
          streamError = event.message || "Unknown error";
        }
      },
    );

    if (streamError) {
      throw new Error(streamError);
    }
    if (!finalWorkflow) {
      throw new Error("Did not receive a build result");
    }

    const empty = isEmptyWorkflow(finalWorkflow);
    if (!empty) {
      lastRunResults = {};
      clearPositionCaches();
      closeAllLoopTabs(true);
      currentWorkflow = finalWorkflow;
      updateEditor();
      refreshWorkflowCanvases();
      setStatus("Build completed", "success");
      addChatMessage("Finished DAG planning and validation. You can review and keep editing on the canvas.", "agent");
    } else {
      setStatus("Waiting for more detail", "warning");
      const hintSource = finalToolGapSuggestions.length ? finalToolGapSuggestions : finalSuggestions;
      const hints = hintSource && hintSource.length
        ? hintSource.map((item, idx) => `${idx + 1}. ${item}`).join("\n")
        : "Please add specifics such as data sources, trigger timing, key checks, or how results should be delivered.";
      const intro = finalToolGapMessage
        ? `${finalToolGapMessage}\n`
        : "The current requirement needs more context to plan a clear flow. Please add more concrete constraints or goals.\n";
      addChatMessage(
        `${intro}${hints}`,
        "agent",
      );
    }

    return {
      workflow: finalWorkflow,
      suggestions: finalSuggestions,
      needsMoreDetail: empty || needsMoreDetail,
      toolGapMessage: finalToolGapMessage,
      toolGapSuggestions: finalToolGapSuggestions,
    };
  } catch (error) {
    setStatus("Build failed", "danger");
    appendLog(`Planning failed: ${error.message}`);
    addChatMessage(`Planning failed: ${error.message}. Please check whether OPENAI_API_KEY is configured.`, "agent");
    return { workflow: null, suggestions: [], needsMoreDetail: false };
  }
}

async function requestRun() {
  setStatus("Running", "warning");
  addChatMessage("Started executing the workflow. Streaming run logs in real time.", "agent");
  appendLog("Starting execution of the current workflow ...");
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
        const latestWorkflow = extractWorkflowFromEvent(event);
        if (event.type === "log" && event.message) {
          appendLog(event.message);
          if (latestWorkflow) {
            maybeApplyWorkflowFromEvent(latestWorkflow, "Run log");
          }
        } else if (event.type === "result") {
          finalResult = event.result || {};
          finalStatus = event.status || "completed";
          if (latestWorkflow) {
            maybeApplyWorkflowFromEvent(latestWorkflow, "Run result");
          }
        } else if (event.type === "error") {
          streamError = event.message || "Unknown error";
        }
      },
    );

    if (streamError) {
      throw new Error(streamError);
    }
    if (!finalResult) {
      throw new Error("Did not receive a run result");
    }

    setStatus(finalStatus === "completed" ? "Run completed" : "Suspended and waiting for callback", "success");
    lastRunResults = finalResult;
    render(currentTab);
    appendLog(`Run result: ${finalStatus}`);
    addChatMessage(`Execution status: ${finalStatus}. Result: ${JSON.stringify(finalResult, null, 2)}`, "agent");
  } catch (error) {
    setStatus("Run failed", "danger");
    appendLog(`Run failed: ${error.message}`);
    addChatMessage(`Execution failed: ${error.message}`, "agent");
  }
}
