function openNodeDialog(node, context = getTabContext()) {
  if (!node) return;
  if (node.type === "condition") {
    openConditionDialog(node, context);
    return;
  }
  if (node.type !== "action") {
    addChatMessage("Only action or condition node parameters can be edited right now.", "agent");
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
  title.textContent = `Edit node: ${node.display_name || node.action_id || node.id}`;
  dialog.appendChild(title);

  const idRow = document.createElement("div");
  idRow.className = "modal__row";
  const idLabel = document.createElement("label");
  idLabel.textContent = "Node ID";
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
  actionLabel.textContent = "Business tool (action_id)";
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
  inputsHeader.textContent = "Input params (use result_of.* or literals)";
  dialog.appendChild(inputsHeader);

  const inputsContainer = document.createElement("div");
  inputsContainer.className = "modal__grid";
  dialog.appendChild(inputsContainer);

  const outputsHeader = document.createElement("div");
  outputsHeader.className = "modal__subtitle";
  outputsHeader.textContent = "Output params";
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
  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "button button--danger";
  deleteBtn.textContent = "Delete node";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "button button--ghost";
  cancelBtn.textContent = "Cancel";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "button button--primary";
  saveBtn.textContent = "Save and refresh";
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
    appendLog(`Node ${node.id} updated and re-rendered`);
    logWorkflowSnapshot(currentWorkflow, "DAG after node params update");
    close();
  });

  deleteBtn.addEventListener("click", () => {
    const confirmed = window.confirm(`Delete node ${node.display_name || node.id}?`);
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
  title.textContent = `Edit condition node: ${node.display_name || node.id}`;
  dialog.appendChild(title);

  const idRow = document.createElement("div");
  idRow.className = "modal__row";
  const idLabel = document.createElement("label");
  idLabel.textContent = "Node ID";
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
  nameLabel.textContent = "Display name";
  const nameInput = document.createElement("input");
  nameInput.className = "modal__input";
  nameInput.value = node.display_name || "";
  nameRow.appendChild(nameLabel);
  nameRow.appendChild(nameInput);
  dialog.appendChild(nameRow);

  const branchHeader = document.createElement("div");
  branchHeader.className = "modal__subtitle";
  branchHeader.textContent = "Condition branches (true/false_to_node)";
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
    field.innerHTML = `<div class="modal__field-label">${label}<span class="modal__hint">Enter a node id or leave blank to end</span></div>`;
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
  paramsHeader.textContent = "Condition params (supports result_of.* or literals)";
  dialog.appendChild(paramsHeader);

  const paramsContainer = document.createElement("div");
  paramsContainer.className = "modal__grid";
  dialog.appendChild(paramsContainer);

  const addParamBtn = document.createElement("button");
  addParamBtn.type = "button";
  addParamBtn.className = "button button--ghost";
  addParamBtn.textContent = "Add parameter";

  const paramRows = [];

  function addParamRow(key = "", value = "") {
    const row = document.createElement("div");
    row.className = "modal__field";

    const keyInput = document.createElement("input");
    keyInput.className = "modal__input";
    keyInput.placeholder = "Parameter name";
    keyInput.value = key;

    const valueInput = document.createElement("textarea");
    valueInput.className = "modal__input modal__textarea";
    valueInput.placeholder = "Parameter value, supports JSON or result_of.* reference";
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
  deleteBtn.textContent = "Delete node";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "button button--ghost";
  cancelBtn.textContent = "Cancel";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "button button--primary";
  saveBtn.textContent = "Save condition";
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
    appendLog(`Condition node ${node.id} updated`);
    logWorkflowSnapshot(currentWorkflow, "DAG after condition update");
    close();
  });

  deleteBtn.addEventListener("click", () => {
    const confirmed = window.confirm(`Delete condition node ${node.display_name || node.id}?`);
    if (!confirmed) return;
    removeNodeFromGraph(node.id, context);
    close();
  });
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
