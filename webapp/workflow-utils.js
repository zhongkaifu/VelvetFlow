export function createEmptyWorkflow() {
  return {
    workflow_name: "",
    description: "",
    nodes: [],
    edges: [],
  };
}

export function generateNodeId(prefix, graph) {
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

export function createDefaultNode(type, graph, options = {}, actionCatalog = {}) {
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

export function normalizeEdge(edge) {
  if (Array.isArray(edge) && edge.length >= 2) {
    return { from_node: edge[0], to_node: edge[1], condition: edge[2] };
  }
  return {
    from_node: edge.from_node || edge.from,
    to_node: edge.to_node || edge.to,
    condition: edge.condition,
    from_field: edge.from_field || edge.fromField,
    to_field: edge.to_field || edge.toField,
  };
}

export function normalizeWorkflow(workflow) {
  const edges = Array.isArray(workflow.edges) ? workflow.edges.map(normalizeEdge) : [];
  return { ...workflow, edges };
}

export function collectParamKeys(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(String);
  if (typeof value === "object") return Object.keys(value);
  return [String(value)];
}

export function paramsSchemaFor(actionId, actionCatalog = {}) {
  const action = actionCatalog[actionId];
  return (action && (action.params_schema || action.arg_schema)) || {};
}

export function outputSchemaFor(actionId, node, actionCatalog = {}) {
  if (node && node.out_params_schema) return node.out_params_schema;
  const action = actionCatalog[actionId];
  return (action && (action.output_schema || action.out_params_schema)) || {};
}

export function extractParamDefs(schema) {
  const properties = schema && typeof schema === "object" ? schema.properties || {} : {};
  const required = Array.isArray(schema && schema.required) ? schema.required : [];
  return Object.entries(properties).map(([name, def]) => ({
    name,
    type: def && def.type ? def.type : "",
    description: def && def.description ? def.description : "",
    required: required.includes(name),
  }));
}

export function extractOutputDefs(schema) {
  const properties = schema && typeof schema === "object" ? schema.properties || {} : {};
  const required = Array.isArray(schema && schema.required) ? schema.required : [];
  return Object.entries(properties).map(([name, def]) => ({
    name,
    type: def && def.type ? def.type : "",
    description: def && def.description ? def.description : "",
    required: required.includes(name),
  }));
}

export function describeNode(node, actionCatalog = {}, lastRunResults = {}) {
  const schemaInputs =
    node.type === "action" ? extractParamDefs(paramsSchemaFor(node.action_id, actionCatalog)) : [];
  const schemaOutputs =
    node.type === "action"
      ? extractOutputDefs(outputSchemaFor(node.action_id, node, actionCatalog))
      : [];

  const rawInputs = collectParamKeys(node.inputs || node.input_params || node.params || node.args);
  const rawOutputs = collectParamKeys(node.outputs || node.output_params || node.output);
  const inputs = Array.from(new Set([...schemaInputs.map((p) => p.name), ...rawInputs]));
  const outputs = Array.from(new Set([...schemaOutputs.map((p) => p.name), ...rawOutputs]));

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

export function summarizeValue(value, limit = 160) {
  try {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    if (text.length > limit) return `${text.slice(0, limit)}…`;
    return text;
  } catch (error) {
    return String(value);
  }
}

export function stringifyBinding(value) {
  if (value && typeof value === "object" && value.__from__) return String(value.__from__);
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch (error) {
    return String(value);
  }
}

export function parseBinding(text) {
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

export function collectLoopExportOptions(node) {
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

export function collectRefsFromValue(value, refs) {
  if (!refs) return;
  if (Array.isArray(value)) {
    value.forEach((item) => collectRefsFromValue(item, refs));
  } else if (value && typeof value === "object") {
    if (value.__from__) {
      const refId = String(value.__from__).split(".")[1];
      if (refId) refs.add(refId);
    }
    Object.values(value).forEach((v) => collectRefsFromValue(v, refs));
  }
}
