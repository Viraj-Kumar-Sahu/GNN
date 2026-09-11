/**
 * ==========================================================
 * CORA GNN EXPLORER - MODERN INTERACTIVE JAVASCRIPT
 * ==========================================================
 */

// Cora 7 Classes Definition & Visual Styling
const CORA_CLASSES = {
  0: { name: "Case_Based", color: "#06b6d4", icon: "📁" },
  1: { name: "Genetic_Algorithms", color: "#10b981", icon: "🧬" },
  2: { name: "Neural_Networks", color: "#8b5cf6", icon: "🧠" },
  3: { name: "Probabilistic_Methods", color: "#f59e0b", icon: "🎲" },
  4: { name: "Reinforcement_Learning", color: "#f43f5e", icon: "🎯" },
  5: { name: "Rule_Learning", color: "#38bdf8", icon: "📜" },
  6: { name: "Theory", color: "#d946ef", icon: "📐" }
};

// Application State
const state = {
  nodesData: null,          // Precomputed metadata for all 2708 nodes
  currentNodeId: 0,         // Currently selected paper ID
  currentPrediction: null,  // Latest prediction from /predict/cora_node
  showLogits: false,        // Toggle logits vs probabilities
  perturbedFeatures: new Set(), // Modified bits in What-If sandbox
  customActiveIndices: [],  // Indices for custom predictor tab
  egoNodes: [],             // Nodes in 1-hop ego network for canvas
  egoEdges: [],             // Edges in 1-hop ego network
  animFrameId: null         // Animation frame for canvas
};

// ==========================================================
// INITIALIZATION
// ==========================================================
document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initEventListeners();
  initHeatmapCanvas();

  showToast("Initializing Cora GNN Explorer...", "info");

  // Load Model Info
  try {
    const modelRes = await fetch("/model_info");
    if (modelRes.ok) {
      const modelInfo = await modelRes.json();
      document.getElementById("nav-model-name").textContent = modelInfo.model_name || "SimpleGCN";
    }
  } catch (err) {
    console.warn("Could not fetch model_info:", err);
  }

  // Load Precomputed Cora Nodes Data
  try {
    const res = await fetch("/cora_nodes_data.json");
    if (!res.ok) throw new Error("Failed to load cora_nodes_data.json");
    state.nodesData = await res.json();
    showToast(`Loaded ${state.nodesData.length} papers & 1,433-dim vocabulary!`, "success");

    // Select initial node (0)
    selectNode(0);
  } catch (err) {
    console.error("Failed to load Cora nodes data:", err);
    showToast("Error loading dataset metadata. Please check server.", "error");
  }
});

// ==========================================================
// TAB SWITCHING
// ==========================================================
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");

      // Redraw canvases if switching to explorer
      if (targetId === "explorer-tab") {
        renderFeatureHeatmap();
        renderEgoCanvas();
      }
    });
  });
}

// ==========================================================
// EVENT LISTENERS
// ==========================================================
function initEventListeners() {
  const nodeInput = document.getElementById("node-input");
  const nodeSlider = document.getElementById("node-slider");
  const btnPredict = document.getElementById("btn-predict");
  const btnRandom = document.getElementById("btn-random");
  const classFilterSelect = document.getElementById("class-filter-select");
  const btnToggleLogits = document.getElementById("btn-toggle-logits");
  const btnCopyFeatures = document.getElementById("btn-copy-features");
  const featSearchInput = document.getElementById("feat-search-input");

  // Node input changes
  nodeInput.addEventListener("change", () => {
    let val = parseInt(nodeInput.value, 10);
    if (isNaN(val)) val = 0;
    val = Math.max(0, Math.min(2707, val));
    selectNode(val);
  });

  // Slider changes
  nodeSlider.addEventListener("input", () => {
    nodeInput.value = nodeSlider.value;
  });
  nodeSlider.addEventListener("change", () => {
    selectNode(parseInt(nodeSlider.value, 10));
  });

  // Predict button
  btnPredict.addEventListener("click", () => {
    selectNode(parseInt(nodeInput.value, 10));
  });

  // Random node
  btnRandom.addEventListener("click", () => {
    const randomId = Math.floor(Math.random() * 2708);
    selectNode(randomId);
  });

  // Hub shortcuts
  document.querySelectorAll(".hub-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const id = parseInt(btn.getAttribute("data-node"), 10);
      selectNode(id);
    });
  });

  // Jump to Class select
  classFilterSelect.addEventListener("change", () => {
    const classId = parseInt(classFilterSelect.value, 10);
    if (isNaN(classId) || !state.nodesData) return;
    // Find matching nodes for this class
    const matching = state.nodesData.filter(n => n.y === classId);
    if (matching.length > 0) {
      // Pick a random sample from that class
      const pick = matching[Math.floor(Math.random() * matching.length)];
      selectNode(pick.id);
    }
  });

  // Toggle Logits view
  btnToggleLogits.addEventListener("click", () => {
    state.showLogits = !state.showLogits;
    btnToggleLogits.textContent = state.showLogits ? "Show Probabilities" : "Show Raw Logits";
    if (state.currentPrediction) {
      renderProbabilityBars(state.currentPrediction);
    }
  });

  // Copy Active Features
  btnCopyFeatures.addEventListener("click", () => {
    if (!state.nodesData) return;
    const node = state.nodesData[state.currentNodeId];
    if (!node) return;
    navigator.clipboard.writeText(JSON.stringify(node.active)).then(() => {
      showToast(`Copied ${node.active.length} active feature indices!`, "success");
    });
  });

  // Feature Search Input
  featSearchInput.addEventListener("input", () => {
    filterActiveFeaturePills(featSearchInput.value.trim());
  });

  // What-If Sandbox Controls
  const btnToggleBit = document.getElementById("btn-toggle-bit");
  const perturbFeatureIdx = document.getElementById("perturb-feature-idx");
  const btnResetPerturb = document.getElementById("btn-reset-perturb");
  const btnRunWhatif = document.getElementById("btn-run-whatif");

  btnToggleBit.addEventListener("click", () => {
    const bit = parseInt(perturbFeatureIdx.value, 10);
    if (isNaN(bit) || bit < 0 || bit > 1432) {
      showToast("Feature index must be between 0 and 1432", "error");
      return;
    }
    togglePerturbationBit(bit);
  });

  btnResetPerturb.addEventListener("click", () => {
    resetPerturbations();
  });

  btnRunWhatif.addEventListener("click", () => {
    runWhatIfPrediction();
  });

  // Custom Predictor Tab
  initCustomPredictorTab();
}

// ==========================================================
// CORE NODE SELECTION & PREDICTION
// ==========================================================
async function selectNode(nodeId) {
  state.currentNodeId = nodeId;

  // Sync Input & Slider
  document.getElementById("node-input").value = nodeId;
  document.getElementById("node-slider").value = nodeId;
  document.getElementById("ego-node-id").textContent = `#${nodeId}`;

  // Reset Perturbations for new node
  state.perturbedFeatures.clear();
  updatePerturbationTags();
  document.getElementById("whatif-result-box").classList.add("hidden");

  // Update UI features immediately from precomputed data
  if (state.nodesData && state.nodesData[nodeId]) {
    const nodeMeta = state.nodesData[nodeId];
    renderFeatureInspector(nodeMeta);
    renderNeighbors(nodeMeta);
    setupEgoNetwork(nodeMeta);
  }

  // Call Model API: POST /predict/cora_node
  const predictBadge = document.getElementById("predict-badge");
  predictBadge.textContent = "Inferencing...";
  predictBadge.style.color = "var(--accent-cyan)";

  try {
    const res = await fetch("/predict/cora_node", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_index: [nodeId] })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Prediction failed");
    }

    const data = await res.json();
    predictBadge.textContent = "Predicted";
    predictBadge.style.color = "var(--accent-emerald)";

    if (data.predictions && data.predictions.length > 0) {
      const pred = data.predictions[0];
      state.currentPrediction = pred;
      renderPrediction(pred);
    }
  } catch (err) {
    console.error("Prediction error:", err);
    predictBadge.textContent = "Error";
    predictBadge.style.color = "var(--accent-rose)";
    showToast(`Inference error: ${err.message}`, "error");
  }
}

// ==========================================================
// PREDICTION DISPLAY
// ==========================================================
function renderPrediction(pred) {
  const classId = pred.predicted_class_id;
  const classInfo = CORA_CLASSES[classId] || { name: pred.predicted_class_name, color: "#8b5cf6", icon: "🎓" };

  const predClassName = document.getElementById("pred-class-name");
  const predIcon = document.getElementById("pred-icon");
  const predBox = document.getElementById("pred-class-box");
  const predConf = document.getElementById("pred-confidence");
  const predTruth = document.getElementById("pred-truth-pill");

  predClassName.textContent = classInfo.name.replace(/_/g, " ");
  predIcon.textContent = classInfo.icon;
  predBox.style.borderColor = classInfo.color;
  predBox.style.boxShadow = `0 4px 20px ${classInfo.color}33`;

  // Highest probability
  const maxProb = (Math.max(...pred.probability) * 100).toFixed(1);
  predConf.textContent = `${maxProb}% Confidence`;
  predConf.style.color = classInfo.color;
  predConf.style.borderColor = `${classInfo.color}66`;

  // Ground Truth comparison
  if (state.nodesData && state.nodesData[pred.node_index]) {
    const trueId = state.nodesData[pred.node_index].y;
    const trueInfo = CORA_CLASSES[trueId];
    const isMatch = (trueId === classId);

    predTruth.textContent = `Ground Truth: ${trueInfo ? trueInfo.name.replace(/_/g, " ") : trueId}`;
    predTruth.className = `truth-pill ${isMatch ? "match" : "mismatch"}`;
  }

  renderProbabilityBars(pred);
}

function renderProbabilityBars(pred) {
  const container = document.getElementById("prob-bars-list");
  container.innerHTML = "";

  const values = state.showLogits ? pred.logits : pred.probability;
  const maxVal = Math.max(...values);
  const minVal = Math.min(...values);

  for (let i = 0; i < 7; i++) {
    const classInfo = CORA_CLASSES[i];
    const val = values[i];
    const isPredicted = (i === pred.predicted_class_id);

    // Calculate percentage width for visual bar
    let barPercent = 0;
    let labelText = "";

    if (state.showLogits) {
      // Scale logits across 0-100%
      const range = maxVal - minVal || 1;
      barPercent = Math.max(4, Math.min(100, ((val - minVal) / range) * 100));
      labelText = val.toFixed(2);
    } else {
      barPercent = Math.max(2, val * 100);
      labelText = (val * 100).toFixed(1) + "%";
    }

    const row = document.createElement("div");
    row.className = "prob-bar-row";

    row.innerHTML = `
      <div class="prob-bar-meta">
        <span class="prob-class-label" style="${isPredicted ? `color: ${classInfo.color}; font-weight: 700;` : ''}">
          <span class="prob-class-indicator" style="background: ${classInfo.color};"></span>
          ${classInfo.name.replace(/_/g, " ")} ${isPredicted ? '✓' : ''}
        </span>
        <span class="prob-val-label" style="${isPredicted ? `color: ${classInfo.color}; font-weight: 700;` : ''}">
          ${labelText}
        </span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${barPercent}%; background: ${classInfo.color}; ${isPredicted ? `box-shadow: 0 0 10px ${classInfo.color};` : ''}"></div>
      </div>
    `;

    container.appendChild(row);
  }
}

// ==========================================================
// FEATURE INSPECTOR (CORE USER REQUIREMENT)
// ==========================================================
function renderFeatureInspector(nodeMeta) {
  const activeCount = nodeMeta.active.length;
  const totalFeatures = 1433;
  const density = ((activeCount / totalFeatures) * 100).toFixed(2);
  const sparsity = (100 - parseFloat(density)).toFixed(2);

  document.getElementById("feat-active-count").textContent = activeCount;
  document.getElementById("feat-density").textContent = `${density}%`;
  document.getElementById("feat-sparsity").textContent = `${sparsity}%`;

  // Render 1,433-Dimensional Heatmap
  renderFeatureHeatmap();

  // Render Active Feature Badges
  renderActiveFeaturePills(nodeMeta.active);
}

// Active Feature Badges
function renderActiveFeaturePills(activeIndices, filterQuery = "") {
  const container = document.getElementById("active-pills-list");
  container.innerHTML = "";

  const filtered = filterQuery 
    ? activeIndices.filter(idx => idx.toString().includes(filterQuery))
    : activeIndices;

  if (filtered.length === 0) {
    container.innerHTML = `<span class="text-muted-sm">No matching active features found.</span>`;
    return;
  }

  filtered.forEach(idx => {
    const pill = document.createElement("span");
    pill.className = "feat-pill";
    pill.textContent = `Feat #${idx}`;
    pill.title = `Click to toggle in What-If Sandbox (Feature ${idx})`;
    pill.addEventListener("click", () => {
      togglePerturbationBit(idx);
    });
    container.appendChild(pill);
  });
}

function filterActiveFeaturePills(query) {
  if (!state.nodesData) return;
  const node = state.nodesData[state.currentNodeId];
  if (!node) return;
  renderActiveFeaturePills(node.active, query);
}

// ==========================================================
// 1,433-FEATURE HEATMAP CANVAS
// ==========================================================
let heatmapCanvas, heatmapCtx;
const HEATMAP_COLS = 58;
const HEATMAP_ROWS = 25; // 58 * 25 = 1450 cells (> 1433)

function initHeatmapCanvas() {
  heatmapCanvas = document.getElementById("feature-canvas");
  heatmapCtx = heatmapCanvas.getContext("2d");

  heatmapCanvas.addEventListener("mousemove", handleHeatmapHover);
  heatmapCanvas.addEventListener("mouseleave", () => {
    document.getElementById("heatmap-hover-info").textContent = "Hover over any tile to inspect feature ID";
  });
  heatmapCanvas.addEventListener("click", handleHeatmapClick);
}

function getActiveFeatureSet() {
  const activeSet = new Set();
  if (state.nodesData && state.nodesData[state.currentNodeId]) {
    state.nodesData[state.currentNodeId].active.forEach(idx => activeSet.add(idx));
  }
  // Apply perturbation toggles
  state.perturbedFeatures.forEach(bit => {
    if (activeSet.has(bit)) {
      activeSet.delete(bit);
    } else {
      activeSet.add(bit);
    }
  });
  return activeSet;
}

function renderFeatureHeatmap() {
  if (!heatmapCanvas || !heatmapCtx) return;

  const w = heatmapCanvas.width;
  const h = heatmapCanvas.height;
  heatmapCtx.clearRect(0, 0, w, h);

  const activeSet = getActiveFeatureSet();
  const cellW = (w - 20) / HEATMAP_COLS;
  const cellH = (h - 14) / HEATMAP_ROWS;

  for (let idx = 0; idx < 1433; idx++) {
    const col = idx % HEATMAP_COLS;
    const row = Math.floor(idx / HEATMAP_COLS);
    const x = 10 + col * cellW;
    const y = 7 + row * cellH;

    const isActive = activeSet.has(idx);

    if (isActive) {
      heatmapCtx.fillStyle = "#00f0ff";
      heatmapCtx.shadowColor = "#00f0ff";
      heatmapCtx.shadowBlur = 5;
      heatmapCtx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
      heatmapCtx.shadowBlur = 0; // reset
    } else {
      heatmapCtx.fillStyle = "#152033";
      heatmapCtx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
    }
  }
}

function getFeatureIndexFromMouse(e) {
  const rect = heatmapCanvas.getBoundingClientRect();
  const scaleX = heatmapCanvas.width / rect.width;
  const scaleY = heatmapCanvas.height / rect.height;

  const mouseX = (e.clientX - rect.left) * scaleX - 10;
  const mouseY = (e.clientY - rect.top) * scaleY - 7;

  const cellW = (heatmapCanvas.width - 20) / HEATMAP_COLS;
  const cellH = (heatmapCanvas.height - 14) / HEATMAP_ROWS;

  const col = Math.floor(mouseX / cellW);
  const row = Math.floor(mouseY / cellH);

  if (col >= 0 && col < HEATMAP_COLS && row >= 0 && row < HEATMAP_ROWS) {
    const idx = row * HEATMAP_COLS + col;
    if (idx >= 0 && idx < 1433) return idx;
  }
  return null;
}

function handleHeatmapHover(e) {
  const idx = getFeatureIndexFromMouse(e);
  const infoEl = document.getElementById("heatmap-hover-info");
  if (idx !== null) {
    const activeSet = getActiveFeatureSet();
    const isActive = activeSet.has(idx);
    infoEl.textContent = `Feature #${idx}: ${isActive ? 'PRESENT (1)' : 'ABSENT (0)'} (Click to toggle)`;
    infoEl.style.color = isActive ? "var(--accent-cyan)" : "var(--text-muted)";
  } else {
    infoEl.textContent = "Hover over any tile to inspect feature ID";
    infoEl.style.color = "var(--text-dim)";
  }
}

function handleHeatmapClick(e) {
  const idx = getFeatureIndexFromMouse(e);
  if (idx !== null) {
    togglePerturbationBit(idx);
  }
}

// ==========================================================
// FEATURE PERTURBATION & WHAT-IF SANDBOX
// ==========================================================
function togglePerturbationBit(bit) {
  if (state.perturbedFeatures.has(bit)) {
    state.perturbedFeatures.delete(bit);
  } else {
    state.perturbedFeatures.add(bit);
  }
  updatePerturbationTags();
  renderFeatureHeatmap();
}

function resetPerturbations() {
  state.perturbedFeatures.clear();
  updatePerturbationTags();
  renderFeatureHeatmap();
  document.getElementById("whatif-result-box").classList.add("hidden");
  showToast("Perturbations reset", "info");
}

function updatePerturbationTags() {
  const container = document.getElementById("perturb-tags-container");
  container.innerHTML = "";

  if (state.perturbedFeatures.size === 0) {
    container.innerHTML = `<span class="text-muted-sm">No modified features. Pick a feature index above to flip it ON or OFF.</span>`;
    return;
  }

  const origActiveSet = new Set(
    state.nodesData && state.nodesData[state.currentNodeId] 
      ? state.nodesData[state.currentNodeId].active 
      : []
  );

  state.perturbedFeatures.forEach(bit => {
    const isAdded = !origActiveSet.has(bit);
    const tag = document.createElement("span");
    tag.className = `toggled-tag ${isAdded ? 'added' : 'removed'}`;
    tag.innerHTML = `
      ${isAdded ? '+' : '-'} Feat #${bit}
      <span class="tag-remove">&times;</span>
    `;
    tag.querySelector(".tag-remove").addEventListener("click", () => {
      togglePerturbationBit(bit);
    });
    container.appendChild(tag);
  });
}

// Calls POST /predict with perturbed 1,433-feature vector
async function runWhatIfPrediction() {
  if (!state.nodesData || !state.nodesData[state.currentNodeId]) return;

  const activeSet = getActiveFeatureSet();
  // Build 1433 float array
  const vector = new Array(1433).fill(0.0);
  activeSet.forEach(idx => {
    if (idx >= 0 && idx < 1433) vector[idx] = 1.0;
  });

  showToast("Running What-If Inference...", "info");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node_feature: [vector],
        edge_index: [[0], [0]] // self-loop
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "What-if prediction failed");
    }

    const data = await res.json();
    if (data.predictions && data.predictions.length > 0) {
      const newPred = data.predictions[0];
      renderWhatIfResult(newPred);
    }
  } catch (err) {
    console.error("What-If error:", err);
    showToast(`What-If failed: ${err.message}`, "error");
  }
}

function renderWhatIfResult(newPred) {
  const box = document.getElementById("whatif-result-box");
  box.classList.remove("hidden");

  const origClass = state.currentPrediction ? state.currentPrediction.predicted_class_name : "Unknown";
  const newClass = newPred.predicted_class_name;
  const newConf = (Math.max(...newPred.probability) * 100).toFixed(1);
  const isShift = (origClass !== newClass);

  box.innerHTML = `
    <div style="font-weight: 700; color: ${isShift ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; margin-bottom: 4px;">
      ${isShift ? '⚠️ Prediction Class Shifted!' : '✓ Class Remained Stable'}
    </div>
    <div>
      Original: <strong>${origClass.replace(/_/g, " ")}</strong> &rarr; 
      What-If: <strong style="color: ${CORA_CLASSES[newPred.predicted_class_id].color}">${newClass.replace(/_/g, " ")}</strong> (${newConf}% Confidence)
    </div>
    <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">
      Active features in test: ${getActiveFeatureSet().size} / 1,433
    </div>
  `;
}

// ==========================================================
// CITATION GRAPH & EGO-NETWORK
// ==========================================================
function renderNeighbors(nodeMeta) {
  document.getElementById("degree-badge").textContent = `Degree: ${nodeMeta.deg}`;
  document.getElementById("nbr-count").textContent = nodeMeta.nbrs.length;

  const container = document.getElementById("neighbors-list");
  container.innerHTML = "";

  if (nodeMeta.nbrs.length === 0) {
    container.innerHTML = `<span class="text-muted-sm">No direct citation links.</span>`;
    return;
  }

  nodeMeta.nbrs.forEach(nbrId => {
    const pill = document.createElement("button");
    pill.className = "nbr-pill";
    pill.textContent = `#${nbrId}`;
    pill.title = `Jump to Paper #${nbrId}`;
    pill.addEventListener("click", () => {
      selectNode(nbrId);
    });
    container.appendChild(pill);
  });
}

function setupEgoNetwork(nodeMeta) {
  const canvas = document.getElementById("ego-canvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const centerX = w / 2;
  const centerY = h / 2;

  // Center node
  const nodes = [{
    id: nodeMeta.id,
    x: centerX,
    y: centerY,
    vx: 0,
    vy: 0,
    r: 16,
    isCenter: true,
    classId: nodeMeta.y
  }];

  const nbrs = nodeMeta.nbrs;
  const count = Math.min(nbrs.length, 25);
  const radius = Math.min(w, h) * 0.38;

  for (let i = 0; i < count; i++) {
    const angle = (i / count) * 2 * Math.PI;
    const nbrId = nbrs[i];
    const nbrMeta = state.nodesData[nbrId];
    const classId = nbrMeta ? nbrMeta.y : 0;

    nodes.push({
      id: nbrId,
      x: centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 20,
      y: centerY + radius * Math.sin(angle) + (Math.random() - 0.5) * 20,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      r: 9,
      isCenter: false,
      classId: classId
    });
  }

  state.egoNodes = nodes;
  renderEgoCanvas();
}

function renderEgoCanvas() {
  const canvas = document.getElementById("ego-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const nodes = state.egoNodes;
  if (!nodes || nodes.length === 0) return;

  const centerNode = nodes[0];

  // Draw Links
  ctx.strokeStyle = "rgba(160, 174, 192, 0.25)";
  ctx.lineWidth = 1.2;

  for (let i = 1; i < nodes.length; i++) {
    ctx.beginPath();
    ctx.moveTo(centerNode.x, centerNode.y);
    ctx.lineTo(nodes[i].x, nodes[i].y);
    ctx.stroke();
  }

  // Draw Nodes
  nodes.forEach(node => {
    const classInfo = CORA_CLASSES[node.classId] || { color: "#8b5cf6" };

    if (node.isCenter) {
      // Glow ring
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r + 6, 0, 2 * Math.PI);
      ctx.fillStyle = `${classInfo.color}33`;
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r, 0, 2 * Math.PI);
    ctx.fillStyle = classInfo.color;
    ctx.shadowColor = classInfo.color;
    ctx.shadowBlur = node.isCenter ? 12 : 4;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = node.isCenter ? 2 : 1;
    ctx.stroke();

    // Text Label
    ctx.fillStyle = "#ffffff";
    ctx.font = node.isCenter ? "bold 10px JetBrains Mono" : "8px JetBrains Mono";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`#${node.id}`, node.x, node.isCenter ? node.y : node.y + node.r + 10);
  });
}

// Click on Ego Canvas to switch node
document.getElementById("ego-canvas").addEventListener("click", e => {
  const canvas = document.getElementById("ego-canvas");
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  for (const node of state.egoNodes) {
    const dx = clickX - node.x;
    const dy = clickY - node.y;
    if (dx * dx + dy * dy <= (node.r + 4) * (node.r + 4)) {
      selectNode(node.id);
      break;
    }
  }
});

// ==========================================================
// TAB 2: CUSTOM GRAPH PREDICTOR (POST /predict)
// ==========================================================
const PRESETS = {
  neural_net: [19, 81, 146, 315, 623, 720, 890, 1102, 1340],
  genetics: [42, 118, 204, 388, 512, 680, 830, 995, 1204],
  theory: [5, 56, 120, 230, 340, 560, 780, 910, 1150],
  probabilistic: [12, 98, 175, 290, 410, 600, 750, 890, 1280],
  random: Array.from({ length: 15 }, () => Math.floor(Math.random() * 1433)).sort((a, b) => a - b)
};

function initCustomPredictorTab() {
  const presetBtns = document.querySelectorAll(".preset-btn");
  const textarea = document.getElementById("custom-features-input");
  const btnSubmit = document.getElementById("btn-submit-custom");

  // Load initial preset
  textarea.value = PRESETS.neural_net.join(", ");

  presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      presetBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const presetKey = btn.getAttribute("data-preset");
      if (presetKey === "random") {
        PRESETS.random = Array.from({ length: 15 }, () => Math.floor(Math.random() * 1433)).sort((a, b) => a - b);
      }
      textarea.value = PRESETS[presetKey].join(", ");
    });
  });

  btnSubmit.addEventListener("click", runCustomPrediction);
}

async function runCustomPrediction() {
  const textarea = document.getElementById("custom-features-input");
  const edgeInput = document.getElementById("custom-edge-input");
  const outputBox = document.getElementById("custom-prediction-output");

  // Parse active indices
  const text = textarea.value.trim();
  if (!text) {
    showToast("Please enter at least one active feature index", "error");
    return;
  }

  const indices = text.split(",")
    .map(s => parseInt(s.trim(), 10))
    .filter(n => !isNaN(n) && n >= 0 && n < 1433);

  if (indices.length === 0) {
    showToast("No valid feature indices found (must be 0 to 1432)", "error");
    return;
  }

  // Construct 1,433 vector
  const vector = new Array(1433).fill(0.0);
  indices.forEach(idx => vector[idx] = 1.0);

  // Parse custom edges if present
  let edgeIndex = null;
  const edgeText = edgeInput.value.trim();
  if (edgeText) {
    try {
      edgeIndex = JSON.parse(edgeText);
    } catch (e) {
      showToast("Invalid JSON format for edge_index. Using default self-loop.", "error");
    }
  }

  outputBox.innerHTML = `<span style="color: var(--accent-cyan);">Sending inference request to POST /predict...</span>`;

  try {
    const payload = {
      node_feature: [vector],
      ...(edgeIndex ? { edge_index: edgeIndex } : {})
    };

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }

    const data = await res.json();
    const pred = data.predictions[0];
    const classInfo = CORA_CLASSES[pred.predicted_class_id] || { color: "#8b5cf6" };
    const conf = (Math.max(...pred.probability) * 100).toFixed(1);

    outputBox.innerHTML = `
      <div style="font-size: 1.1rem; font-weight: 800; color: ${classInfo.color}; margin-bottom: 8px;">
        ${classInfo.name.replace(/_/g, " ")} (${conf}% Confidence)
      </div>
      <div style="margin-bottom: 12px; color: var(--text-muted); font-size: 0.76rem;">
        Active Features (${indices.length} of 1,433): [${indices.slice(0, 15).join(", ")}${indices.length > 15 ? '...' : ''}]
      </div>
      <div style="font-weight: 700; margin-bottom: 6px; color: var(--text-bright);">Class Probabilities:</div>
      <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; color: #a5b4fc; overflow-x: auto;">${
        JSON.stringify(
          pred.probability.map((p, i) => ({ class: CORA_CLASSES[i].name, prob: (p * 100).toFixed(2) + "%" })),
          null,
          2
        )
      }</pre>
    `;
    showToast("Custom prediction completed!", "success");
  } catch (err) {
    console.error("Custom predict error:", err);
    outputBox.innerHTML = `<span style="color: var(--accent-rose);">Error: ${err.message}</span>`;
    showToast(`Error: ${err.message}`, "error");
  }
}

// ==========================================================
// TOAST NOTIFICATION UTILITY
// ==========================================================
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
