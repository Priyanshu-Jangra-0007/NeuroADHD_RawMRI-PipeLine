const state = {
  health: null,
  runs: [],
  selectedRunId: null,
  selectedSubjectId: null,
  refreshTimer: null,
  loadingSubject: false,
  toastTimer: null,
};

const $ = (id) => document.getElementById(id);

async function api(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const payload = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : await response.blob();
  if (!response.ok) throw new Error(payload?.error || `Request failed (${response.status})`);
  return payload;
}

function showToast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => el.classList.remove("show"), 3600);
}

function fmtDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

function statusClass(status) {
  if (status === "SUCCESS") return "status-success";
  if (status === "RUNNING" || status === "STARTING") return "status-running";
  if (status === "FAILED" || status === "PARTIAL_FAILURE") return "status-failed";
  return "status-other";
}

function renderRuns() {
  const body = $("runsBody");
  if (!state.runs.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty-cell">No pipeline runs yet. Start a synthetic run above.</td></tr>';
    return;
  }
  body.innerHTML = state.runs.map((run) => {
    const selected = run.cohort_id === state.selectedRunId ? "selected" : "";
    return `<tr class="${selected}" data-cohort="${escapeHtml(run.cohort_id)}">
      <td><span class="run-id">${escapeHtml(run.cohort_id)}</span></td>
      <td>${escapeHtml(fmtDate(run.created_utc))}</td>
      <td><span class="sub-count">${run.succeeded || 0} / ${run.total_subjects || 0} completed</span></td>
      <td><span class="status-tag ${statusClass(run.status)}">${escapeHtml(run.status || "UNKNOWN")}</span></td>
      <td>↗</td>
    </tr>`;
  }).join("");
  body.querySelectorAll("tr[data-cohort]").forEach((row) => {
    row.addEventListener("click", () => selectCohort(row.dataset.cohort));
  });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function updateStats() {
  $("statRuns").textContent = state.runs.length;
  $("statSubjects").textContent = state.runs.reduce((sum, item) => sum + (item.succeeded || 0), 0);
}

async function refreshRuns() {
  try {
    const data = await api("/api/runs");
    state.runs = data.runs || [];
    renderRuns();
    updateStats();
    if (state.selectedRunId) await refreshSelectedCohort();
  } catch (error) {
    showToast(error.message);
  }
}

async function refreshSelectedCohort() {
  if (!state.selectedRunId) return;
  try {
    const cohort = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}`);
    const summary = cohort.summary || {};
    const done = cohort.subjects.filter((s) => s.status === "SUCCESS").length;
    const failed = cohort.subjects.filter((s) => s.status === "FAILED").length;
    const total = cohort.subjects.length;
    const terminal = cohort.status === "SUCCESS" || cohort.status === "FAILED" || cohort.status === "PARTIAL_FAILURE";
    $("runProgress").classList.remove("hidden");
    $("runProgressTitle").textContent = terminal ? `Run ${cohort.status.toLowerCase().replaceAll("_", " ")}` : "Pipeline executing";
    $("runProgressText").textContent = `${done} / ${total} subjects`;
    $("runProgressBar").style.width = `${total ? (done / total) * 100 : 0}%`;
    $("runProgressHint").textContent = failed
      ? `${failed} subject(s) failed. Inspect the runner log for the failure details.`
      : terminal ? "All subject checkpoints have been written." : "Subjects are processed sequentially; outputs appear as each subject finishes.";
    $("selectedRunTitle").textContent = cohort.cohort_run_id || cohort.cohort_id;
    $("selectedRunSub").textContent = `${cohort.status} · ${fmtDate(cohort.created_utc)} · STRICTLY_SYNTHETIC_ONLY`;
    const previous = state.selectedSubjectId;
    const selectable = cohort.subjects.filter((s) => s.status === "SUCCESS");
    $("subjectSelect").innerHTML = cohort.subjects.map((s) =>
      `<option value="${escapeHtml(s.subject_id)}" ${s.status !== "SUCCESS" ? "disabled" : ""}>${escapeHtml(s.subject_id)} · ${escapeHtml(s.status)}</option>`
    ).join("");
    if (selectable.length) {
      const chosen = selectable.some((s) => s.subject_id === previous) ? previous : selectable[0].subject_id;
      $("subjectSelect").value = chosen;
      if (state.selectedSubjectId !== chosen || !state._lastDetailRun || state._lastDetailRun !== state.selectedRunId) {
        state.selectedSubjectId = chosen;
        state._lastDetailRun = state.selectedRunId;
        await loadSubject(chosen);
      }
    } else {
      state.selectedSubjectId = null;
      clearSubjectView();
      $("resultsSection").classList.remove("hidden");
      $("subjectMetrics").querySelectorAll(".metric-card strong").forEach((el) => el.textContent = "—");
    }
  } catch (error) {
    showToast(error.message);
  }
}

function clearSubjectView() {
  $("qcFacts").innerHTML = '<div class="empty-cell">Subject QC will appear when a synthetic subject completes.</div>';
  $("qcImages").innerHTML = "";
  $("modelOutput").textContent = "Model output will appear when available.";
}

async function selectCohort(cohortId) {
  state.selectedRunId = cohortId;
  state.selectedSubjectId = null;
  state._lastDetailRun = null;
  renderRuns();
  $("resultsSection").classList.remove("hidden");
  clearSubjectView();
  await refreshSelectedCohort();
  $("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setMetric(id, value, suffix = "") {
  $(id).textContent = value === null || value === undefined ? "—" : `${value}${suffix}`;
}

async function loadSubject(subjectId) {
  if (!state.selectedRunId || state.loadingSubject) return;
  state.loadingSubject = true;
  state.selectedSubjectId = subjectId;
  try {
    const root = `/api/runs/${encodeURIComponent(state.selectedRunId)}/subjects/${encodeURIComponent(subjectId)}`;
    const detail = await api(root);
    setMetric("meanFd", Number(detail.mean_fd_mm).toFixed(3), " mm");
    setMetric("maxFd", Number(detail.max_fd_mm).toFixed(3), " mm");
    setMetric("peakMem", Number(detail.peak_rss_gb).toFixed(2), " GB");
    setMetric("runtime", Number(detail.runtime_seconds).toFixed(1), " s");

    const facts = [
      ["TR", detail.tr_seconds == null ? "—" : `${detail.tr_seconds} s`],
      ["BOLD volumes", detail.n_volumes ?? "—"],
      ["CC200 ROIs", detail.n_rois ?? "—"],
      ["Connectivity edges", (detail.n_connectivity_features ?? 0).toLocaleString()],
      ["Confound columns", detail.confound_columns ?? "—"],
      ["Confound rank", detail.confound_rank ?? "—"],
      ["Zero-variance ROIs", detail.zero_variance_rois ?? "—"],
      ["QC thresholds", "Pending approval"],
    ];
    $("qcFacts").innerHTML = facts.map(([label, value]) =>
      `<div class="qc-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`
    ).join("");
    drawLineChart($("fdChart"), detail.fd_series || [], "#59d4c5", "FD (mm)");

    const imageNames = [
      ["qc_registration_epi_to_t1w.png", "EPI → T1w registration"],
      ["qc_registration_t1w_to_mni.png", "T1w → MNI normalization"],
      ["qc_tissue_masks.png", "Tissue masks"],
      ["qc_parcellation_cc200.png", "CC200 parcellation"],
    ];
    $("qcImages").innerHTML = imageNames.map(([filename, label]) =>
      `<figure class="qc-image"><a href="${root}/artifact/${filename}" target="_blank" rel="noopener"><img loading="lazy" src="${root}/artifact/${filename}" alt="${escapeHtml(label)}"></a><figcaption>${escapeHtml(label)}</figcaption></figure>`
    ).join("");

    renderModel(detail.model_prediction);
    await loadTimeseries();
    await loadConnectivity();
  } catch (error) {
    showToast(error.message);
  } finally {
    state.loadingSubject = false;
  }
}

function renderModel(prediction) {
  const root = $("modelOutput");
  if (!prediction) {
    root.textContent = "No model inference artifact is available for this run. Configure the trusted model mount on the server to enable synthetic demonstration inference.";
    return;
  }
  root.innerHTML = `
    <strong>${escapeHtml(prediction.class_label ?? "Prediction available")}</strong>
    <div class="model-prob">
      <div>ADHD probability<strong>${(Number(prediction.adhd_probability) * 100).toFixed(1)}%</strong></div>
      <div>Control probability<strong>${(Number(prediction.control_probability) * 100).toFixed(1)}%</strong></div>
    </div>
    <small>${escapeHtml(prediction.evaluation_mode || "SYNTHETIC_DEMONSTRATION_ONLY")}</small>
    <p>${escapeHtml(prediction.diagnostic_disclaimer || "Not approved or validated for clinical diagnostic use.")}</p>`;
}

function drawLineChart(canvas, values, stroke, yLabel) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = Number(canvas.getAttribute("height")) || 150;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const left = 42, right = 12, top = 12, bottom = 25;
  const plotW = width - left - right, plotH = height - top - bottom;
  ctx.strokeStyle = "#293750";
  ctx.lineWidth = 1;
  ctx.font = "9px DM Mono, monospace";
  ctx.fillStyle = "#778aa7";
  for (let i = 0; i <= 3; i++) {
    const y = top + (plotH * i / 3);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
  }
  if (!values.length) return;
  let min = Math.min(...values), max = Math.max(...values);
  if (max - min < 1e-9) { min -= 0.5; max += 0.5; }
  const pad = (max - min) * 0.12;
  min -= pad; max += pad;
  ctx.fillText(max.toFixed(2), 3, top + 4);
  ctx.fillText(min.toFixed(2), 3, top + plotH);
  ctx.fillText("0", left, height - 7);
  ctx.fillText(String(values.length), width - 30, height - 7);
  ctx.strokeStyle = stroke; ctx.lineWidth = 1.8; ctx.beginPath();
  values.forEach((v, i) => {
    const x = left + (values.length === 1 ? 0 : (i / (values.length - 1)) * plotW);
    const y = top + (1 - (v - min) / (max - min)) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function loadTimeseries() {
  if (!state.selectedRunId || !state.selectedSubjectId) return;
  const roi = Number($("roiSelect").value || 1);
  const data = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/subjects/${encodeURIComponent(state.selectedSubjectId)}/timeseries?roi=${roi}`);
  drawLineChart($("timeChart"), data.values, "#8caeff", `ROI ${roi}`);
}

async function loadConnectivity() {
  if (!state.selectedRunId || !state.selectedSubjectId) return;
  const data = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/subjects/${encodeURIComponent(state.selectedSubjectId)}/connectivity`);
  drawHeatmap($("connectivityCanvas"), data.matrix);
  const vals = [];
  for (let i = 0; i < data.matrix.length; i++) for (let j = i + 1; j < data.matrix.length; j++) vals.push(data.matrix[i][j]);
  const min = Math.min(...vals), max = Math.max(...vals);
  $("matrixRange").textContent = `z range ${min.toFixed(2)} to ${max.toFixed(2)}`;
}

function drawHeatmap(canvas, matrix) {
  const ctx = canvas.getContext("2d");
  const n = matrix.length;
  const image = ctx.createImageData(n, n);
  let maxAbs = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) maxAbs = Math.max(maxAbs, Math.abs(matrix[i][j]));
  maxAbs = maxAbs || 1;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const value = Math.max(-1, Math.min(1, matrix[i][j] / maxAbs));
      const offset = (i * n + j) * 4;
      if (value < 0) {
        const t = value + 1;
        image.data[offset] = Math.round(45 + 175 * t);
        image.data[offset + 1] = Math.round(112 + 115 * t);
        image.data[offset + 2] = Math.round(190 + 65 * t);
      } else {
        const t = 1 - value;
        image.data[offset] = Math.round(224 + 22 * t);
        image.data[offset + 1] = Math.round(100 + 125 * t);
        image.data[offset + 2] = Math.round(112 + 110 * t);
      }
      image.data[offset + 3] = 255;
    }
  }
  const temp = document.createElement("canvas");
  temp.width = n; temp.height = n;
  temp.getContext("2d").putImageData(image, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(temp, 0, 0, canvas.width, canvas.height);
}

function renderIntakeResult(sessionId, result) {
  $("intakeResult").classList.remove("hidden");
  $("intakeResultTitle").textContent = \`\${result.kind === "report" ? "Report extraction" : "MRI scan validation"} · \${result.status || result.validation?.status || result.report_extraction?.status || "COMPLETE"}\`;
  $("intakeResultText").textContent = JSON.stringify({ session_id: sessionId, ...result }, null, 2);
  const links = $("intakeArtifactLinks");
  links.replaceChildren();
  const reports = result.report_extraction?.files || [];
  reports.filter((item) => item.text_artifact).forEach((item) => {
    const link = document.createElement("a");
    const filename = item.text_artifact.split("/").pop();
    link.href = \`/api/intake/sessions/\${encodeURIComponent(sessionId)}/artifacts/\${encodeURIComponent(filename)}\`;
    link.textContent = \`Download extracted text: \${item.filename}\`;
    link.target = "_blank";
    link.rel = "noopener";
    links.appendChild(link);
  });
  $("intakeResult").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function uploadIntake(kind, files, button) {
  if (!files || !files.length) {
    showToast("Choose at least one file first.");
    return;
  }
  const progress = $("intakeProgress");
  button.disabled = true;
  progress.classList.remove("hidden");
  let sessionId = null;
  try {
    progress.textContent = "Creating a secure local intake session…";
    const created = await api("/api/intake/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind }),
    });
    sessionId = created.session_id;
    for (let index = 0; index < files.length; index++) {
      const file = files[index];
      progress.textContent = \`Uploading \${index + 1} of \${files.length}: \${file.name} (\${(file.size / 1048576).toFixed(1)} MB)…\`;
      await api(\`/api/intake/sessions/\${encodeURIComponent(sessionId)}/files\`, {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Upload-Filename": encodeURIComponent(file.name),
        },
        body: file,
      });
    }
    progress.textContent = kind === "scan"
      ? "Checking NIfTI dimensions, BOLD metadata and required scan pairing…"
      : "Extracting report text with locally installed PDF/OCR tools…";
    const action = kind === "scan" ? "validate" : "process-reports";
    const result = await api(\`/api/intake/sessions/\${encodeURIComponent(sessionId)}/\${action}\`, {
      method: "POST",
    });
    renderIntakeResult(sessionId, result);
    progress.textContent = \`Intake complete · session \${sessionId}\`;
    showToast(kind === "scan" ? "MRI files uploaded and validation completed." : "Report upload and extraction completed.");
  } catch (error) {
    progress.textContent = sessionId
      ? \`Upload stopped. Intake session \${sessionId} was retained for review. \${error.message}\`
      : \`Could not create intake session. \${error.message}\`;
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}

async function startRun(event) {
  event.preventDefault();
  const button = $("runButton");
  button.disabled = true;
  button.textContent = "Starting synthetic pipeline…";
  const payload = {
    n_subjects: Number($("nSubjects").value),
    t_vol: Number($("tVol").value),
    seed: Number($("seed").value),
    atlas_mode: $("atlasMode").value,
    model_inference: $("modelInference").checked,
  };
  try {
    const started = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.selectedRunId = started.cohort_id;
    state.selectedSubjectId = null;
    state._lastDetailRun = null;
    $("resultsSection").classList.remove("hidden");
    clearSubjectView();
    await refreshRuns();
    $("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
    showToast("Synthetic MRI cohort started.");
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = '<span class="button-play">▶</span> Run MRI pipeline <span class="button-arrow">→</span>';
  }
}

async function init() {
  try {
    state.health = await api("/api/health");
    $("serverStatus").textContent = "Local server online";
    $("modelInference").disabled = !state.health.model_dir_available;
    $("modelAvailability").textContent = state.health.model_dir_available
      ? "Trusted frozen model mount detected. Predictions remain synthetic demonstrations."
      : "Not configured. Runs will produce MRI and connectivity outputs without model inference.";
    if (state.health.real_data_authorized !== false || state.health.execution_policy !== "STRICTLY_SYNTHETIC_ONLY") {
      $("serverStatus").textContent = "Safety configuration error";
      showToast("Unexpected server safety configuration. Run controls remain unavailable.");
      $("runButton").disabled = true;
    }
    await refreshRuns();
    state.refreshTimer = setInterval(refreshRuns, 3000);
  } catch (error) {
    $("serverStatus").textContent = "Offline";
    showToast(`Dashboard API unavailable: ${error.message}`);
  }
}

$("runForm").addEventListener("submit", startRun);
$("uploadZipButton").addEventListener("click", () => {
  const files = $("sessionZipInput").files;
  if (files.length !== 1) return showToast("Choose one complete MRI session ZIP.");
  uploadIntake("scan", files, $("uploadZipButton"));
});
$("uploadFilesButton").addEventListener("click", () => {
  uploadIntake("scan", $("scanFilesInput").files, $("uploadFilesButton"));
});
$("uploadReportsButton").addEventListener("click", () => {
  uploadIntake("report", $("reportFilesInput").files, $("uploadReportsButton"));
});
$("closeIntakeResult").addEventListener("click", () => $("intakeResult").classList.add("hidden"));
$("refreshRuns").addEventListener("click", refreshRuns);
$("subjectSelect").addEventListener("change", (event) => loadSubject(event.target.value));
$("roiSelect").addEventListener("change", () => loadTimeseries().catch((e) => showToast(e.message)));
$("viewRunLog").addEventListener("click", async () => {
  if (!state.selectedRunId) return;
  try {
    const result = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/log`);
    $("runLogText").textContent = result.available ? result.text : "No runner log has been written yet.";
    $("runLogPanel").classList.remove("hidden");
  } catch (error) {
    showToast(error.message);
  }
});
$("closeRunLog").addEventListener("click", () => $("runLogPanel").classList.add("hidden"));
for (let roi = 1; roi <= 200; roi++) {
  const option = document.createElement("option");
  option.value = String(roi); option.textContent = `ROI ${String(roi).padStart(3, "0")}`;
  if (roi === 1) option.selected = true;
  $("roiSelect").appendChild(option);
}
init();
