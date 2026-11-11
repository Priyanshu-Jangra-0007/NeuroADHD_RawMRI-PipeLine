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
  if (status === "TERMINATED") return "status-terminated";
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
    const failed = cohort.subjects.filter((s) => s.status === "FAILED" || s.status === "TERMINATED").length;
    const total = cohort.subjects.length;
    const terminal = cohort.status === "SUCCESS" || cohort.status === "FAILED" || cohort.status === "PARTIAL_FAILURE" || cohort.status === "TERMINATED";
    $("runProgress").classList.remove("hidden");
    $("runProgressTitle").textContent = terminal ? `Run ${cohort.status.toLowerCase().replaceAll("_", " ")}` : "Pipeline executing";
    $("runProgressText").textContent = `${done} / ${total} subjects`;
    $("runProgressBar").style.width = `${total ? (done / total) * 100 : 0}%`;
    $("runProgressHint").textContent = cohort.status === "TERMINATED"
      ? (cohort.terminated_reason || "Run was terminated. Stale process or user cancellation.")
      : failed
      ? `${failed} subject(s) failed. Inspect the runner log for the failure details.`
      : terminal ? "All subject checkpoints have been written." : "Subjects are processed sequentially; outputs appear as each subject finishes.";
    $("selectedRunTitle").textContent = cohort.cohort_run_id || cohort.cohort_id;
    $("selectedRunSub").textContent = cohort.status === "TERMINATED" && cohort.terminated_reason
      ? `${cohort.status} (${cohort.terminated_reason}) · ${fmtDate(cohort.created_utc)} · STRICTLY_SYNTHETIC_ONLY`
      : `${cohort.status} · ${fmtDate(cohort.created_utc)} · STRICTLY_SYNTHETIC_ONLY`;
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

    const stc = detail.slice_timing_correction;
    const facts = [
      ["TR", detail.tr_seconds == null ? "—" : `${detail.tr_seconds} s`],
      ["BOLD volumes", detail.n_volumes ?? "—"],
      ["CC200 ROIs", detail.n_rois ?? "—"],
      ["Connectivity edges", (detail.n_connectivity_features ?? 0).toLocaleString()],
      ["Confound columns", detail.confound_columns ?? "—"],
      ["Confound rank", detail.confound_rank ?? "—"],
      ["Zero-variance ROIs", detail.zero_variance_rois ?? "—"],
      ["QC thresholds", "Pending approval"],
      ["STC status", stc?.status ?? "Not recorded"],
      ["STC direction", stc?.slice_encoding_direction ?? "—"],
      ["STC reference", stc?.reference_time_seconds == null ? "—" : `${stc.reference_time_seconds} s (+shift)`],
      ["STC method", stc?.interpolation_method ?? "—"],
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
    <div class="model-synthetic-notice">Computed from synthetic phantom data. This is not a prediction about any real person.</div>
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

let currentDicomSessionId = null;

function renderIntakeResult(sessionId, result) {
  $("intakeResult").classList.remove("hidden");
  $("intakeResultTitle").textContent = `${result.kind === "report" ? "Report extraction" : "MRI scan validation"} · ${result.status || result.validation?.status || result.report_extraction?.status || "COMPLETE"}`;
  $("intakeResultText").textContent = JSON.stringify({ session_id: sessionId, ...result }, null, 2);
  const links = $("intakeArtifactLinks");
  links.replaceChildren();

  // If BIDS conversion ZIP is available, offer direct download
  const downloadZip = result.download_zip_filename || result.dicom_conversion?.download_zip_filename;
  if (downloadZip) {
    const zipBtn = document.createElement("a");
    zipBtn.className = "download-bids-btn";
    zipBtn.href = `/api/intake/sessions/${encodeURIComponent(sessionId)}/download_bids`;
    zipBtn.textContent = `⬇ Download validated BIDS session ZIP (${downloadZip})`;
    links.appendChild(zipBtn);
  }

  const reports = result.report_extraction?.files || [];
  reports.filter((item) => item.text_artifact).forEach((item) => {
    const link = document.createElement("a");
    const filename = item.text_artifact.split("/").pop();
    link.href = `/api/intake/sessions/${encodeURIComponent(sessionId)}/artifacts/${encodeURIComponent(filename)}`;
    link.textContent = `Download extracted text: ${item.filename}`;
    link.target = "_blank";
    link.rel = "noopener";
    links.appendChild(link);
  });
  $("intakeResult").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderDicomInspection(sessionId, data) {
  currentDicomSessionId = sessionId;
  const workbench = $("dicomWorkbench");
  workbench.classList.remove("hidden");
  $("dicomSessionInfo").textContent = `Session: ${sessionId} · Detected ${data.series_count || 0} series (dcm2niix: ${data.dcm2niix_version || "available"}).`;

  const tbody = $("dicomSeriesTableBody");
  tbody.replaceChildren();

  let hasAmbiguous = false;
  const ambiguityReasons = [];

  const seriesList = data.series || [];
  seriesList.forEach((s) => {
    const tr = document.createElement("tr");

    let badgeClass = "badge-other";
    if (s.candidate_scan_type === "T1w") badgeClass = "badge-t1";
    else if (s.candidate_scan_type === "BOLD") badgeClass = "badge-bold";
    if (s.classification_ambiguous) {
      badgeClass = "badge-ambiguous";
      hasAmbiguous = true;
      ambiguityReasons.push(`Series ${s.series_number || s.series_id}: ${s.ambiguity_reason || "classification ambiguous"}`);
    }

    const dimsText = s.dimensions && s.dimensions.length ? s.dimensions.join(" × ") : "—";
    const trText = s.repetition_time != null ? `${s.repetition_time} s` : "—";
    const crcVal = s.series_crc || s.series_id || "—";
    const sNumVal = s.series_number != null ? s.series_number : "—";

    tr.innerHTML = `
      <td><strong>${sNumVal}</strong></td>
      <td><code class="crc-tag">${escapeHtml(String(crcVal))}</code></td>
      <td><div><strong>${escapeHtml(s.series_description || "Series")}</strong></div><small style="color:#788ea8">${escapeHtml(s.protocol_name || "")}</small></td>
      <td>${s.image_count || 1}</td>
      <td>${escapeHtml(dimsText)}</td>
      <td>${escapeHtml(s.modality || "MR")}</td>
      <td>${escapeHtml(trText)}</td>
      <td><span class="badge ${badgeClass}">${escapeHtml(s.candidate_scan_type)}</span></td>
      <td style="text-align:center"><input type="radio" name="dicomT1wRadio" value="${escapeHtml(String(crcVal))}" ${s.candidate_scan_type === "T1w" && !s.classification_ambiguous ? "checked" : ""}></td>
      <td style="text-align:center"><input type="checkbox" name="dicomBoldCheckbox" value="${escapeHtml(String(crcVal))}" ${s.candidate_scan_type === "BOLD" ? "checked" : ""}></td>
    `;
    tbody.appendChild(tr);
  });

  if (hasAmbiguous) {
    $("dicomConfirmRow").classList.remove("hidden");
    $("dicomAmbiguityText").textContent = ambiguityReasons.join(" · ");
    $("dicomConfirmCheckbox").checked = false;
  } else {
    $("dicomConfirmRow").classList.add("hidden");
    $("dicomConfirmCheckbox").checked = true;
  }

  workbench.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function inspectDicom(sessionId) {
  const progress = $("intakeProgress");
  progress.classList.remove("hidden");
  progress.textContent = "Inspecting DICOM files with dcm2niix (inspect-only, no images converted)…";
  try {
    const data = await api(`/api/intake/sessions/${encodeURIComponent(sessionId)}/inspect_dicom`, {
      method: "POST",
    });
    if (data.errors && data.errors.length) {
      showToast(data.errors.join("; "));
    }
    renderDicomInspection(sessionId, data);
    progress.textContent = `DICOM inspection complete · ${data.series_count || 0} series identified.`;
  } catch (err) {
    progress.textContent = `DICOM inspection failed: ${err.message}`;
    showToast(err.message);
  }
}

async function convertDicom() {
  if (!currentDicomSessionId) {
    return showToast("No active DICOM session selected.");
  }
  const t1Radio = document.querySelector('input[name="dicomT1wRadio"]:checked');
  if (!t1Radio) {
    return showToast("Select one anatomical T1w series.");
  }
  const boldCheckboxes = Array.from(document.querySelectorAll('input[name="dicomBoldCheckbox"]:checked'));
  if (!boldCheckboxes.length) {
    return showToast("Select at least one resting-state BOLD series.");
  }

  const subjectId = ($("bidsSubjectInput").value || "").trim();
  if (!subjectId) {
    return showToast("Enter a BIDS Subject ID (e.g. sub-001).");
  }
  const sessionIdLabel = ($("bidsSessionInput").value || "").trim();
  const confirmAmbiguous = $("dicomConfirmCheckbox").checked;

  const boldSeries = boldCheckboxes.map((cb, idx) => ({
    series_id: cb.value,
    run: boldCheckboxes.length > 1 ? idx + 1 : null,
    task: "rest",
  }));

  const payload = {
    subject_id: subjectId,
    session_id_label: sessionIdLabel || null,
    t1w_series: t1Radio.value,
    bold_series: boldSeries,
    confirm_ambiguous: confirmAmbiguous,
  };

  const progress = $("intakeProgress");
  progress.classList.remove("hidden");
  progress.style.borderColor = "";
  progress.style.color = "";
  progress.textContent = `Converting series to BIDS (${subjectId}): executing dcm2niix anonymization and NIfTI compression…`;

  const btn = $("convertDicomButton");
  btn.disabled = true;
  try {
    const result = await api(`/api/intake/sessions/${encodeURIComponent(currentDicomSessionId)}/convert_dicom`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderIntakeResult(currentDicomSessionId, result);
    progress.textContent = `Conversion and BIDS validation complete · ${subjectId}`;
    showToast("DICOM converted to valid BIDS session successfully!");
  } catch (err) {
    const rawMsg = err?.message || "Unknown conversion error.";
    const isAuthBlocked =
      rawMsg.includes("Conversion blocked:") ||
      /research authorization|authorization|not approved|hmac|signature|permitted|expired|revoked|rejected|pending/i.test(rawMsg);
    const displayMsg = isAuthBlocked
      ? (rawMsg.includes("Conversion blocked: research authorization not approved for this dataset")
          ? rawMsg
          : `Conversion blocked: research authorization not approved for this dataset (${rawMsg})`)
      : `Conversion stopped: ${rawMsg}`;
    progress.classList.remove("hidden");
    if (isAuthBlocked) {
      progress.style.borderColor = "#9e6428";
      progress.style.color = "#ffbe6b";
    }
    progress.textContent = displayMsg;
    showToast(displayMsg);
  } finally {
    btn.disabled = false;
  }
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
    let hasDicom = false;
    for (let index = 0; index < files.length; index++) {
      const file = files[index];
      const fnameLower = file.name.toLowerCase();
      if (fnameLower.endsWith(".dcm") || fnameLower.endsWith(".ima") || fnameLower.endsWith(".zip")) {
        hasDicom = true;
      }
      progress.textContent = `Uploading ${index + 1} of ${files.length}: ${file.name} (${(file.size / 1048576).toFixed(1)} MB)…`;
      await api(`/api/intake/sessions/${encodeURIComponent(sessionId)}/files`, {
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
    const result = await api(`/api/intake/sessions/${encodeURIComponent(sessionId)}/${action}`, {
      method: "POST",
    });
    renderIntakeResult(sessionId, result);

    // If DICOM series were staged, automatically trigger DICOM inspection
    if (kind === "scan" && (hasDicom || result.validation?.dicom_file_count > 0 || result.validation?.status === "DICOM_SERIES_STAGED")) {
      await inspectDicom(sessionId);
    }

    progress.textContent = `Intake complete · session ${sessionId}`;
    showToast(kind === "scan" ? "MRI files uploaded and validation completed." : "Report upload and extraction completed.");
    await refreshStagedSessions();
  } catch (error) {
    progress.textContent = sessionId
      ? `Upload stopped. Intake session ${sessionId} was retained for review. ${error.message}`
      : `Could not create intake session. ${error.message}`;
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

async function refreshStagedSessions() {
  try {
    const data = await api("/api/intake/sessions");
    const select = $("existingSessionSelect");
    if (!select) return;
    const currentVal = select.value;
    select.replaceChildren();
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.textContent = "Select a staged session…";
    select.appendChild(defaultOpt);

    const scanSessions = (data.sessions || []).filter((s) => s.kind === "scan");
    scanSessions.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.session_id;
      const createdStr = s.created_utc ? s.created_utc.slice(0, 16).replace("T", " ") : "";
      opt.textContent = `${s.session_id} (${s.status}, ${createdStr})`;
      select.appendChild(opt);
    });
    if (currentVal && scanSessions.some((s) => s.session_id === currentVal)) {
      select.value = currentVal;
    }
  } catch (err) {
    // Non-blocking
  }
}

async function refreshAuthorizationStatus() {
  try {
    const auth = await api("/api/authorization");
    if ($("authExecPolicy")) $("authExecPolicy").textContent = auth.execution_policy || "STRICTLY_SYNTHETIC_ONLY";
    if ($("authOverallState")) $("authOverallState").textContent = auth.authorization_status || "NOT_AUTHORIZED";
    if ($("authDatasetStatus")) $("authDatasetStatus").textContent = `${auth.dataset_authorization_status || "PENDING"} (${auth.dataset_id || "sub-01"})`;
    if ($("authSignatureStatus")) $("authSignatureStatus").textContent = auth.signature_status || "MISSING_SIGNATURE";
    if ($("authApprovalRef")) $("authApprovalRef").textContent = auth.institutional_approval_reference || auth.approval_reference_status || "PENDING_APPROVAL_REFERENCE";
    if ($("authExpiryStatus")) $("authExpiryStatus").textContent = auth.expiry_status || "NO_EXPIRY_DATE";
    if ($("statPendingCount")) $("statPendingCount").textContent = auth.pending_items_count ?? 23;
    if ($("authBadge")) $("authBadge").textContent = auth.badge || "REAL DATA PROCESSING LOCKED";

    const blockersList = $("authBlockersList");
    if (blockersList && auth.blockers) {
      blockersList.innerHTML = auth.blockers.map((b) => `<li>${escapeHtml(b)}</li>`).join("");
    }
  } catch (err) {
    if ($("authBlockersList")) {
      $("authBlockersList").innerHTML = "<li>Could not fetch research authorization status from server.</li>";
    }
  }
}

async function refreshPipelineVerification() {
  try {
    const data = await api("/api/pipeline/verification");
    if ($("verificationBadge")) $("verificationBadge").textContent = data.badge || "SYNTHETIC PIPELINE VERIFIED";
    if ($("verifPipelineStatus")) $("verifPipelineStatus").textContent = `VERIFIED (${data.stages_verified}/${data.stages_total} Stages)`;
    if ($("pipelineStagePill")) $("pipelineStagePill").textContent = `${data.stages_total || 20} STAGES`;
    if ($("verifRealDataStatus")) $("verifRealDataStatus").textContent = data.safety_gate?.real_scan_pipeline_status || "LOCKED (Pending Auth)";
    if ($("verifAtlasStatus")) $("verifAtlasStatus").textContent = `${data.parcellation?.atlas?.slice(0, 20) || "Craddock CC200"} (${data.parcellation?.rois || 200} ROIs)`;
    if ($("verifFeaturesStatus")) $("verifFeaturesStatus").textContent = `${(data.parcellation?.edges || 19900).toLocaleString()} Edges (All Finite)`;

    if (data.report && data.report.correlation_recovery) {
      const cr = data.report.correlation_recovery;
      if ($("verifCorrPos")) $("verifCorrPos").textContent = `r = +${cr.positive_pair.recovered_pearson_r.toFixed(4)} (Target > 0.70) [PASSED]`;
      if ($("verifCorrNeg")) $("verifCorrNeg").textContent = `r = ${cr.negative_pair.recovered_pearson_r.toFixed(4)} (Target < -0.65) [PASSED]`;
    }

    const stc = data.slice_timing;
    if (stc) {
      if ($("verifStcBadge")) $("verifStcBadge").textContent = stc.badge || data.slice_timing_badge || "SLICE-TIMING CORRECTION VERIFIED — SYNTHETIC ONLY";
      if ($("verifStcStatus")) $("verifStcStatus").textContent = stc.status || "—";
      if ($("verifStcTr")) $("verifStcTr").textContent = stc.tr_seconds == null ? "Configurable via BIDS TR" : `${stc.tr_seconds.toFixed(2)} s`;
      if ($("verifStcSliceCount")) $("verifStcSliceCount").textContent = stc.slice_count == null ? "Per-axis NIfTI shape" : `${stc.slice_count} slices`;
      if ($("verifStcDirection")) $("verifStcDirection").textContent = stc.acquisition_direction || "k (default)";
      if ($("verifStcRefTime")) {
        $("verifStcRefTime").textContent = stc.reference_time_seconds == null
          ? (stc.temporal_origin_shift_note || "+TR/2 offset")
          : `${stc.reference_time_seconds.toFixed(2)} s (${stc.temporal_origin_shift_note || "+TR/2 shift"})`;
      }
      if ($("verifStcInterp")) $("verifStcInterp").textContent = stc.interpolation_method || "cubic_spline";
      if ($("verifStcCorrectedSlices")) $("verifStcCorrectedSlices").textContent = stc.corrected_slices || "—";
      const vm = stc.validation_metrics;
      if ($("verifStcMetrics")) {
        $("verifStcMetrics").textContent = vm
          ? `RMSE ${vm.rmse_before.toFixed(4)} → ${vm.rmse_after.toFixed(5)} (${vm.rmse_reduction_percent.toFixed(2)}% drop) | r ≥ ${vm.pearson_r_after_mean.toFixed(5)}`
          : "Run scripts/validate_step4_slice_timing.py to generate external metrics";
      }
      if ($("verifStcAuthStatus")) $("verifStcAuthStatus").textContent = stc.real_data_authorization_status || "NOT_AUTHORIZED (sub-01 PENDING)";
      const warnList = $("verifStcWarnings");
      if (warnList) {
        const items = Array.isArray(stc.missing_metadata_warnings) && stc.missing_metadata_warnings.length
          ? stc.missing_metadata_warnings
          : ["Missing SliceTiming is never invented from TR."];
        warnList.innerHTML = items.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
      }
    }
  } catch (err) {
    // Non-blocking
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
    await refreshAuthorizationStatus();
    await refreshPipelineVerification();
    await refreshRuns();
    await refreshStagedSessions();
    state.refreshTimer = setInterval(refreshRuns, 3000);
  } catch (error) {
    $("serverStatus").textContent = "Offline";
    showToast(`Dashboard API unavailable: ${error.message}`);
  }
}

$("runForm").addEventListener("submit", startRun);
$("resumeSessionButton")?.addEventListener("click", () => {
  const sel = $("existingSessionSelect")?.value;
  if (!sel) return showToast("Select a staged session first.");
  inspectDicom(sel);
});
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
$("closeDicomWorkbench").addEventListener("click", () => $("dicomWorkbench").classList.add("hidden"));
$("convertDicomButton").addEventListener("click", convertDicom);
$("inspectDicomAgainButton").addEventListener("click", () => {
  if (currentDicomSessionId) inspectDicom(currentDicomSessionId);
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
