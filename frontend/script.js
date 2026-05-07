let currentHouseId = null;
let currentHouseNumber = "";
let currentTaskCode = "";
let activeHistoryHouseId = null;
let historyHouseDirectory = [];
let historyHousePage = 1;
let taskPollTimer = null;
const previewUrls = new Map();

const BUNDLE_FIELDS = [
    { id: "long_view", label: "长景" },
    { id: "medium_view", label: "中景" },
    { id: "close_view", label: "近景" },
];
const MAX_BATCH_UPLOAD_BYTES = 100 * 1024 * 1024;
const TASK_PHASE_LABELS = {
    queued: "排队中 / Queued",
    running: "执行中 / Running",
    completed: "已完成 / Completed",
    failed: "失败 / Failed",
};
const CAPTURE_SCALE_LABELS = {
    long_view: "长景",
    medium_view: "中景",
    close_view: "近景",
};
const TARGET_SPEC_FALLBACKS = [
    {
        key: "square_120",
        label: "120 × 120 mm 大靶标",
        description: "适合检测区域较大、裂缝较长或需要更远识别距离的场景。",
        target_size_mm: [120, 120],
        qr_size_mm: 96,
        crack_size_category: "large",
        crack_size_label: "较大裂缝 / 大范围检测区",
    },
    {
        key: "square_80",
        label: "80 × 80 mm 标准靶标",
        description: "默认推荐，适合常规检测区域和大多数墙体裂缝测量。",
        target_size_mm: [80, 80],
        qr_size_mm: 60,
        crack_size_category: "standard",
        crack_size_label: "常规裂缝 / 默认推荐",
        recommended: true,
    },
    {
        key: "square_40",
        label: "40 × 40 mm 小靶标",
        description: "适合局部细裂缝和近距离拍摄，打印时建议保持原始比例。",
        target_size_mm: [40, 40],
        qr_size_mm: 28,
        crack_size_category: "fine",
        crack_size_label: "细裂缝 / 局部检测区",
    },
    {
        key: "square_20",
        label: "20 × 20 mm 微型靶标",
        description: "适合微细裂缝或张贴空间非常受限的部位，建议近距离正拍。",
        target_size_mm: [20, 20],
        qr_size_mm: 14,
        crack_size_category: "micro",
        crack_size_label: "微细裂缝 / 狭小张贴区",
    },
];
let availableTargetSpecs = TARGET_SPEC_FALLBACKS.slice();
let mediumTargetPrecheck = {
    status: "idle",
    passed: false,
    message: "等待选择中景图像后进行二维码与四角锚点预检。",
};

const VALIDATION_INPUT_SELECTOR = "input, select, textarea";
const VALIDATION_CARD_SELECTOR = ".field-card, .upload-config-card, .upload-slot, .field-addon";
const FORM_SUMMARY_CONFIG = {
    infoForm: {
        summaryId: "infoFormSummary",
        completeText: (total) => `已完成 ${total}/${total} 项必填内容，可以直接提交房屋档案。`,
        pendingText: (missing, total) => `还有 ${missing}/${total} 项必填内容待完成。`,
    },
    targetForm: {
        summaryId: "targetFormSummary",
        completeText: (total) => `已完成 ${total}/${total} 项关键参数，可以直接生成靶标。`,
        pendingText: (missing, total) => `还有 ${missing}/${total} 项关键参数待填写。`,
    },
    uploadForm: {
        summaryId: "uploadFormSummary",
        completeText: (total) => `已完成 ${total}/${total} 个必传项，可以启动分析。`,
        pendingText: (missing, total) => `还差 ${missing}/${total} 个必传项未补齐。`,
    },
};

function setStatus(elementId, kind, text) {
    const element = document.getElementById(elementId);
    if (!element.dataset.baseClass) {
        element.dataset.baseClass = Array.from(element.classList)
            .filter((className) => !className.startsWith("status"))
            .join(" ");
    }
    element.className = `${element.dataset.baseClass ? `${element.dataset.baseClass} ` : ""}status status-${kind}`;
    element.textContent = text;
}

function setGlobalStatus(kind, text) {
    setStatus("globalStatus", kind, text);
}

function setButtonState(buttonId, disabled, text) {
    const button = document.getElementById(buttonId);
    button.disabled = disabled;
    button.textContent = text;
}

function setVisible(elementId, visible) {
    document.getElementById(elementId).classList.toggle("hidden", !visible);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function isElementVisible(element) {
    if (!element) {
        return false;
    }

    if (element.closest(".hidden, .hidden-field")) {
        return false;
    }

    const style = window.getComputedStyle(element);
    return style.display !== "none" && style.visibility !== "hidden";
}

function isValidationCandidate(element) {
    return Boolean(element) && !element.disabled && isElementVisible(element);
}

function isRequiredField(element) {
    return Boolean(element?.required) && isValidationCandidate(element);
}

function isFieldFilled(element) {
    if (!isRequiredField(element)) {
        return true;
    }

    if (element.type === "file") {
        return Boolean(element.files?.length);
    }

    return element.value.trim() !== "";
}

function getFieldCard(element) {
    return element.closest(VALIDATION_CARD_SELECTOR);
}

function updateFieldCardState(card) {
    if (!card) {
        return;
    }

    const requiredFields = Array.from(card.querySelectorAll(VALIDATION_INPUT_SELECTOR)).filter(isRequiredField);
    const hasRequiredField = requiredFields.length > 0;
    const hasMissingField = hasRequiredField && requiredFields.some((field) => !isFieldFilled(field));

    card.classList.toggle("card-required", hasRequiredField);
    card.classList.toggle("card-missing", hasMissingField);
    card.classList.toggle("card-complete", hasRequiredField && !hasMissingField);
}

function updateFieldValidationState(element) {
    if (!element) {
        return;
    }

    const active = isRequiredField(element);
    const missing = active && !isFieldFilled(element);

    element.classList.toggle("input-required", active);
    element.classList.toggle("input-missing", missing);
    element.classList.toggle("input-complete", active && !missing);

    if (active) {
        element.setAttribute("aria-invalid", missing ? "true" : "false");
    } else {
        element.removeAttribute("aria-invalid");
    }

    updateFieldCardState(getFieldCard(element));
}

function updateFormSummary(formId) {
    const form = document.getElementById(formId);
    const config = FORM_SUMMARY_CONFIG[formId];
    const summary = config ? document.getElementById(config.summaryId) : null;
    if (!form || !config || !summary) {
        return;
    }

    const requiredFields = Array.from(form.querySelectorAll(VALIDATION_INPUT_SELECTOR)).filter(isRequiredField);
    const total = requiredFields.length;
    const missing = requiredFields.filter((field) => !isFieldFilled(field)).length;

    summary.textContent = missing === 0 ? config.completeText(total) : config.pendingText(missing, total);
    summary.classList.toggle("workspace-summary-count-pending", missing > 0);
    summary.classList.toggle("workspace-summary-count-complete", total > 0 && missing === 0);
}

function updateFormValidationState(formId) {
    const form = document.getElementById(formId);
    if (!form) {
        return;
    }

    Array.from(form.querySelectorAll(VALIDATION_INPUT_SELECTOR)).forEach(updateFieldValidationState);
    Array.from(form.querySelectorAll(VALIDATION_CARD_SELECTOR)).forEach(updateFieldCardState);
    updateFormSummary(formId);
}

function focusFirstMissingField(formId) {
    const form = document.getElementById(formId);
    if (!form) {
        return null;
    }

    const firstMissingField = Array.from(form.querySelectorAll(VALIDATION_INPUT_SELECTOR))
        .filter(isRequiredField)
        .find((field) => !isFieldFilled(field));

    if (!firstMissingField) {
        return null;
    }

    updateFieldValidationState(firstMissingField);
    firstMissingField.focus({ preventScroll: true });
    firstMissingField.scrollIntoView({ behavior: "smooth", block: "center" });
    return firstMissingField;
}

function initializeFormValidation(formId) {
    const form = document.getElementById(formId);
    if (!form) {
        return;
    }

    Array.from(form.querySelectorAll(VALIDATION_INPUT_SELECTOR)).forEach((field) => {
        field.addEventListener("input", () => updateFormValidationState(formId));
        field.addEventListener("change", () => updateFormValidationState(formId));
        field.addEventListener("blur", () => updateFieldValidationState(field));
    });

    updateFormValidationState(formId);
}

function initializeWorkspaceValidation() {
    Object.keys(FORM_SUMMARY_CONFIG).forEach(initializeFormValidation);
}

function toggleOtherInput(fieldId, { focus = true } = {}) {
    const select = document.getElementById(fieldId);
    const otherInput = document.getElementById(`${fieldId}_other`);
    const otherWrap = document.getElementById(`${fieldId}_other_wrap`);
    const useOther = select.value === "其他";

    if (otherWrap) {
        otherWrap.classList.toggle("hidden-field", !useOther);
    } else {
        otherInput.classList.toggle("hidden-field", !useOther);
    }

    otherInput.disabled = !useOther;
    otherInput.required = useOther;

    if (!useOther) {
        otherInput.value = "";
    } else if (focus) {
        window.requestAnimationFrame(() => otherInput.focus());
    }

    const form = select.closest("form");
    if (form?.id) {
        updateFormValidationState(form.id);
    }
}

function getFieldValue(fieldId) {
    const select = document.getElementById(fieldId);
    const otherInput = document.getElementById(`${fieldId}_other`);
    if (select.value === "其他") {
        return otherInput.value.trim();
    }
    return select.value.trim();
}

function updateHouseBinding(houseId, houseNumber) {
    currentHouseId = houseId;
    currentHouseNumber = houseNumber;
    document.getElementById("houseBindingText").textContent = houseId
        ? `当前已绑定房屋：ID ${houseId} / 编号 ${houseNumber}`
        : "尚未绑定房屋信息。";
    syncTargetHouseNumber(Boolean(houseNumber));
}

function clearPreviewUrl(fieldId) {
    const url = previewUrls.get(fieldId);
    if (url) {
        URL.revokeObjectURL(url);
        previewUrls.delete(fieldId);
    }
}

function resetBundleQualityPanel() {
    document.getElementById("bundleQualitySummary").textContent = "等待上传后进行检测。";
    document.getElementById("bundleQualityList").innerHTML = "";
    setVisible("bundleQualityPanel", false);
    setVisible("overrideUploadButton", false);
}

function formatTaskPhase(status) {
    return TASK_PHASE_LABELS[status] || status || "-";
}

function localizeTaskMessage(message) {
    if (!message) {
        return "";
    }

    const knownMessages = {
        "Task queued for analysis.": "任务已排队，等待分析 / Task queued for analysis.",
        "Analysis is running.": "任务分析中 / Analysis is running.",
        "Analysis completed.": "分析完成 / Analysis completed.",
        "Analysis completed with errors.": "分析完成，但存在错误 / Analysis completed with errors.",
        "Analysis failed.": "分析失败 / Analysis failed.",
        "Task interrupted before completion.": "任务在完成前中断 / Task interrupted before completion.",
        "Detection task not found": "未找到检测任务 / Detection task not found",
        "No task status available.": "暂无任务状态 / No task status available.",
        "Task queued.": "任务已创建，等待执行 / Task queued.",
        "Task is in progress.": "任务执行中 / Task is in progress.",
        "Task finished successfully.": "任务已完成 / Task finished successfully.",
        "Task finished with partial errors.": "任务已完成，但存在部分错误 / Task finished with partial errors.",
        "Task failed.": "任务失败 / Task failed.",
    };

    if (knownMessages[message]) {
        return knownMessages[message];
    }

    const processingMatch = message.match(/^Processing ([a-z_]+) \((\d+)\/(\d+)\)\.$/);
    if (processingMatch) {
        const [, captureScale, index, total] = processingMatch;
        const captureScaleLabel = CAPTURE_SCALE_LABELS[captureScale] || captureScale;
        return `正在处理 ${captureScaleLabel}（${index}/${total}） / Processing ${captureScale} (${index}/${total}).`;
    }

    return message;
}

function stopTaskPolling() {
    if (taskPollTimer !== null) {
        window.clearInterval(taskPollTimer);
        taskPollTimer = null;
    }
}

function getTaskStatusClass(task) {
    if (task?.status === "completed") {
        return task.has_errors ? "task-status-warning" : "task-status-success";
    }
    if (task?.status === "failed") {
        return "task-status-error";
    }
    if (task?.status === "queued" || task?.status === "running") {
        return "task-status-running";
    }
    return "task-status-idle";
}

function getUploadedTimeText(item, fallback = "未知时间") {
    return item?.uploaded_at_display || item?.uploaded_at || item?.created_at_display || item?.created_at || fallback;
}

function getGeneratedTimeText(item, fallback = "未知时间") {
    return item?.generated_at_display || item?.created_at_display || item?.generated_at || item?.created_at || fallback;
}

function renderDetectionTaskStatus(task) {
    const panel = document.getElementById("taskStatusPanel");
    panel.className = `task-status-panel ${getTaskStatusClass(task)}`;
    document.getElementById("taskStatusSummary").textContent =
        localizeTaskMessage(task?.message) || "暂无任务状态 / No task status available.";
    document.getElementById("taskStatusCode").textContent = task?.task_code || "-";
    document.getElementById("taskStatusBundle").textContent = task?.bundle_code || "-";
    document.getElementById("taskStatusProgress").textContent =
        task?.total_items ? `${task.processed_items || 0}/${task.total_items} (${task.progress_percent || 0}%)` : "-";
    document.getElementById("taskStatusPhase").textContent = formatTaskPhase(task?.status);
    document.getElementById("taskStatusTime").textContent =
        task?.completed_at_display || task?.started_at_display || task?.uploaded_at_display || "-";
    document.getElementById("taskStatusError").textContent = task?.error_detail || "";
    setVisible("taskStatusPanel", true);
}

async function fetchDetectionTask(taskCode, { silent = false } = {}) {
    if (!taskCode) {
        if (!silent) {
            setGlobalStatus("error", "请输入任务编号 / Task code is required.");
        }
        return null;
    }

    try {
        const response = await fetch(`/tasks/${encodeURIComponent(taskCode)}`);
        const result = await parseResponsePayload(response);
        if (!response.ok) {
            if (!silent) {
                setGlobalStatus("error", localizeTaskMessage(result.message) || "加载任务状态失败 / Failed to load task status.");
            }
            return null;
        }

        currentTaskCode = result.task_code || taskCode;
        document.getElementById("taskLookupCode").value = currentTaskCode;
        renderDetectionTaskStatus(result);

        if (Array.isArray(result.quality_results) && result.quality_results.length) {
            renderBundleQualityResults(result.quality_results, false);
        }

        if (result.status === "completed") {
            stopTaskPolling();
            if (Array.isArray(result.results) && result.results.length) {
                renderBatchUploadResult(result);
            }
            if (currentHouseId) {
                await loadHouseDetections(currentHouseId);
            }
            await loadSurveySummary();
            setStatus(
                "uploadStatus",
                result.has_errors ? "warning" : "success",
                result.has_errors
                    ? "任务已完成，但存在部分错误 / Task finished with partial errors."
                    : "任务已完成 / Task finished successfully."
            );
            setGlobalStatus(
                result.has_errors ? "warning" : "success",
                result.has_errors
                    ? `任务 ${currentTaskCode} 已完成，但存在部分错误 / Task ${currentTaskCode} finished with partial errors.`
                    : `任务 ${currentTaskCode} 已完成 / Task ${currentTaskCode} finished successfully.`
            );
        } else if (result.status === "failed") {
            stopTaskPolling();
            if (Array.isArray(result.results) && result.results.length) {
                renderBatchUploadResult(result);
            }
            setStatus("uploadStatus", "error", localizeTaskMessage(result.message) || "任务失败 / Task failed.");
            setGlobalStatus(
                "error",
                localizeTaskMessage(result.message) || `任务 ${currentTaskCode} 失败 / Task ${currentTaskCode} failed.`
            );
        } else if (!silent) {
            setStatus("uploadStatus", "loading", localizeTaskMessage(result.message) || "任务执行中 / Task is in progress.");
            setGlobalStatus("loading", `任务 ${currentTaskCode} 当前状态：${formatTaskPhase(result.status)}`);
        }

        return result;
    } catch (error) {
        if (!silent) {
            setGlobalStatus("error", `加载任务状态失败 / Failed to load task status: ${error}`);
        }
        return null;
    }
}

function startTaskPolling(taskCode) {
    stopTaskPolling();
    currentTaskCode = taskCode;
    document.getElementById("taskLookupCode").value = taskCode;
    taskPollTimer = window.setInterval(() => {
        fetchDetectionTask(taskCode, { silent: true });
    }, 2000);
}

async function queryDetectionTask() {
    const taskCode = document.getElementById("taskLookupCode").value.trim();
    if (!taskCode) {
        setGlobalStatus("error", "请先输入任务编号 / Enter a task code first.");
        return;
    }

    stopTaskPolling();
    const task = await fetchDetectionTask(taskCode);
    if (task && (task.status === "queued" || task.status === "running")) {
        startTaskPolling(task.task_code || taskCode);
    }
}

function renderMediumTargetPrecheck(kind, text) {
    const panel = document.getElementById("targetPrecheckPanel");
    const summary = document.getElementById("targetPrecheckSummary");
    panel.className = `target-precheck-panel target-precheck-${kind}`;
    summary.textContent = text;
    setVisible("targetPrecheckPanel", true);
}

function resetMediumTargetPrecheck() {
    mediumTargetPrecheck = {
        status: "idle",
        passed: false,
        message: "等待选择中景图像后进行二维码与四角锚点预检。",
    };
    renderMediumTargetPrecheck("idle", mediumTargetPrecheck.message);
}

async function runMediumTargetPrecheck(file) {
    if (!file) {
        resetMediumTargetPrecheck();
        return;
    }

    mediumTargetPrecheck = {
        status: "loading",
        passed: false,
        message: "正在检查中景图中的二维码与四角锚点...",
    };
    renderMediumTargetPrecheck("loading", mediumTargetPrecheck.message);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/targets/precheck", {
            method: "POST",
            body: formData,
        });
        const result = await parseResponsePayload(response);
        const passed = Boolean(result.passed);
        mediumTargetPrecheck = {
            status: passed ? "success" : "error",
            passed,
            message: result.message || (passed ? "中景二维码预检通过。" : "中景二维码预检未通过。"),
        };
        renderMediumTargetPrecheck(mediumTargetPrecheck.status, mediumTargetPrecheck.message);
    } catch (error) {
        mediumTargetPrecheck = {
            status: "error",
            passed: false,
            message: `中景二维码预检失败：${error}`,
        };
        renderMediumTargetPrecheck("error", mediumTargetPrecheck.message);
    }
}

function syncTargetHouseNumber(force = false) {
    const targetInput = document.getElementById("target_house_number");
    const sourceInput = document.getElementById("house_number");
    if (!targetInput || !sourceInput) {
        return;
    }

    const sourceValue = (currentHouseNumber || sourceInput.value || "").trim();
    if (force || !targetInput.value.trim()) {
        targetInput.value = sourceValue;
    }

    updateFormValidationState("targetForm");
}

function setDefaultTargetInspectionTime() {
    const input = document.getElementById("target_inspection_at");
    if (!input || input.value) {
        return;
    }

    const now = new Date();
    const offset = now.getTimezoneOffset();
    const localTime = new Date(now.getTime() - offset * 60 * 1000);
    input.value = localTime.toISOString().slice(0, 16);
}

function getAvailableTargetSpecs() {
    return Array.isArray(availableTargetSpecs) && availableTargetSpecs.length ? availableTargetSpecs : TARGET_SPEC_FALLBACKS;
}

function findTargetSpecByKey(specKey) {
    return getAvailableTargetSpecs().find((spec) => spec.key === specKey) || null;
}

function formatTargetSize(spec = {}) {
    if (Array.isArray(spec.target_size_mm) && spec.target_size_mm.length === 2) {
        return `${spec.target_size_mm[0]} × ${spec.target_size_mm[1]} mm`;
    }
    if (spec.frame_size_mm) {
        return `${spec.frame_size_mm} × ${spec.frame_size_mm} mm`;
    }
    return "-";
}

function getRecommendedTargetSpec(specs, crackSizeCategory) {
    if (crackSizeCategory) {
        const matched = specs.filter((spec) => spec.crack_size_category === crackSizeCategory);
        if (matched.length) {
            return matched.find((spec) => spec.recommended) || matched[0];
        }
    }
    return specs.find((spec) => spec.recommended) || specs[0] || null;
}

function updateTargetSpecGuidance() {
    const crackSizeSelect = document.getElementById("target_crack_size");
    const specSelect = document.getElementById("target_spec");
    const crackHint = document.getElementById("target_crack_size_hint");
    const specHint = document.getElementById("target_spec_hint");
    if (!specSelect) {
        return;
    }

    const specs = getAvailableTargetSpecs();
    const selectedSpec = findTargetSpecByKey(specSelect.value) || specs[0] || null;
    const crackSizeValue = crackSizeSelect ? crackSizeSelect.value : "";
    const recommendedSpec = getRecommendedTargetSpec(specs, crackSizeValue);

    if (crackHint && crackSizeSelect) {
        const crackScaleText = crackSizeSelect.selectedOptions[0]?.textContent?.trim() || "常规裂缝 / 默认推荐";
        crackHint.textContent = recommendedSpec
            ? `${crackScaleText}。建议：${recommendedSpec.label}。`
            : `${crackScaleText}。`;
    }

    if (specHint) {
        specHint.textContent = selectedSpec
            ? `${selectedSpec.label}，尺寸 ${formatTargetSize(selectedSpec)}，二维码边长 ${selectedSpec.qr_size_mm ?? "-"} mm。`
            : "请选择靶标规格。";
    }
}

function syncTargetSpecByCrackSize(force = false) {
    const select = document.getElementById("target_spec");
    if (!select) {
        return;
    }

    const specs = getAvailableTargetSpecs();
    const crackSizeValue = document.getElementById("target_crack_size")?.value || "";
    const currentSpec = findTargetSpecByKey(select.value);
    const recommendedSpec = getRecommendedTargetSpec(specs, crackSizeValue);
    const shouldReplace = force || !currentSpec;

    if (recommendedSpec && shouldReplace) {
        select.value = recommendedSpec.key;
    }

    updateTargetSpecGuidance();
    updateFormValidationState("targetForm");
}

function renderTargetSpecOptions(specs) {
    const select = document.getElementById("target_spec");
    if (!select) {
        return;
    }

    const normalizedSpecs = Array.isArray(specs) && specs.length ? specs : TARGET_SPEC_FALLBACKS;
    const previousValue = select.value;
    availableTargetSpecs = normalizedSpecs.slice();
    select.innerHTML = normalizedSpecs
        .map((spec) => {
            const badge = spec.recommended ? "（推荐）" : "";
            return `<option value="${escapeHtml(spec.key)}">${escapeHtml(spec.label)}${badge}</option>`;
        })
        .join("");

    if (previousValue && normalizedSpecs.some((spec) => spec.key === previousValue)) {
        select.value = previousValue;
    }

    syncTargetSpecByCrackSize(!previousValue);
    updateFormValidationState("targetForm");
}

async function loadTargetSpecs() {
    try {
        const response = await fetch("/targets/specs");
        const result = await parseResponsePayload(response);
        if (!response.ok || !Array.isArray(result.specs) || !result.specs.length) {
            renderTargetSpecOptions(TARGET_SPEC_FALLBACKS);
            return;
        }
        renderTargetSpecOptions(result.specs);
    } catch (_error) {
        renderTargetSpecOptions(TARGET_SPEC_FALLBACKS);
    }
}

function renderTargetResult(payload) {
    const container = document.getElementById("targetResult");
    const files = payload.files || {};
    const spec = payload.spec || {};
    const metaCards = [
        { label: "靶标编号", value: payload.target_id || "-" },
        { label: "房屋编号", value: payload.metadata?.house_number || payload.profile?.house_number || "-" },
        { label: "靶标规格", value: spec.label || "-" },
        { label: "靶标尺寸", value: formatTargetSize(spec) },
        { label: "二维码边长", value: spec.qr_size_mm ? `${spec.qr_size_mm} mm` : "-" },
        { label: "适用裂缝", value: spec.crack_size_label || "-" },
        { label: "检测时间", value: payload.inspection_at_display || payload.profile?.inspection_at_display || "-" },
        { label: "报告编号", value: payload.report_reference || payload.profile?.report_reference || "-" },
    ];
    const scanUrl = payload.scan_url || payload.detail_url || payload.profile?.detail_url || "";
    const scanWarning = payload.scan_url_warning || "";
    const summaryFacts = [
        { label: "靶标规格", value: spec.label || "-" },
        { label: "打印尺寸", value: formatTargetSize(spec) },
        { label: "二维码边长", value: spec.qr_size_mm ? `${spec.qr_size_mm} mm` : "-" },
    ];

    container.innerHTML = `
        <div class="target-preview-card">
            <div class="target-preview-head">
                <div>
                    <span class="hero-side-kicker">打印预览</span>
                    <strong>靶标版面</strong>
                </div>
            </div>
            <div class="target-preview-frame">
                ${
                    payload.preview_url
                        ? `<img src="${escapeHtml(payload.preview_url)}" alt="二维码靶标预览">`
                        : `<p class="target-preview-placeholder">暂无靶标预览</p>`
                }
            </div>
        </div>
        <div class="target-meta-card">
            <div class="target-meta-head">
                <div>
                    <span class="hero-side-kicker">关联信息</span>
                    <strong>靶标参数与档案信息</strong>
                </div>
                <span class="meta-chip">可打印 / 可扫码</span>
            </div>
            <div class="target-summary-strip">
                ${summaryFacts
                    .map(
                        (item) => `
                            <article>
                                <span>${escapeHtml(item.label)}</span>
                                <strong>${escapeHtml(item.value)}</strong>
                            </article>
                        `
                    )
                    .join("")}
            </div>
            <div class="target-meta-grid">
                ${metaCards
                    .map(
                        (item) => `
                            <article>
                                <span>${escapeHtml(item.label)}</span>
                                <strong>${escapeHtml(item.value)}</strong>
                            </article>
                        `
                    )
                    .join("")}
            </div>
            <div class="target-meta-note">
                <p class="target-meta-text">编码内容已写入二维码与 JSON 清单。建议优先下载 PDF 打印；扫码后可直接查看该靶标的档案信息和报告入口。</p>
                ${scanUrl ? `<p class="target-meta-text target-scan-url"><strong>扫码详情页：</strong>${escapeHtml(scanUrl)}</p>` : ""}
                ${scanWarning ? `<p class="target-meta-text target-scan-warning">${escapeHtml(scanWarning)}</p>` : ""}
            </div>
            <div class="target-link-row">
                ${payload.detail_url ? `<a class="button-link" href="${escapeHtml(payload.detail_url)}" target="_blank" rel="noopener noreferrer">打开详情页</a>` : ""}
                ${files.pdf_url ? `<a class="button-link" href="${escapeHtml(files.pdf_url)}" target="_blank" rel="noopener noreferrer">下载 PDF</a>` : ""}
                ${files.png_url ? `<a class="button-link" href="${escapeHtml(files.png_url)}" target="_blank" rel="noopener noreferrer">下载 PNG</a>` : ""}
                ${files.manifest_url ? `<a class="button-link" href="${escapeHtml(files.manifest_url)}" target="_blank" rel="noopener noreferrer">查看 JSON</a>` : ""}
            </div>
        </div>
    `;
    setVisible("targetResult", true);
}

function arrangeSplitPreviewWorkspace() {
    const uploadPreviewStack = document.getElementById("uploadPreviewStack");

    if (uploadPreviewStack) {
        ["long_view_preview_card", "medium_view_preview_card", "close_view_preview_card"].forEach((cardId) => {
            const card = document.getElementById(cardId);
            if (card && card.parentElement !== uploadPreviewStack) {
                uploadPreviewStack.appendChild(card);
            }
        });
    }
}

async function generateTarget() {
    updateFormValidationState("targetForm");
    const houseNumber = document.getElementById("target_house_number").value.trim();
    const inspectionRegion = document.getElementById("target_region").value.trim();
    const sceneType = document.getElementById("target_scene_type").value;
    const inspectionAt = document.getElementById("target_inspection_at").value;
    const reportReference = document.getElementById("target_report_reference").value.trim();
    const notes = document.getElementById("target_notes").value.trim();
    const specKey = document.getElementById("target_spec").value;

    if (!houseNumber || !inspectionRegion || !sceneType || !inspectionAt || !specKey) {
        focusFirstMissingField("targetForm");
        setStatus("targetStatus", "error", "请完整填写房屋编号、检测时间、检测区域、场景类型和靶标规格。");
        setGlobalStatus("error", "二维码靶标生成参数不完整。");
        return;
    }

    setButtonState("generateTargetButton", true, "生成中...");
    setStatus("targetStatus", "loading", "正在生成二维码靶标...");
    setGlobalStatus("loading", "正在生成二维码靶标，请稍候。");

    try {
        const response = await fetch("/targets/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                house_id: currentHouseId,
                house_number: houseNumber,
                inspection_region: inspectionRegion,
                scene_type: sceneType,
                inspection_at: inspectionAt,
                report_reference: reportReference,
                notes,
                spec_key: specKey,
            }),
        });
        const result = await parseResponsePayload(response);

        if (!response.ok) {
            setStatus("targetStatus", "error", result.message || "二维码靶标生成失败。");
            setGlobalStatus("error", result.message || "二维码靶标生成失败。");
            return;
        }

        renderTargetResult(result);
        setStatus("targetStatus", "success", "二维码靶标已生成，可直接下载打印并扫码查看详情。");
        setGlobalStatus("success", "二维码靶标已生成。建议先打印并张贴靶标，再进行中景与近景拍摄。");
    } catch (error) {
        setStatus("targetStatus", "error", `二维码靶标生成失败：${error}`);
        setGlobalStatus("error", `二维码靶标生成失败：${error}`);
    } finally {
        setButtonState("generateTargetButton", false, "生成二维码靶标");
    }
}

function getQualityBadgeClass(status) {
    if (status === "good") {
        return "quality-good";
    }
    if (status === "warning") {
        return "quality-warning";
    }
    if (status === "reject") {
        return "quality-reject";
    }
    return "quality-idle";
}

function formatBytes(bytes) {
    if (bytes >= 1024 * 1024) {
        return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    }
    return `${(bytes / 1024).toFixed(0)} KB`;
}

function getBundleSelectionSummary() {
    let totalBytes = 0;
    const selected = [];

    BUNDLE_FIELDS.forEach((field) => {
        const input = document.getElementById(field.id);
        if (input.files.length) {
            const file = input.files[0];
            totalBytes += file.size;
            selected.push({
                label: field.label,
                size: file.size,
                name: file.name,
            });
        }
    });

    return {
        totalBytes,
        selected,
        isComplete: selected.length === BUNDLE_FIELDS.length,
        exceedsLimit: totalBytes > MAX_BATCH_UPLOAD_BYTES,
    };
}

function updateBundleSizeSummary() {
    const panel = document.getElementById("bundleSizePanel");
    const summary = document.getElementById("bundleSizeSummary");
    const { totalBytes, selected, isComplete, exceedsLimit } = getBundleSelectionSummary();

    panel.className = "bundle-size-panel";
    if (!selected.length) {
        summary.textContent = "尚未选择完整的长景 / 中景 / 近景图像。";
        return;
    }

    const detail = selected.map((item) => `${item.label} ${formatBytes(item.size)}`).join("，");
    const totalText = `当前已选择 ${selected.length}/3 张图，总大小 ${formatBytes(totalBytes)}。${detail}。`;

    if (exceedsLimit) {
        panel.classList.add("bundle-size-error");
        summary.textContent = `${totalText} 已超过 100MB 上限，请压缩图片或更换较小文件后再上传。`;
        return;
    }

    if (isComplete) {
        const remaining = MAX_BATCH_UPLOAD_BYTES - totalBytes;
        if (remaining < 15 * 1024 * 1024) {
            panel.classList.add("bundle-size-warning");
            summary.textContent = `${totalText} 已接近 100MB 上限，剩余可用空间 ${formatBytes(remaining)}。`;
            return;
        }

        panel.classList.add("bundle-size-good");
        summary.textContent = `${totalText} 在 100MB 限制内，可以上传。`;
        return;
    }

    summary.textContent = `${totalText} 还需要补齐剩余图像后才能上传。`;
}

async function parseResponsePayload(response) {
    const rawText = await response.text();
    if (!rawText) {
        return {};
    }

    try {
        return JSON.parse(rawText);
    } catch (_error) {
        if (rawText.trim().startsWith("<!doctype") || rawText.trim().startsWith("<html")) {
            return {
                message: `服务器返回了非 JSON 错误页面，状态码 ${response.status}。请查看后端终端日志。`,
            };
        }
        return {
            message: rawText.trim(),
        };
    }
}

async function previewBundleImage(fieldId) {
    const input = document.getElementById(fieldId);
    const card = document.getElementById(`${fieldId}_preview_card`);
    const image = document.getElementById(`${fieldId}_preview`);
    const meta = document.getElementById(`${fieldId}_preview_meta`);
    const name = document.getElementById(`${fieldId}_preview_name`);

    resetBundleQualityPanel();

    if (!input.files.length) {
        card.classList.add("hidden");
        image.removeAttribute("src");
        meta.textContent = "尚未读取图像信息";
        name.textContent = "尚未选择图像";
        clearPreviewUrl(fieldId);
        if (fieldId === "medium_view") {
            resetMediumTargetPrecheck();
        }
        updateBundleSizeSummary();
        return;
    }

    const file = input.files[0];
    clearPreviewUrl(fieldId);
    const url = URL.createObjectURL(file);
    previewUrls.set(fieldId, url);
    image.src = url;
    name.textContent = file.name;
    card.classList.remove("hidden");

    const sizeMb = (file.size / 1024 / 1024).toFixed(2);
    const preview = new Image();
    preview.onload = () => {
        meta.textContent = `${preview.width} × ${preview.height} / ${sizeMb} MB`;
    };
    preview.src = url;
    if (fieldId === "medium_view") {
        await runMediumTargetPrecheck(file);
    }
    updateBundleSizeSummary();
}

function renderBundleQualityResults(results, rejected = false) {
    const qualityList = document.getElementById("bundleQualityList");
    qualityList.innerHTML = "";

    results.forEach((item) => {
        const report = item.quality_report || {};
        const metrics = report.metrics || {};
        const warnings = (report.warnings || [])
            .map((warning) => `<li>${escapeHtml(warning)}</li>`)
            .join("");

        const block = document.createElement("article");
        block.className = "bundle-quality-card";
        block.innerHTML = `
            <div class="bundle-quality-head">
                <strong>${escapeHtml(item.capture_scale || "未标注")}</strong>
                <span class="quality-badge ${getQualityBadgeClass(report.status)}">${escapeHtml(report.status || "待检测")}</span>
            </div>
            <p class="bundle-quality-file">${escapeHtml(item.original_filename || "-")}</p>
            <div class="bundle-quality-meta">
                <span>质量分：${escapeHtml(report.score ?? "-")}</span>
                <span>分辨率：${metrics.width && metrics.height ? `${metrics.width} × ${metrics.height}` : "-"}</span>
                <span>亮度：${escapeHtml(metrics.brightness ?? "-")}</span>
                <span>边缘强度：${escapeHtml(metrics.edge_strength ?? "-")}</span>
            </div>
            <p class="bundle-quality-summary">${escapeHtml(report.summary || "")}</p>
            ${warnings ? `<ul class="bundle-quality-warnings">${warnings}</ul>` : ""}
        `;
        qualityList.appendChild(block);
    });

    document.getElementById("bundleQualitySummary").textContent = rejected
        ? "三张图中至少有一张未通过预检，请补拍或强制继续。"
        : "三张图的质量预检结果如下。";
    setVisible("bundleQualityPanel", true);
    setVisible("overrideUploadButton", rejected);
}

function formatMetricValue(mmValue, pxValue, unitLabel = "") {
    if (mmValue !== null && mmValue !== undefined) {
        return `${Number(mmValue).toFixed(3)} mm${unitLabel}`;
    }
    if (pxValue !== null && pxValue !== undefined) {
        return `${Number(pxValue).toFixed(3)} px${unitLabel}`;
    }
    return "-";
}

function formatRiskScore(risk = {}) {
    if (risk.score === null || risk.score === undefined || risk.score === "") {
        return "-";
    }
    return `${risk.score} 分`;
}

function formatRiskThresholds(thresholds = []) {
    if (!Array.isArray(thresholds) || !thresholds.length) {
        return "";
    }

    return thresholds
        .map((item) => {
            const minScore = item.min_score ?? 0;
            const maxScore = item.max_score;
            const scoreRange = maxScore === null || maxScore === undefined ? `${minScore}+` : `${minScore}-${maxScore}`;
            return `${item.risk_level || "-"} ${scoreRange} 分`;
        })
        .join(" / ");
}

function formatCaptureScale(value) {
    if (!value) {
        return "-";
    }
    return CAPTURE_SCALE_LABELS[value] || value;
}

function getRiskToneClass(level) {
    const text = String(level || "");
    if (text.includes("高")) {
        return "tone-danger";
    }
    if (text.includes("中")) {
        return "tone-warning";
    }
    if (text.includes("低") || text.includes("无")) {
        return "tone-safe";
    }
    return "tone-neutral";
}

function getStageToneClass(status) {
    const text = String(status || "").toLowerCase();
    if (text.includes("complete") || text.includes("完成")) {
        return "tone-safe";
    }
    if (text.includes("fail") || text.includes("错误") || text.includes("失败")) {
        return "tone-danger";
    }
    if (text.includes("run") || text.includes("queue") || text.includes("进行") || text.includes("等待")) {
        return "tone-warning";
    }
    return "tone-neutral";
}

function buildMetaChipGroup(items = []) {
    const chips = items
        .filter((item) => item && item.label && item.value !== null && item.value !== undefined && item.value !== "")
        .map(
            (item) => `
                <span class="meta-chip ${escapeHtml(item.toneClass || "")}">
                    <em>${escapeHtml(item.label)}</em>
                    <strong>${escapeHtml(item.value)}</strong>
                </span>
            `
        )
        .join("");

    return chips ? `<div class="meta-chip-group">${chips}</div>` : "";
}

function getHistoryHousePageCount() {
    return Math.max(1, Math.ceil(historyHouseDirectory.length / 6));
}

function goHistoryHousePage(page) {
    historyHousePage = Math.min(Math.max(1, page), getHistoryHousePageCount());
    renderHistoryHouseDirectory();
}

function renderHistoryHousePager() {
    const pager = document.getElementById("historyHousePager");
    if (!pager) {
        return;
    }

    const totalPages = getHistoryHousePageCount();
    if (historyHouseDirectory.length <= 6) {
        pager.innerHTML = "";
        setVisible("historyHousePager", false);
        return;
    }

    historyHousePage = Math.min(Math.max(1, historyHousePage), totalPages);
    pager.innerHTML = `
        <button type="button" class="secondary-button history-house-pager-button" onclick="goHistoryHousePage(${historyHousePage - 1})" ${historyHousePage <= 1 ? "disabled" : ""}>上一页</button>
        <span class="history-house-pager-status">${historyHousePage}/${totalPages}</span>
        <button type="button" class="secondary-button history-house-pager-button" onclick="goHistoryHousePage(${historyHousePage + 1})" ${historyHousePage >= totalPages ? "disabled" : ""}>下一页</button>
    `;
    setVisible("historyHousePager", true);
}

function renderDashboardOverview(elementId, cards = []) {
    const container = document.getElementById(elementId);
    if (!container) {
        return;
    }

    const validCards = cards.filter((card) => card && (card.title || card.value || card.detail));
    if (!validCards.length) {
        container.innerHTML = "";
        setVisible(elementId, false);
        return;
    }

    container.innerHTML = validCards
        .map(
            (card) => `
                <article class="dashboard-card ${escapeHtml(card.toneClass || "")}">
                    <span>${escapeHtml(card.title || "-")}</span>
                    <strong>${escapeHtml(card.value || "-")}</strong>
                    ${card.detail ? `<p>${escapeHtml(card.detail)}</p>` : ""}
                </article>
            `
        )
        .join("");
    container.className = "dashboard-overview";
    setVisible(elementId, true);
}

function renderBatchResultOverview(payload) {
    const results = Array.isArray(payload.results) ? payload.results : [];
    if (!results.length) {
        renderDashboardOverview("batchResultOverview", []);
        return;
    }

    const mediumHighRiskCount = results.filter((item) => {
        const level = item.risk_assessment?.risk_level || item.risk_level || "";
        return level === "中风险" || level === "高风险";
    }).length;
    const quantifiedCount = results.filter((item) => {
        const quant = item.quantification || {};
        return quant.max_width_mm !== null && quant.max_width_mm !== undefined;
    }).length;
    const reportReady = payload.bundle_report_url ? "已生成" : "处理中";

    renderDashboardOverview("batchResultOverview", [
        {
            title: "批次编号",
            value: payload.bundle_code || "-",
            detail: payload.task_code ? `任务 ${payload.task_code}` : "当前未返回任务编号",
            toneClass: "tone-neutral",
        },
        {
            title: "图像数量",
            value: `${results.length} 张`,
            detail: "长景 / 中景 / 近景批量结果",
            toneClass: "tone-safe",
        },
        {
            title: "中高风险",
            value: `${mediumHighRiskCount} 项`,
            detail: "建议优先查看风险说明与报告",
            toneClass: mediumHighRiskCount ? "tone-warning" : "tone-safe",
        },
        {
            title: "mm 量化",
            value: `${quantifiedCount} / ${results.length}`,
            detail: "已建立物理尺度的结果数量",
            toneClass: quantifiedCount ? "tone-safe" : "tone-neutral",
        },
        {
            title: "总报告",
            value: reportReady,
            detail: reportReady === "已生成" ? "可直接下载三景总报告" : "等待批次处理完成",
            toneClass: reportReady === "已生成" ? "tone-safe" : "tone-warning",
        },
        {
            title: "上传时间",
            value: payload.uploaded_at_display || "-",
            detail: payload.generated_at_display ? `结果生成 ${payload.generated_at_display}` : "等待结果生成",
            toneClass: payload.generated_at_display ? "tone-safe" : "tone-neutral",
        },
    ]);
}

function renderHistoryOverview(records) {
    if (!records.length) {
        renderDashboardOverview("historyOverview", []);
        return;
    }

    const latestRecord = records[0];
    const quantifiedCount = records.filter((record) => {
        const quant = record.quantification || {};
        return quant.max_width_mm !== null && quant.max_width_mm !== undefined;
    }).length;
    const mediumHighRiskCount = records.filter((record) => {
        const level = record.risk_assessment?.risk_level || record.risk_level || "";
        return level === "中风险" || level === "高风险";
    }).length;
    const latestRiskLevel = latestRecord.risk_assessment?.risk_level || latestRecord.risk_level || "-";

    renderDashboardOverview("historyOverview", [
        {
            title: "历史记录",
            value: `${records.length} 次`,
            detail: "当前房屋已完成检测次数",
            toneClass: "tone-neutral",
        },
        {
            title: "最新风险",
            value: latestRiskLevel,
            detail: getGeneratedTimeText(latestRecord),
            toneClass: getRiskToneClass(latestRiskLevel),
        },
        {
            title: "中高风险记录",
            value: `${mediumHighRiskCount} 次`,
            detail: "建议结合趋势分析持续复核",
            toneClass: mediumHighRiskCount ? "tone-warning" : "tone-safe",
        },
        {
            title: "mm 量化记录",
            value: `${quantifiedCount} 次`,
            detail: "已完成物理尺度换算的历史记录",
            toneClass: quantifiedCount ? "tone-safe" : "tone-neutral",
        },
    ]);
}

/*
function renderHistoryHouseDirectory(houses = historyHouseDirectory) {
    const board = document.getElementById("historyHouseBoard");
    const list = document.getElementById("historyHouseList");
    const hasItems = Array.isArray(houses) && houses.length > 0;
    historyHouseDirectory = hasItems ? houses.slice() : [];

    const boardTitle = board?.querySelector(".section-subhead strong");
    const boardDescription = board?.querySelector(".section-subhead p");
    if (boardTitle) {
        boardTitle.textContent = "\u5386\u53f2\u68c0\u6d4b\u623f\u5c4b";
    }
    if (boardDescription) {
        boardDescription.textContent = "\u6309\u884c\u5c55\u793a\u5df2\u6709\u68c0\u6d4b\u8bb0\u5f55\u7684\u623f\u5c4b\uff0c\u6bcf\u9875\u6700\u591a 6 \u680b\u3002\u70b9\u51fb\u623f\u53f7\u53ef\u6253\u5f00\u6700\u65b0\u62a5\u544a\uff0c\u70b9\u51fb\u201c\u67e5\u770b\u5386\u53f2\u201d\u53ef\u5c55\u5f00\u5386\u6b21\u68c0\u6d4b\u8bb0\u5f55\u3002";
    }

    if (!list) {
        return;
    }

    if (!hasItems) {
        list.innerHTML = "";
        setVisible("historyHouseBoard", false);
        if (!activeHistoryHouseId) {
            setVisible("historyPanel", false);
        }
        return;
    }

    list.innerHTML = historyHouseDirectory
        .map((item) => {
            const isActive = Number(item.house_id) === Number(activeHistoryHouseId);
            const reportUrl = item.latest_report_view_url || item.latest_report_url || "#";
            const bundleLink = item.latest_bundle_report_url
                ? `<a class="button-link compact-link" href="${escapeHtml(item.latest_bundle_report_url)}?download=0" target="_blank" rel="noopener noreferrer">鎵撳紑鎬绘姤鍛?/a>`
                : "";
            return `
                <article class="history-item history-house-card${isActive ? " history-house-card-active" : ""}">
                    <div class="history-header">
                        <div>
                            <a class="history-house-link" href="${escapeHtml(reportUrl)}" target="_blank" rel="noopener noreferrer">
                                ${escapeHtml(item.house_number || "-")}
                            </a>
                            <p>${escapeHtml(item.latest_detection_code || "鏆傛棤妫€娴嬬紪鍙?)}</p>
                        </div>
                        <div class="history-head-meta">
                            <span class="meta-chip ${getRiskToneClass(item.latest_risk_level || "-")}">
                                <em>鏈€鏂伴闄?/em>
                                <strong>${escapeHtml(item.latest_risk_level || "-")}</strong>
                            </span>
                            <span class="history-time">${escapeHtml(item.latest_created_at || "鏈煡鏃堕棿")}</span>
                        </div>
                    </div>
                    ${buildMetaChipGroup([
                        { label: "鍘嗗彶璁板綍", value: `${item.history_count || 0} 娆?` },
                        { label: "鏋勪欢", value: item.latest_component_type || "-" },
                        { label: "鍦烘櫙", value: item.latest_scenario_type || "-" },
                    ])}
                    <div class="history-actions">
                        <button type="button" class="secondary-button" onclick="openHouseHistory(${Number(item.house_id)})">鏌ョ湅鍘嗗彶</button>
                        <a class="button-link compact-link" href="${escapeHtml(reportUrl)}" target="_blank" rel="noopener noreferrer">鎵撳紑鏈€鏂版姤鍛?/a>
                        ${bundleLink}
                    </div>
                </article>
            `;
        })
        .join("");

    setVisible("historyHouseBoard", true);
    setVisible("historyPanel", true);
}

function openHouseHistory(houseId) {
    activeHistoryHouseId = houseId;
    renderHistoryHouseDirectory();
    loadHouseDetections(houseId);
}

*/

function renderHistoryHouseDirectory(houses = historyHouseDirectory) {
    const board = document.getElementById("historyHouseBoard");
    const list = document.getElementById("historyHouseList");
    const hasItems = Array.isArray(houses) && houses.length > 0;
    historyHouseDirectory = hasItems ? houses.slice() : [];
    const boardTitle = board?.querySelector(".section-subhead strong");
    const boardDescription = board?.querySelector(".section-subhead p");

    if (boardTitle) {
        boardTitle.textContent = "\u5386\u53f2\u68c0\u6d4b\u623f\u5c4b";
    }
    if (boardDescription) {
        boardDescription.textContent = "\u4ee5\u4e0b\u623f\u5c4b\u5df2\u6709\u68c0\u6d4b\u8bb0\u5f55\u3002\u70b9\u51fb\u623f\u53f7\u53ef\u76f4\u63a5\u6253\u5f00\u6700\u65b0\u68c0\u6d4b\u62a5\u544a\uff0c\u70b9\u51fb\u201c\u67e5\u770b\u5386\u53f2\u201d\u53ef\u5c55\u5f00\u8be5\u623f\u5c4b\u7684\u5386\u6b21\u68c0\u6d4b\u8bb0\u5f55\u3002";
    }

    if (!list) {
        return;
    }

    if (!hasItems) {
        list.innerHTML = "";
        renderHistoryHousePager();
        setVisible("historyHouseBoard", false);
        if (!activeHistoryHouseId) {
            setVisible("historyPanel", false);
        }
        return;
    }

    const activeIndex = historyHouseDirectory.findIndex((item) => Number(item.house_id) === Number(activeHistoryHouseId));
    const totalPages = getHistoryHousePageCount();
    if (activeIndex >= 0) {
        historyHousePage = Math.floor(activeIndex / 6) + 1;
    }
    historyHousePage = Math.min(Math.max(1, historyHousePage), totalPages);
    const pageStart = (historyHousePage - 1) * 6;
    const pageItems = historyHouseDirectory.slice(pageStart, pageStart + 6);

    list.innerHTML = pageItems
        .map((item, index) => {
            const isActive = Number(item.house_id) === Number(activeHistoryHouseId);
            const reportUrl = item.latest_report_view_url || item.latest_report_url || "#";
            const bundleLink = item.latest_bundle_report_url
                ? `<a class="button-link compact-link" href="${escapeHtml(item.latest_bundle_report_url)}?download=0" target="_blank" rel="noopener noreferrer">\u603b\u62a5\u544a</a>`
                : "";
            const order = pageStart + index + 1;

            return `
                <article class="history-item history-house-card${isActive ? " history-house-card-active" : ""}">
                    <div class="history-house-row">
                        <div class="history-house-main">
                            <a class="history-house-link" href="${escapeHtml(reportUrl)}" target="_blank" rel="noopener noreferrer">
                                <span class="history-house-order">${order}.</span>
                                <span>${escapeHtml(item.house_number || "-")}</span>
                            </a>
                            <p class="history-house-code">${escapeHtml(item.latest_detection_code || "\u6682\u65e0\u68c0\u6d4b\u7f16\u53f7")}</p>
                        </div>
                        <div class="history-house-side">
                            <span class="meta-chip ${getRiskToneClass(item.latest_risk_level || "-")}">
                                <em>\u6700\u65b0\u98ce\u9669</em>
                                <strong>${escapeHtml(item.latest_risk_level || "-")}</strong>
                            </span>
                            <span class="history-time">${escapeHtml(item.latest_generated_at_display || item.latest_created_at_display || "\u672a\u77e5\u65f6\u95f4")}</span>
                        </div>
                    </div>
                    <div class="history-house-row history-house-row-meta">
                        ${buildMetaChipGroup([
                            { label: "\u5386\u53f2\u6b21\u6570", value: `${item.history_count || 0}` },
                            { label: "\u6784\u4ef6", value: item.latest_component_type || "-" },
                            { label: "\u573a\u666f", value: item.latest_scenario_type || "-" },
                        ])}
                        <div class="history-actions history-house-actions">
                            <button type="button" class="secondary-button" onclick="openHouseHistory(${Number(item.house_id)})">\u67e5\u770b\u5386\u53f2</button>
                            <a class="button-link compact-link" href="${escapeHtml(reportUrl)}" target="_blank" rel="noopener noreferrer">\u6700\u65b0\u62a5\u544a</a>
                            ${bundleLink}
                        </div>
                    </div>
                </article>
            `;
        })
        .join("");

    renderHistoryHousePager();
    setVisible("historyHouseBoard", true);
    setVisible("historyPanel", true);
}

function openHouseHistory(houseId) {
    activeHistoryHouseId = houseId;
    const targetIndex = historyHouseDirectory.findIndex((item) => Number(item.house_id) === Number(houseId));
    if (targetIndex >= 0) {
        historyHousePage = Math.floor(targetIndex / 6) + 1;
    }
    renderHistoryHouseDirectory();
    loadHouseDetections(houseId);
}

function renderSurveyInsights(summary) {
    const container = document.getElementById("surveyInsights");
    if (!container) {
        return;
    }

    const distribution = summary.risk_distribution || {};
    const highRiskCount = distribution["高风险"] || 0;
    const mediumRiskCount = distribution["中风险"] || 0;
    const houseCount = summary.house_count || 0;
    const headline = highRiskCount
        ? `当前有 ${highRiskCount} 栋高风险房屋需要优先复核。`
        : mediumRiskCount
            ? `当前暂无高风险房屋，但有 ${mediumRiskCount} 栋中风险房屋需要关注。`
            : "当前平台暂无中高风险房屋，可继续扩大巡检覆盖范围。";

    container.innerHTML = `
        <div>
            <p class="upload-hero-kicker">Survey Insight</p>
            <strong>平台巡检态势</strong>
            <p>${escapeHtml(headline)}</p>
        </div>
        <div class="dashboard-hero-metrics">
            <article>
                <span>房屋覆盖</span>
                <strong>${escapeHtml(houseCount)}</strong>
            </article>
            <article>
                <span>高风险</span>
                <strong>${escapeHtml(highRiskCount)}</strong>
            </article>
            <article>
                <span>中风险</span>
                <strong>${escapeHtml(mediumRiskCount)}</strong>
            </article>
        </div>
    `;
    container.className = "dashboard-hero";
    setVisible("surveyInsights", true);
}

function buildDetailSection(title, body, { open = false, className = "" } = {}) {
    if (!body || !body.trim()) {
        return "";
    }

    return `
        <details class="detail-section ${escapeHtml(className)}"${open ? " open" : ""}>
            <summary>
                <span>${escapeHtml(title)}</span>
                <em>展开 / 收起</em>
            </summary>
            <div class="detail-section-body">
                ${body}
            </div>
        </details>
    `;
}

function buildRiskBreakdownCard(risk = {}) {
    const breakdown = Array.isArray(risk.score_breakdown) ? risk.score_breakdown : [];
    const notes = Array.isArray(risk.explanation_notes) ? risk.explanation_notes : [];
    const hasContent =
        breakdown.length ||
        notes.length ||
        risk.score !== null && risk.score !== undefined ||
        risk.risk_level ||
        risk.risk_summary;

    if (!hasContent) {
        return "";
    }

    const thresholdText = formatRiskThresholds(risk.level_thresholds || []);
    const breakdownHtml = breakdown.length
        ? `
            <ul class="risk-breakdown-list">
                ${breakdown
                    .map(
                        (item) => `
                            <li class="risk-breakdown-item">
                                <div class="risk-breakdown-line">
                                    <span class="risk-breakdown-points">+${escapeHtml(item.score ?? 0)} 分</span>
                                    <strong>${escapeHtml(item.title || "评分项")}</strong>
                                </div>
                                <p>${escapeHtml(item.detail || "-")}</p>
                            </li>
                        `
                    )
                    .join("")}
            </ul>
        `
        : '<p class="risk-breakdown-empty">当前未触发明显加分项，系统按常规观察范围处理。</p>';

    const notesHtml = notes.length
        ? `
            <ul class="risk-note-list">
                ${notes.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
            </ul>
        `
        : "";

    const thresholdHtml = thresholdText
        ? `<p class="risk-threshold-text">分级阈值：${escapeHtml(thresholdText)}</p>`
        : "";

    return `
        <article class="note-card risk-breakdown-card">
            <div class="risk-breakdown-head">
                <h3>风险评分依据</h3>
                <strong>${escapeHtml(risk.risk_level || "未生成")} / ${escapeHtml(formatRiskScore(risk))}</strong>
            </div>
            ${breakdownHtml}
            ${notesHtml}
            ${thresholdHtml}
        </article>
    `;
}

function formatMarkerMode(markerDetection = {}) {
    if (markerDetection.detection_method === "qr_anchor_rectified" || markerDetection.status === "anchor_rectified") {
        return "二维码 + 四角矫正";
    }
    if (markerDetection.detection_method === "qr_decode" || markerDetection.status === "qr_detected") {
        return "二维码解码";
    }
    if (markerDetection.detection_method === "legacy_rect" || markerDetection.status === "fallback_detected") {
        return "矩形回退";
    }
    if (markerDetection.status === "completed") {
        return "旧版靶标识别";
    }
    return "-";
}

function formatPlaneMode(quantification = {}) {
    if (quantification.plane_mode === "rectified_plane") {
        return "透视矫正平面";
    }
    if (quantification.plane_mode === "image_plane") {
        return "原图像平面";
    }
    return "-";
}

function buildStageCards(stages = {}) {
    const recognition = stages.recognition || {};
    const quantification = stages.quantification || {};
    return `
        <div class="stage-grid">
            <article class="stage-card">
                <div class="stage-title-row">
                    <span class="stage-kicker">识别阶段</span>
                    <span class="stage-status-chip ${getStageToneClass(recognition.status)}">${escapeHtml(recognition.status || "-")}</span>
                </div>
                <p>${escapeHtml(recognition.summary || "暂无识别阶段说明。")}</p>
                <span class="stage-meta">${escapeHtml(recognition.physical_scale || "未建立物理比例")}</span>
            </article>
            <article class="stage-card">
                <div class="stage-title-row">
                    <span class="stage-kicker">量化阶段</span>
                    <span class="stage-status-chip ${getStageToneClass(quantification.status)}">${escapeHtml(quantification.status || "-")}</span>
                </div>
                <p>${escapeHtml(quantification.summary || "暂无量化阶段说明。")}</p>
            </article>
        </div>
    `;
}

function buildQuantificationTable(markerDetection = {}, quantification = {}) {
    return `
        <div class="quant-table">
            <div><span>识别方式</span><strong>${escapeHtml(formatMarkerMode(markerDetection))}</strong></div>
            <div><span>量化平面</span><strong>${escapeHtml(formatPlaneMode(quantification))}</strong></div>
            <div><span>靶标数量</span><strong>${escapeHtml(markerDetection.marker_count ?? "-")}</strong></div>
            <div><span>mm/pixel</span><strong>${markerDetection.physical_scale_mm_per_pixel ? escapeHtml(Number(markerDetection.physical_scale_mm_per_pixel).toFixed(6)) : "-"}</strong></div>
            <div><span>最大宽度</span><strong>${escapeHtml(formatMetricValue(quantification.max_width_mm, quantification.max_width_px))}</strong></div>
            <div><span>平均宽度</span><strong>${escapeHtml(formatMetricValue(quantification.avg_width_mm, quantification.avg_width_px))}</strong></div>
            <div><span>中位宽度</span><strong>${escapeHtml(formatMetricValue(quantification.median_width_mm, quantification.median_width_px))}</strong></div>
            <div><span>裂缝长度</span><strong>${escapeHtml(formatMetricValue(quantification.crack_length_mm, quantification.crack_length_px))}</strong></div>
            <div><span>裂缝角度</span><strong>${quantification.crack_angle_deg !== null && quantification.crack_angle_deg !== undefined ? `${escapeHtml(Number(quantification.crack_angle_deg).toFixed(2))} deg` : "-"}</strong></div>
            <div><span>样本点数</span><strong>${escapeHtml(quantification.sample_count ?? "-")}</strong></div>
        </div>
    `;
}

function buildResultStatTable(items = []) {
    const rows = items
        .filter((item) => item && item.label && item.value !== null && item.value !== undefined && item.value !== "")
        .map(
            (item) => `
                <div>
                    <dt>${escapeHtml(item.label)}</dt>
                    <dd>${escapeHtml(item.value)}</dd>
                </div>
            `
        )
        .join("");

    return rows ? `<dl class="result-stat-table">${rows}</dl>` : "";
}

function buildResultNoteTable(items = []) {
    const rows = items
        .filter((item) => item && item.label && item.value !== null && item.value !== undefined && item.value !== "")
        .map(
            (item) => `
                <div>
                    <dt>${escapeHtml(item.label)}</dt>
                    <dd>${escapeHtml(item.value)}</dd>
                </div>
            `
        )
        .join("");

    return rows ? `<dl class="result-note-table">${rows}</dl>` : "";
}

function buildResultMediaCard(title, url, emptyText) {
    return `
        <article class="result-card">
            <div class="result-card-head">
                <span class="note-kicker">图像输出</span>
                <h3>${escapeHtml(title)}</h3>
            </div>
            ${url ? `<img src="${escapeHtml(url)}" alt="${escapeHtml(title)}">` : `<div class="result-placeholder">${escapeHtml(emptyText)}</div>`}
            ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">打开图像</a>` : ""}
        </article>
    `;
}

function buildResultCard(item) {
    const quality = item.quality_report || {};
    const risk = item.risk_assessment || {
        risk_level: item.risk_level,
        score: item.risk_score,
        risk_summary: item.risk_summary,
        recommendation: item.recommendation,
    };
    const markerDetection = item.marker_detection || {};
    const quantification = item.quantification || {};
    const analysisStages = item.analysis_stages || {};
    const captureScale = formatCaptureScale(item.capture_scale || "未标注");
    const riskLevel = risk.risk_level || "未生成";
    const reportLink = item.report_url
        ? `<a class="button-link compact-link" href="${escapeHtml(item.report_url)}" target="_blank" rel="noopener noreferrer">下载报告</a>`
        : "";
    const resultStats = buildResultStatTable([
        { label: "上传时间", value: getUploadedTimeText(item) },
        { label: "生成时间", value: getGeneratedTimeText(item) },
        { label: "质量分", value: quality.score ?? "-" },
        { label: "风险评分", value: formatRiskScore(risk) },
        { label: "最大宽度", value: formatMetricValue(quantification.max_width_mm, quantification.max_width_px) },
        { label: "裂缝长度", value: formatMetricValue(quantification.crack_length_mm, quantification.crack_length_px) },
        { label: "裂缝占比", value: item.segmentation?.crack_area_ratio ?? "-" },
        {
            label: "mm/pixel",
            value: markerDetection.physical_scale_mm_per_pixel
                ? Number(markerDetection.physical_scale_mm_per_pixel).toFixed(6)
                : "-",
        },
        { label: "识别方式", value: formatMarkerMode(markerDetection) },
        { label: "量化平面", value: formatPlaneMode(quantification) },
        { label: "靶标数量", value: markerDetection.marker_count ?? "-" },
        { label: "样本点数", value: quantification.sample_count ?? "-" },
    ]);
    const resultNotes = buildResultNoteTable([
        { label: "质量说明", value: quality.summary || item.message || "-" },
        { label: "风险说明", value: risk.risk_summary || item.segmentation_error || item.message || "-" },
        { label: "处置建议", value: risk.recommendation || item.recommendation || "暂无处置建议。" },
    ]);
    const analysisDetail = [
        buildRiskBreakdownCard(risk),
        buildStageCards(analysisStages),
        `
            <article class="note-card quant-note-card">
                <span class="note-kicker">量化阶段说明</span>
                <p>${escapeHtml(quantification.message || "当前暂无量化说明。")}</p>
            </article>
        `,
        buildQuantificationTable(markerDetection, quantification),
    ].join("");
    const mediaDetail = `
        <div class="result-media-head">
            <div>
                <span class="note-kicker">图像预览</span>
                <p>原图、识别叠加、量化结果集中显示。</p>
            </div>
        </div>
        <div class="result-grid">
            ${buildResultMediaCard("原图", item.file_url, "无原图")}
            ${buildResultMediaCard("裂缝掩码", item.segmentation?.mask_url, "无掩码")}
            ${buildResultMediaCard("识别叠加图", item.segmentation?.overlay_url, "无识别图")}
            ${buildResultMediaCard("靶标识别图", markerDetection.annotated_image_url, "未生成靶标图")}
            ${buildResultMediaCard("量化叠加图", quantification.quant_overlay_url, "未生成量化图")}
            ${buildResultMediaCard("宽度统计图", quantification.width_chart_url, "未生成统计图")}
        </div>
        <dl class="result-meta">
            <div>
                <dt>裂缝像素占比</dt>
                <dd>${escapeHtml(item.segmentation?.crack_area_ratio ?? "-")}</dd>
            </div>
            <div>
                <dt>裂缝像素数</dt>
                <dd>${escapeHtml(item.segmentation?.crack_pixel_count ?? "-")}</dd>
            </div>
            <div>
                <dt>推理设备</dt>
                <dd>${escapeHtml(item.segmentation?.device ?? "-")}</dd>
            </div>
            <div>
                <dt>切块数量</dt>
                <dd>${escapeHtml(item.segmentation?.patch_count ?? "-")}</dd>
            </div>
            <div>
                <dt>靶标状态</dt>
                <dd>${escapeHtml(markerDetection.status || "-")}</dd>
            </div>
            <div>
                <dt>量化状态</dt>
                <dd>${escapeHtml(quantification.status || "-")}</dd>
            </div>
        </dl>
    `;
    return `
        <article class="batch-result-card result-shell">
            <div class="result-card-split">
                <div class="result-info-column">
                    <div class="batch-result-head">
                        <div class="result-title-block">
                            <span class="result-capture-tag">${escapeHtml(captureScale)}</span>
                            <strong>${escapeHtml(item.detection_code || item.original_filename || "未命名结果")}</strong>
                            <p>${escapeHtml(item.original_filename || "")}</p>
                        </div>
                        <div class="result-badges">
                            <span class="quality-badge ${getQualityBadgeClass(quality.status)}">${escapeHtml(quality.status || "未知")}</span>
                            <span class="meta-chip ${getRiskToneClass(riskLevel)}">
                                <em>风险</em>
                                <strong>${escapeHtml(riskLevel)}</strong>
                            </span>
                        </div>
                    </div>
                    ${buildMetaChipGroup([
                        { label: "检测编号", value: item.detection_code || "-" },
                        { label: "识别方式", value: formatMarkerMode(markerDetection) },
                        { label: "量化平面", value: formatPlaneMode(quantification) },
                        { label: "样本点", value: quantification.sample_count ?? "-" },
                    ])}
                    ${resultStats}
                    ${resultNotes}
                    <div class="report-actions">
                        ${reportLink}
                    </div>
                    ${buildDetailSection("分析阶段与量化指标", analysisDetail, { open: true })}
                </div>
                <aside class="result-media-column">
                    ${mediaDetail}
                </aside>
            </div>
        </article>
    `;
}

function renderBundleProjection(projection) {
    const panel = document.getElementById("bundleProjectionPanel");
    if (!projection) {
        panel.innerHTML = "";
        setVisible("bundleProjectionPanel", false);
        return;
    }

    const projectedItems = Array.isArray(projection.projected_items) ? projection.projected_items : [];
    const bbox = projection.component_bbox || null;
    const badgeClass = projection.status === "completed" ? "quality-good" : "quality-warning";
    const badgeText = projection.status === "completed" ? "已生成" : "未生成";
    const bboxText = bbox
        ? `关注框：x ${bbox.x} / y ${bbox.y} / w ${bbox.width} / h ${bbox.height}`
        : "关注框：当前未生成";
    const itemList = projectedItems.length
        ? `
            <div class="bundle-projection-list">
                ${projectedItems
                    .map((item) => {
                        const itemBbox = item.bbox || null;
                        const itemBboxText = itemBbox
                            ? `回投框 x ${itemBbox.x} / y ${itemBbox.y} / w ${itemBbox.width} / h ${itemBbox.height}`
                            : "未生成回投框";
                        return `
                            <article class="summary-card bundle-projection-item">
                                <span>${escapeHtml(item.capture_scale || "-")}</span>
                                <strong>${escapeHtml(item.detection_code || "-")}</strong>
                                <p>${escapeHtml(itemBboxText)}</p>
                            </article>
                        `;
                    })
                    .join("")}
            </div>
        `
        : "";

    panel.className = "note-card bundle-projection-panel";
    panel.innerHTML = `
        <div class="bundle-projection-head">
            <div>
                <h3>长景构件框选 + 中近景回投</h3>
                <p>${escapeHtml(projection.message || "当前批次尚未生成长景回投结果。")}</p>
            </div>
            <span class="quality-badge ${badgeClass}">${badgeText}</span>
        </div>
        ${projection.overview_image_url ? `<img src="${escapeHtml(projection.overview_image_url)}" alt="长景构件框选与中近景回投总览">` : ""}
        <p class="bundle-projection-meta">${escapeHtml(bboxText)}</p>
        ${itemList}
    `;
    setVisible("bundleProjectionPanel", true);
}

function renderBatchUploadResult(payload) {
    const results = payload.results || [];
    const list = document.getElementById("batchResultList");
    renderBatchResultOverview(payload);
    list.innerHTML = results.map(buildResultCard).join("");
    renderBundleProjection(payload.bundle_projection);
    const batchBundleActions = document.getElementById("batchBundleActions");
    if (payload.bundle_report_url) {
        const projectionLink = payload.bundle_projection?.overview_image_url
            ? `
                <a class="button-link" href="${escapeHtml(payload.bundle_projection.overview_image_url)}" target="_blank" rel="noopener noreferrer">
                    打开长景回投图
                </a>
            `
            : "";
        batchBundleActions.innerHTML = `
            <a class="button-link" href="${escapeHtml(payload.bundle_report_url)}" target="_blank" rel="noopener noreferrer">
                下载三景总报告
            </a>
            ${projectionLink}
        `;
        setVisible("batchBundleActions", true);
    } else {
        batchBundleActions.innerHTML = "";
        setVisible("batchBundleActions", false);
    }
    setVisible("uploadResult", true);
}

function renderTrendSummary(summary) {
    if (!summary) {
        setVisible("trendSummaryCard", false);
        return;
    }

    const element = document.getElementById("trendSummaryText");
    const card = document.getElementById("trendSummaryCard");
    card.className = `trend-card trend-${summary.status || "insufficient"}`;
    element.textContent = summary.message || "至少需要两次检测后才能生成趋势分析。";
    setVisible("trendSummaryCard", true);
}

function renderHistory(records) {
    const historyList = document.getElementById("historyList");
    historyList.innerHTML = "";
    setVisible("historyList", true);
    renderHistoryOverview(records);

    if (!records.length) {
        historyList.innerHTML = '<p class="history-empty">当前房屋还没有检测记录。</p>';
        setVisible("historyPanel", true);
        return;
    }

    records.forEach((record, index) => {
        const item = document.createElement("article");
        item.className = "history-item";
        const risk = record.risk_assessment || {
            risk_level: record.risk_level,
            score: record.risk_score,
            risk_summary: record.risk_summary,
            recommendation: record.recommendation,
        };
        const markerDetection = record.marker_detection || {};
        const quantification = record.quantification || {};
        const analysisStages = record.analysis_stages || {};
        const warnings = (record.quality_warnings || [])
            .map((warning) => `<li>${escapeHtml(warning)}</li>`)
            .join("");
        const riskLevel = risk.risk_level || record.risk_level || "-";
        const detailBody = `
            <div class="result-notes history-story-grid">
                <article class="note-card">
                    <span class="note-kicker">风险摘要</span>
                    <p class="history-summary">${escapeHtml(risk.risk_summary || record.risk_summary || "暂无风险说明。")}</p>
                </article>
                <article class="note-card">
                    <span class="note-kicker">量化摘要</span>
                    <p class="history-summary">${escapeHtml(quantification.message || "暂无量化说明。")}</p>
                </article>
                <article class="note-card">
                    <span class="note-kicker">处置建议</span>
                    <p class="history-summary">${escapeHtml(risk.recommendation || record.recommendation || "暂无处置建议。")}</p>
                </article>
            </div>
            ${buildStageCards(analysisStages)}
            ${buildQuantificationTable(markerDetection, quantification)}
            ${buildRiskBreakdownCard(risk)}
            ${warnings ? `<ul class="history-warnings">${warnings}</ul>` : ""}
            <div class="history-links">
                ${record.source_image_url ? `<a href="${escapeHtml(record.source_image_url)}" target="_blank" rel="noopener noreferrer">原图</a>` : ""}
                ${record.mask_url ? `<a href="${escapeHtml(record.mask_url)}" target="_blank" rel="noopener noreferrer">掩码</a>` : ""}
                ${record.overlay_url ? `<a href="${escapeHtml(record.overlay_url)}" target="_blank" rel="noopener noreferrer">识别图</a>` : ""}
                ${markerDetection.annotated_image_url ? `<a href="${escapeHtml(markerDetection.annotated_image_url)}" target="_blank" rel="noopener noreferrer">靶标图</a>` : ""}
                ${quantification.quant_overlay_url ? `<a href="${escapeHtml(quantification.quant_overlay_url)}" target="_blank" rel="noopener noreferrer">量化图</a>` : ""}
                ${quantification.width_chart_url ? `<a href="${escapeHtml(quantification.width_chart_url)}" target="_blank" rel="noopener noreferrer">统计图</a>` : ""}
            </div>
            <div class="history-actions">
                <a class="button-link compact-link" href="${escapeHtml(record.report_url || `/detections/${record.id}/report`)}" target="_blank" rel="noopener noreferrer">下载报告</a>
                ${record.bundle_report_url ? `<a class="button-link compact-link" href="${escapeHtml(record.bundle_report_url)}" target="_blank" rel="noopener noreferrer">下载总报告</a>` : ""}
                <button type="button" class="secondary-button" onclick="rerunDetection(${record.id})">重新分析</button>
                <button type="button" class="danger-button" onclick="deleteDetection(${record.id})">删除记录</button>
            </div>
        `;

        item.innerHTML = `
            <div class="history-header">
                <div>
                    <span class="result-capture-tag">${escapeHtml(formatCaptureScale(record.capture_scale || "-"))}</span>
                    <strong>${escapeHtml(record.detection_code || record.original_filename)}</strong>
                    <p>${escapeHtml(record.original_filename || "")}</p>
                </div>
                <div class="history-head-meta">
                    <span class="meta-chip ${getRiskToneClass(riskLevel)}">
                        <em>风险</em>
                        <strong>${escapeHtml(riskLevel)}</strong>
                    </span>
                    <span class="history-time">${escapeHtml(getGeneratedTimeText(record))}</span>
                </div>
            </div>
            ${buildMetaChipGroup([
                { label: "构件", value: record.component_type || "-" },
                { label: "场景", value: record.scenario_type || "-" },
                { label: "质量", value: `${record.quality_status || "-"} / ${record.quality_score ?? "-"}` },
                { label: "评分", value: formatRiskScore(risk), toneClass: getRiskToneClass(riskLevel) },
                { label: "裂缝占比", value: record.crack_area_ratio ?? "-" },
            ])}
            <p class="history-lead">${escapeHtml(risk.risk_summary || record.risk_summary || "暂无风险说明。")}</p>
            ${buildDetailSection("查看本次分析详情", detailBody, { open: index === 0, className: "history-detail-section" })}
        `;
        historyList.appendChild(item);
    });

    setVisible("historyPanel", true);
}

function renderSurveySummary(summary) {
    const cards = document.getElementById("surveySummaryCards");
    const riskList = document.getElementById("surveyRiskList");
    const distribution = summary.risk_distribution || {};
    renderHistoryHouseDirectory(summary.houses_with_history || []);
    renderSurveyInsights(summary);
    cards.innerHTML = `
        <article class="summary-card tone-neutral">
            <span>检测记录</span>
            <strong>${escapeHtml(summary.completed_records ?? 0)}</strong>
        </article>
        <article class="summary-card tone-safe">
            <span>房屋数量</span>
            <strong>${escapeHtml(summary.house_count ?? 0)}</strong>
        </article>
        <article class="summary-card tone-warning">
            <span>中高风险</span>
            <strong>${escapeHtml((distribution["中风险"] || 0) + (distribution["高风险"] || 0))}</strong>
        </article>
        <article class="summary-card tone-danger">
            <span>高风险</span>
            <strong>${escapeHtml(distribution["高风险"] || 0)}</strong>
        </article>
    `;

    const highRiskHouses = summary.high_risk_houses || [];
    if (!highRiskHouses.length) {
        riskList.innerHTML = '<p class="history-empty">当前暂无中高风险房屋清单。</p>';
        return;
    }

    riskList.innerHTML = highRiskHouses
        .map(
            (item) => `
                <article class="history-item">
                    <div class="history-header">
                        <div>
                            <strong>${escapeHtml(item.house_number || "-")}</strong>
                            <p>${escapeHtml(item.detection_code || "")}</p>
                        </div>
                        <div class="history-head-meta">
                            <span class="meta-chip ${getRiskToneClass(item.risk_level)}">
                                <em>风险</em>
                                <strong>${escapeHtml(item.risk_level || "-")}</strong>
                            </span>
                            <span class="history-time">${escapeHtml(getGeneratedTimeText(item))}</span>
                        </div>
                    </div>
                    ${buildMetaChipGroup([
                        { label: "构件", value: item.component_type || "-" },
                        { label: "场景", value: item.scenario_type || "-" },
                        { label: "最大宽度", value: item.max_width_mm !== null && item.max_width_mm !== undefined ? `${Number(item.max_width_mm).toFixed(3)} mm` : "-" },
                        { label: "裂缝长度", value: item.crack_length_mm !== null && item.crack_length_mm !== undefined ? `${Number(item.crack_length_mm).toFixed(3)} mm` : "-" },
                    ])}
                </article>
            `
        )
        .join("");
}

async function loadSurveySummary() {
    setButtonState("refreshSurveyButton", true, "刷新中...");
    try {
        const response = await fetch("/survey/summary");
        const result = await parseResponsePayload(response);
        if (!response.ok) {
            setGlobalStatus("error", result.message || "普查汇总加载失败。");
            return;
        }
        renderSurveySummary(result);
    } catch (error) {
        setGlobalStatus("error", `普查汇总加载失败：${error}`);
    } finally {
        setButtonState("refreshSurveyButton", false, "刷新汇总");
    }
}

async function loadHouseDetections(houseId) {
    if (!houseId) {
        setVisible("historyCurrentHint", false);
        setVisible("historyOverview", false);
        setVisible("trendSummaryCard", false);
        setVisible("historyList", false);
        setVisible("historyPanel", historyHouseDirectory.length > 0);
        return;
    }

    try {
        const response = await fetch(`/house/${houseId}/detections`);
        const result = await response.json();
        if (!response.ok) {
            setVisible("historyCurrentHint", false);
            setVisible("historyOverview", false);
            setVisible("trendSummaryCard", false);
            setVisible("historyList", false);
            setVisible("historyPanel", historyHouseDirectory.length > 0);
            return;
        }

        activeHistoryHouseId = houseId;
        document.getElementById("historyCurrentHint").textContent = result.house_number
            ? `褰撳墠姝ｅ湪鏌ョ湅鎴垮眿 ${result.house_number} 鐨勫巻鍙叉娴嬭褰曘€?`
            : "褰撳墠姝ｅ湪鏌ョ湅鎴垮眿鍘嗗彶妫€娴嬭褰曘€?";
        setVisible("historyCurrentHint", true);
        document.getElementById("historyCurrentHint").textContent = result.house_number
            ? `\u5f53\u524d\u6b63\u5728\u67e5\u770b\u623f\u5c4b ${result.house_number} \u7684\u5386\u53f2\u68c0\u6d4b\u7ed3\u679c\u3001\u8d8b\u52bf\u4fe1\u606f\u548c\u62a5\u544a\u5165\u53e3\u3002`
            : "\u5f53\u524d\u6b63\u5728\u67e5\u770b\u8be5\u623f\u5c4b\u7684\u5386\u53f2\u68c0\u6d4b\u8bb0\u5f55\u3002";
        renderHistoryHouseDirectory();
        renderTrendSummary(result.trend_summary);
        renderHistory(result.records || []);
        setVisible("historyPanel", true);
    } catch (error) {
        console.error("Failed to load house detections:", error);
    }
}

async function submitHouseInfo() {
    updateFormValidationState("infoForm");
    const houseNumber = document.getElementById("house_number").value.trim();
    const houseType = getFieldValue("house_type");
    const crackLocation = getFieldValue("crack_location");
    const detectionType = getFieldValue("detection_type");

    if (!houseNumber || !houseType || !crackLocation || !detectionType) {
        focusFirstMissingField("infoForm");
        setStatus("infoStatus", "error", "请完整填写房屋编号、房屋类型、裂缝位置和检测任务。");
        setGlobalStatus("error", "房屋档案信息尚未填写完整。");
        return;
    }

    const data = {
        house_number: houseNumber,
        house_type: houseType,
        crack_location: crackLocation,
        detection_type: detectionType,
    };

    setButtonState("submitInfoButton", true, "提交中...");
    setStatus("infoStatus", "loading", "正在提交房屋档案...");
    setGlobalStatus("loading", "正在建立房屋档案，请稍候。");

    try {
        const response = await fetch("/submit_house_info", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        const result = await parseResponsePayload(response);

        if (response.status === 409 && result.house_id) {
            updateHouseBinding(result.house_id, houseNumber);
            await loadHouseDetections(result.house_id);
            setStatus("infoStatus", "success", `该房屋已存在，已自动绑定到档案 ID ${result.house_id}。`);
            setGlobalStatus("success", `已绑定已有房屋档案 ID ${result.house_id}。`);
            return;
        }

        if (!response.ok) {
            setStatus("infoStatus", "error", result.message || "房屋档案提交失败。");
            setGlobalStatus("error", result.message || "房屋档案提交失败。");
            return;
        }

        updateHouseBinding(result.house_id, houseNumber);
        await loadHouseDetections(result.house_id);
        setStatus("infoStatus", "success", `房屋档案已创建，档案 ID ${result.house_id}。`);
        setGlobalStatus("success", `房屋档案创建成功，后续图像将绑定到房屋 ${houseNumber}。`);
    } catch (error) {
        setStatus("infoStatus", "error", `房屋档案提交失败：${error}`);
        setGlobalStatus("error", `房屋档案提交失败：${error}`);
    } finally {
        setButtonState("submitInfoButton", false, "提交房屋档案");
    }
}

function buildUploadFormData(forceLowQuality = false) {
    const formData = new FormData();
    formData.append("scenario_type", document.getElementById("scenario_type").value);
    formData.append("component_type", document.getElementById("component_type").value);

    BUNDLE_FIELDS.forEach((field) => {
        const input = document.getElementById(field.id);
        if (input.files.length) {
            formData.append(field.id, input.files[0]);
        }
    });

    if (currentHouseId) {
        formData.append("house_id", currentHouseId);
    }
    if (forceLowQuality) {
        formData.append("allow_low_quality", "true");
    }

    return formData;
}

function validateBundleSelection() {
    const missing = BUNDLE_FIELDS.filter((field) => !document.getElementById(field.id).files.length);
    return missing;
}

async function uploadPhoto(forceLowQuality = false) {
    updateFormValidationState("uploadForm");
    const missing = validateBundleSelection();
    if (missing.length) {
        focusFirstMissingField("uploadForm");
        const names = missing.map((field) => field.label).join("、");
        setStatus("uploadStatus", "error", `请先补齐这几张图像：${names}。`);
        setGlobalStatus("error", `当前还缺少 ${names} 图像，无法执行三图成组上传。`);
        return;
    }

    if (mediumTargetPrecheck.status === "loading") {
        setStatus("uploadStatus", "warning", "中景二维码预检仍在进行中，请等待完成后再上传。");
        setGlobalStatus("warning", "中景二维码预检尚未完成。");
        return;
    }

    if (!mediumTargetPrecheck.passed) {
        setStatus("uploadStatus", "error", mediumTargetPrecheck.message || "中景图未通过二维码与四角锚点预检，已阻止上传。");
        setGlobalStatus("error", mediumTargetPrecheck.message || "中景图未通过二维码与四角锚点预检。");
        return;
    }

    const bundleSummary = getBundleSelectionSummary();
    if (bundleSummary.exceedsLimit) {
        setStatus("uploadStatus", "error", `三张图总大小 ${formatBytes(bundleSummary.totalBytes)}，已超过 100MB 上限，请压缩后再上传。`);
        setGlobalStatus("error", "本次批量上传已超过 100MB 上限，前端已阻止提交。");
        updateBundleSizeSummary();
        return;
    }

    setVisible("uploadResult", false);
    setButtonState("uploadButton", true, forceLowQuality ? "强制分析中..." : "批量分析中...");
    setStatus("uploadStatus", "loading", "长景 / 中景 / 近景三张图正在上传并执行批量检测...");
    setGlobalStatus(
        "loading",
        currentHouseId
            ? `正在批量分析三张图像，结果会自动归档到房屋 ID ${currentHouseId}。`
            : "正在批量分析三张图像，当前未绑定房屋档案，结果不会进入历史趋势。"
    );

    try {
        stopTaskPolling();
        const response = await fetch("/tasks/upload-batch", {
            method: "POST",
            body: buildUploadFormData(forceLowQuality),
        });
        const result = await parseResponsePayload(response);

        if (response.status === 422) {
            renderBundleQualityResults(result.results || [], true);
            setStatus("uploadStatus", "warning", result.message || "三张图中存在质量未通过的照片。");
            setGlobalStatus("warning", result.message || "三张图中存在质量未通过的照片。");
            return;
        }

        if (!response.ok) {
            renderBundleQualityResults(result.results || []);
            if (result.results?.length) {
                renderBatchUploadResult(result);
            }
            setStatus("uploadStatus", "error", result.message || "批量上传失败。");
            setGlobalStatus("error", result.message || "批量上传失败。");
            return;
        }

        currentTaskCode = result.task_code || "";
        document.getElementById("taskLookupCode").value = currentTaskCode;
        renderDetectionTaskStatus(result);
        renderBundleQualityResults(result.quality_results || []);
        setStatus("uploadStatus", "loading", localizeTaskMessage(result.message) || "任务已创建，等待执行 / Task queued.");
        setGlobalStatus(
            "loading",
            currentTaskCode
                ? `任务 ${currentTaskCode} 已排队，状态将自动刷新 / Task queued. Status will refresh automatically.`
                : "任务已创建，等待执行 / Task queued."
        );
        if (currentTaskCode) {
            startTaskPolling(currentTaskCode);
        }
        return;
    } catch (error) {
        setStatus("uploadStatus", "error", `批量上传失败：${error}`);
        setGlobalStatus("error", `批量上传失败：${error}`);
    } finally {
        setButtonState("uploadButton", false, "上传三景图并启动分析");
    }
}

async function deleteDetection(recordId) {
    if (!window.confirm("确认删除这条检测记录吗？")) {
        return;
    }

    setGlobalStatus("loading", "正在删除检测记录...");
    try {
        const response = await fetch(`/detections/${recordId}`, { method: "DELETE" });
        const result = await parseResponsePayload(response);
        if (!response.ok) {
            setGlobalStatus("error", result.message || "删除记录失败。");
            return;
        }

        const historyHouseId = activeHistoryHouseId || currentHouseId;
        if (historyHouseId) {
            await loadHouseDetections(historyHouseId);
        }
        await loadSurveySummary();
        setGlobalStatus("success", "检测记录已删除。");
    } catch (error) {
        setGlobalStatus("error", `删除记录失败：${error}`);
    }
}

async function rerunDetection(recordId) {
    setGlobalStatus("loading", "正在重新执行裂缝分析，请稍候...");
    try {
        const response = await fetch(`/detections/${recordId}/rerun`, { method: "POST" });
        const result = await parseResponsePayload(response);
        if (!response.ok) {
            setGlobalStatus("error", result.segmentation_error || result.message || "重新分析失败。");
            return;
        }

        const historyHouseId = activeHistoryHouseId || currentHouseId;
        if (historyHouseId) {
            await loadHouseDetections(historyHouseId);
        }
        await loadSurveySummary();
        setGlobalStatus("success", "重新分析完成，历史记录已刷新。");
    } catch (error) {
        setGlobalStatus("error", `重新分析失败：${error}`);
    }
}

function initializeHistoryPanelCopy() {
    const boardTitle = document.querySelector("#historyHouseBoard .section-subhead strong");
    const boardDescription = document.querySelector("#historyHouseBoard .section-subhead p");
    const currentHint = document.getElementById("historyCurrentHint");

    if (boardTitle) {
        boardTitle.textContent = "历史检测房屋";
    }
    if (boardDescription) {
        boardDescription.textContent =
            "以下房屋已存在检测记录。点击房号可直接打开最新检测报告，点击“查看历史”可展开该房屋的历次检测记录。";
    }
    if (currentHint) {
        currentHint.textContent = "选择房屋后，这里将显示该房屋的历史检测结果、趋势信息和报告入口。";
    }
}

function setCommandNavigationActive(activeHref) {
    document.querySelectorAll(".command-rail-item[href^='#']").forEach((link) => {
        link.classList.toggle("command-rail-item-active", link.getAttribute("href") === activeHref);
    });
    document.querySelectorAll(".platform-nav-item[href^='#']").forEach((link) => {
        link.classList.toggle("platform-nav-item-active", link.getAttribute("href") === activeHref);
    });
}

function initializeCommandNavigation() {
    const links = Array.from(
        document.querySelectorAll(".command-rail-item[href^='#'], .platform-nav-item[href^='#']")
    );
    if (!links.length) {
        return;
    }

    const sections = [];
    const seen = new Set();
    links.forEach((link) => {
        const href = link.getAttribute("href");
        if (!href || seen.has(href)) {
            return;
        }
        const target = document.querySelector(href);
        if (!target) {
            return;
        }
        seen.add(href);
        sections.push({ href, target });
    });

    if (!sections.length) {
        return;
    }

    const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    const setActive = (href) => {
        if (!href) {
            return;
        }
        setCommandNavigationActive(href);
    };

    links.forEach((link) => {
        const href = link.getAttribute("href");
        const section = sections.find((item) => item.href === href);
        if (!section) {
            return;
        }
        link.addEventListener("click", (event) => {
            event.preventDefault();
            setActive(href);
            section.target.scrollIntoView({
                behavior: prefersReducedMotion ? "auto" : "smooth",
                block: "start",
            });
            if (window.history?.replaceState) {
                window.history.replaceState(null, "", href);
            }
        });
    });

    const initialHref =
        sections.find((item) => item.href === window.location.hash)?.href ||
        sections.find((item) => item.target.getBoundingClientRect().top >= 0)?.href ||
        sections[0].href;
    setActive(initialHref);

    if (typeof IntersectionObserver !== "function") {
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];

            if (visible?.target?.id) {
                setActive(`#${visible.target.id}`);
            }
        },
        {
            rootMargin: "-18% 0px -58% 0px",
            threshold: [0.18, 0.35, 0.55],
        }
    );

    sections.forEach(({ target }) => observer.observe(target));
}

async function loadHouseDetections(houseId) {
    if (!houseId) {
        setVisible("historyCurrentHint", false);
        setVisible("historyOverview", false);
        setVisible("trendSummaryCard", false);
        setVisible("historyList", false);
        setVisible("historyPanel", historyHouseDirectory.length > 0);
        return;
    }

    try {
        const response = await fetch(`/house/${houseId}/detections`);
        const result = await response.json();
        if (!response.ok) {
            setVisible("historyCurrentHint", false);
            setVisible("historyOverview", false);
            setVisible("trendSummaryCard", false);
            setVisible("historyList", false);
            setVisible("historyPanel", historyHouseDirectory.length > 0);
            return;
        }

        activeHistoryHouseId = houseId;
        const historyCurrentHint = document.getElementById("historyCurrentHint");
        if (historyCurrentHint) {
            historyCurrentHint.textContent = result.house_number
                ? `当前正在查看房屋 ${result.house_number} 的历史检测结果、趋势信息和报告入口。`
                : "当前正在查看该房屋的历史检测记录。";
        }
        setVisible("historyCurrentHint", true);
        renderHistoryHouseDirectory();
        renderTrendSummary(result.trend_summary);
        renderHistory(result.records || []);
        setVisible("historyPanel", true);
    } catch (error) {
        console.error("Failed to load house detections:", error);
    }
}

arrangeSplitPreviewWorkspace();
updateHouseBinding(null, "");
setDefaultTargetInspectionTime();
updateBundleSizeSummary();
resetMediumTargetPrecheck();
loadTargetSpecs();
initializeHistoryPanelCopy();
initializeCommandNavigation();
loadSurveySummary();
syncTargetHouseNumber(false);
initializeWorkspaceValidation();
document.getElementById("house_number").addEventListener("input", () => syncTargetHouseNumber(false));
document.getElementById("target_crack_size")?.addEventListener("change", () => syncTargetSpecByCrackSize(true));
document.getElementById("target_spec")?.addEventListener("change", updateTargetSpecGuidance);
toggleOtherInput("house_type", { focus: false });
toggleOtherInput("crack_location", { focus: false });
toggleOtherInput("detection_type", { focus: false });
