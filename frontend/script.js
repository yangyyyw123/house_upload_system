let currentHouseId = null;
let currentHouseNumber = "";
const previewUrls = new Map();

const BUNDLE_FIELDS = [
    { id: "long_view", label: "长景" },
    { id: "medium_view", label: "中景" },
    { id: "close_view", label: "近景" },
];
const MAX_BATCH_UPLOAD_BYTES = 100 * 1024 * 1024;
const TARGET_SPEC_FALLBACKS = [
    {
        key: "a4_qr80",
        label: "A4 / 80mm 主靶标",
        description: "推荐。优先保证中景照片可识别，适合常规巡检与裂缝精细测量。",
        recommended: true,
    },
    {
        key: "a4_qr60",
        label: "A4 / 60mm 紧凑靶标",
        description: "打印空间较小时使用。",
        recommended: false,
    },
    {
        key: "a5_qr60",
        label: "A5 / 60mm 便携靶标",
        description: "便于现场随身携带，建议近距离拍摄。",
        recommended: false,
    },
];
let mediumTargetPrecheck = {
    status: "idle",
    passed: false,
    message: "等待选择中景图像后进行二维码与四角锚点预检。",
};

function setStatus(elementId, kind, text) {
    const element = document.getElementById(elementId);
    element.className = `status status-${kind}`;
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

function toggleOtherInput(fieldId) {
    const select = document.getElementById(fieldId);
    const otherInput = document.getElementById(`${fieldId}_other`);
    const useOther = select.value === "其他";
    otherInput.classList.toggle("hidden-field", !useOther);
    otherInput.disabled = !useOther;
    otherInput.required = useOther;
    if (!useOther) {
        otherInput.value = "";
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
        : "尚未绑定房屋信息。提交房屋档案后，后续图像会自动归档到对应房屋。";
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
}

function renderTargetSpecOptions(specs) {
    const select = document.getElementById("target_spec");
    if (!select) {
        return;
    }

    select.innerHTML = specs
        .map((spec) => {
            const badge = spec.recommended ? "（推荐）" : "";
            return `<option value="${escapeHtml(spec.key)}">${escapeHtml(spec.label)}${badge}</option>`;
        })
        .join("");
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
        { label: "靶标规格", value: spec.label || "-" },
        { label: "二维码边长", value: spec.qr_size_mm ? `${spec.qr_size_mm} mm` : "-" },
        { label: "外框尺寸", value: spec.frame_size_mm ? `${spec.frame_size_mm} mm` : "-" },
    ];

    container.innerHTML = `
        <div class="target-preview-card">
            ${payload.preview_url ? `<img src="${escapeHtml(payload.preview_url)}" alt="二维码靶标预览">` : ""}
        </div>
        <div class="target-meta-card">
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
            <p class="target-meta-text">编码内容已写入二维码与 JSON 清单。建议优先下载 PDF 打印，PNG 用于预览和电子转发。</p>
            <div class="target-link-row">
                ${files.pdf_url ? `<a class="button-link" href="${escapeHtml(files.pdf_url)}" target="_blank" rel="noopener noreferrer">下载 PDF</a>` : ""}
                ${files.png_url ? `<a class="button-link" href="${escapeHtml(files.png_url)}" target="_blank" rel="noopener noreferrer">下载 PNG</a>` : ""}
                ${files.manifest_url ? `<a class="button-link" href="${escapeHtml(files.manifest_url)}" target="_blank" rel="noopener noreferrer">查看 JSON</a>` : ""}
            </div>
        </div>
    `;
    setVisible("targetResult", true);
}

async function generateTarget() {
    const houseNumber = document.getElementById("target_house_number").value.trim();
    const inspectionRegion = document.getElementById("target_region").value.trim();
    const sceneType = document.getElementById("target_scene_type").value;
    const specKey = document.getElementById("target_spec").value;

    if (!houseNumber || !inspectionRegion || !sceneType || !specKey) {
        setStatus("targetStatus", "error", "请完整填写房屋编号、检测区域、场景类型和靶标规格。");
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
                house_number: houseNumber,
                inspection_region: inspectionRegion,
                scene_type: sceneType,
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
        setStatus("targetStatus", "success", "二维码靶标已生成，可直接下载 PNG / PDF。");
        setGlobalStatus("success", "二维码靶标已生成。建议先打印靶标，再进行中景与近景拍摄。");
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
                <span class="stage-kicker">识别阶段</span>
                <strong>${escapeHtml(recognition.status || "-")}</strong>
                <p>${escapeHtml(recognition.summary || "暂无识别阶段说明。")}</p>
                <span class="stage-meta">${escapeHtml(recognition.physical_scale || "未建立物理比例")}</span>
            </article>
            <article class="stage-card">
                <span class="stage-kicker">量化阶段</span>
                <strong>${escapeHtml(quantification.status || "-")}</strong>
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

function buildResultMediaCard(title, url, emptyText) {
    return `
        <article class="result-card">
            <h3>${escapeHtml(title)}</h3>
            ${url ? `<img src="${escapeHtml(url)}" alt="${escapeHtml(title)}">` : `<div class="result-placeholder">${escapeHtml(emptyText)}</div>`}
            ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">打开图像</a>` : ""}
        </article>
    `;
}

function buildResultCard(item) {
    const quality = item.quality_report || {};
    const risk = item.risk_assessment || {};
    const markerDetection = item.marker_detection || {};
    const quantification = item.quantification || {};
    const analysisStages = item.analysis_stages || {};
    const reportLink = item.report_url
        ? `<a class="button-link compact-link" href="${escapeHtml(item.report_url)}" target="_blank" rel="noopener noreferrer">下载报告</a>`
        : "";
    return `
        <article class="batch-result-card">
            <div class="batch-result-head">
                <div>
                    <strong>${escapeHtml(item.capture_scale || "未标注")}</strong>
                    <p>${escapeHtml(item.original_filename || "")}</p>
                </div>
                <span class="quality-badge ${getQualityBadgeClass(quality.status)}">${escapeHtml(quality.status || "未知")}</span>
            </div>
            <div class="summary-grid batch-summary-grid">
                <article class="summary-card">
                    <span>检测编号</span>
                    <strong>${escapeHtml(item.detection_code || "-")}</strong>
                </article>
                <article class="summary-card">
                    <span>风险等级</span>
                    <strong>${escapeHtml(risk.risk_level || "未生成")}</strong>
                </article>
                <article class="summary-card">
                    <span>质量状态</span>
                    <strong>${escapeHtml(quality.status || "-")} / ${escapeHtml(quality.score ?? "-")}</strong>
                </article>
                <article class="summary-card">
                    <span>裂缝占比</span>
                    <strong>${escapeHtml(item.segmentation?.crack_area_ratio ?? "-")}</strong>
                </article>
            </div>
            <div class="result-notes">
                <article class="note-card">
                    <h3>质量说明</h3>
                    <p>${escapeHtml(quality.summary || item.message || "-")}</p>
                </article>
                <article class="note-card">
                    <h3>风险说明</h3>
                    <p>${escapeHtml(risk.risk_summary || item.segmentation_error || item.message || "-")}</p>
                </article>
            </div>
            ${buildStageCards(analysisStages)}
            <article class="note-card quant-note-card">
                <h3>量化阶段说明</h3>
                <p>${escapeHtml(quantification.message || "当前暂无量化说明。")}</p>
            </article>
            ${buildQuantificationTable(markerDetection, quantification)}
            <div class="report-actions">
                ${reportLink}
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
        </article>
    `;
}

function renderBatchUploadResult(payload) {
    const results = payload.results || [];
    const list = document.getElementById("batchResultList");
    list.innerHTML = results.map(buildResultCard).join("");
    const batchBundleActions = document.getElementById("batchBundleActions");
    if (payload.bundle_report_url) {
        batchBundleActions.innerHTML = `
            <a class="button-link" href="${escapeHtml(payload.bundle_report_url)}" target="_blank" rel="noopener noreferrer">
                下载三景总报告
            </a>
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

    if (!records.length) {
        historyList.innerHTML = '<p class="history-empty">当前房屋还没有检测记录。</p>';
        setVisible("historyPanel", true);
        return;
    }

    records.forEach((record) => {
        const item = document.createElement("article");
        item.className = "history-item";
        const markerDetection = record.marker_detection || {};
        const quantification = record.quantification || {};
        const analysisStages = record.analysis_stages || {};
        const warnings = (record.quality_warnings || [])
            .map((warning) => `<li>${escapeHtml(warning)}</li>`)
            .join("");

        item.innerHTML = `
            <div class="history-header">
                <div>
                    <strong>${escapeHtml(record.detection_code || record.original_filename)}</strong>
                    <p>${escapeHtml(record.original_filename || "")}</p>
                </div>
                <span>${escapeHtml(record.created_at || "未知时间")}</span>
            </div>
            <div class="history-meta">
                <span>尺度：${escapeHtml(record.capture_scale || "-")}</span>
                <span>构件：${escapeHtml(record.component_type || "-")}</span>
                <span>场景：${escapeHtml(record.scenario_type || "-")}</span>
                <span>质量：${escapeHtml(record.quality_status || "-")} / ${escapeHtml(record.quality_score ?? "-")}</span>
                <span>风险：${escapeHtml(record.risk_level || "-")}</span>
                <span>裂缝占比：${escapeHtml(record.crack_area_ratio ?? "-")}</span>
            </div>
            ${buildStageCards(analysisStages)}
            ${buildQuantificationTable(markerDetection, quantification)}
            <p class="history-summary">${escapeHtml(record.risk_summary || "暂无风险说明。")}</p>
            <p class="history-summary">${escapeHtml(quantification.message || "暂无量化说明。")}</p>
            <p class="history-summary">${escapeHtml(record.recommendation || "暂无处置建议。")}</p>
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
        historyList.appendChild(item);
    });

    setVisible("historyPanel", true);
}

function renderSurveySummary(summary) {
    const cards = document.getElementById("surveySummaryCards");
    const riskList = document.getElementById("surveyRiskList");
    const distribution = summary.risk_distribution || {};
    cards.innerHTML = `
        <article class="summary-card">
            <span>检测记录</span>
            <strong>${escapeHtml(summary.completed_records ?? 0)}</strong>
        </article>
        <article class="summary-card">
            <span>房屋数量</span>
            <strong>${escapeHtml(summary.house_count ?? 0)}</strong>
        </article>
        <article class="summary-card">
            <span>中高风险</span>
            <strong>${escapeHtml((distribution["中风险"] || 0) + (distribution["高风险"] || 0))}</strong>
        </article>
        <article class="summary-card">
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
                        <span>${escapeHtml(item.created_at || "未知时间")}</span>
                    </div>
                    <div class="history-meta">
                        <span>风险：${escapeHtml(item.risk_level || "-")}</span>
                        <span>构件：${escapeHtml(item.component_type || "-")}</span>
                        <span>场景：${escapeHtml(item.scenario_type || "-")}</span>
                        <span>最大宽度：${item.max_width_mm !== null && item.max_width_mm !== undefined ? `${escapeHtml(Number(item.max_width_mm).toFixed(3))} mm` : "-"}</span>
                        <span>裂缝长度：${item.crack_length_mm !== null && item.crack_length_mm !== undefined ? `${escapeHtml(Number(item.crack_length_mm).toFixed(3))} mm` : "-"}</span>
                    </div>
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
        setVisible("historyPanel", false);
        return;
    }

    try {
        const response = await fetch(`/house/${houseId}/detections`);
        const result = await response.json();
        if (!response.ok) {
            setVisible("historyPanel", false);
            return;
        }

        renderTrendSummary(result.trend_summary);
        renderHistory(result.records || []);
    } catch (error) {
        console.error("Failed to load house detections:", error);
    }
}

async function submitHouseInfo() {
    const houseNumber = document.getElementById("house_number").value.trim();
    const houseType = getFieldValue("house_type");
    const crackLocation = getFieldValue("crack_location");
    const detectionType = getFieldValue("detection_type");

    if (!houseNumber || !houseType || !crackLocation || !detectionType) {
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
    const missing = validateBundleSelection();
    if (missing.length) {
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
        const response = await fetch("/upload-batch", {
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

        renderBundleQualityResults(result.results || []);
        renderBatchUploadResult(result);
        if (currentHouseId) {
            await loadHouseDetections(currentHouseId);
        }
        setStatus("uploadStatus", "success", "三张图像检测完成，结果已更新到页面下方。");
        setGlobalStatus("success", "长景 / 中景 / 近景三张图像均已完成批量检测。");
    } catch (error) {
        setStatus("uploadStatus", "error", `批量上传失败：${error}`);
        setGlobalStatus("error", `批量上传失败：${error}`);
    } finally {
        setButtonState("uploadButton", false, "上传长景/中景/近景并分析");
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

        if (currentHouseId) {
            await loadHouseDetections(currentHouseId);
        }
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

        if (currentHouseId) {
            await loadHouseDetections(currentHouseId);
        }
        setGlobalStatus("success", "重新分析完成，历史记录已刷新。");
    } catch (error) {
        setGlobalStatus("error", `重新分析失败：${error}`);
    }
}

updateHouseBinding(null, "");
updateBundleSizeSummary();
resetMediumTargetPrecheck();
loadTargetSpecs();
loadSurveySummary();
syncTargetHouseNumber(false);
document.getElementById("house_number").addEventListener("input", () => syncTargetHouseNumber(false));
