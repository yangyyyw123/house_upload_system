let currentHouseId = null;
let currentHouseNumber = "";

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

function showResultPanel(show) {
    document.getElementById("uploadResult").classList.toggle("hidden", !show);
}

function showHistoryPanel(show) {
    document.getElementById("historyPanel").classList.toggle("hidden", !show);
}

function fillPreview(imageId, linkId, url) {
    const image = document.getElementById(imageId);
    const link = document.getElementById(linkId);
    image.src = url;
    link.href = url;
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
    const text = houseId
        ? `当前已绑定房屋：ID ${houseId} / 编号 ${houseNumber}`
        : "尚未绑定房屋信息。建议先提交房屋信息，再上传图片，便于归档和历史追踪。";
    document.getElementById("houseBindingText").textContent = text;
}

function previewSelectedImage() {
    const fileInput = document.getElementById("photo");
    const previewCard = document.getElementById("selectedPreviewCard");
    const previewImage = document.getElementById("selectedPreview");
    const previewName = document.getElementById("selectedFileName");

    if (!fileInput.files.length) {
        previewCard.classList.add("hidden");
        previewImage.removeAttribute("src");
        previewName.textContent = "尚未选择图片";
        return;
    }

    const file = fileInput.files[0];
    previewName.textContent = file.name;
    previewCard.classList.remove("hidden");
    previewImage.src = URL.createObjectURL(file);
}

function renderUploadResult(data) {
    if (!data.segmentation) {
        showResultPanel(false);
        return;
    }

    fillPreview("sourcePreview", "sourceLink", data.file_url);
    fillPreview("maskPreview", "maskLink", data.segmentation.mask_url);
    fillPreview("overlayPreview", "overlayLink", data.segmentation.overlay_url);

    document.getElementById("crackAreaRatio").textContent = data.segmentation.crack_area_ratio;
    document.getElementById("crackPixelCount").textContent = data.segmentation.crack_pixel_count;
    document.getElementById("deviceName").textContent = data.segmentation.device;
    document.getElementById("patchCount").textContent = data.segmentation.patch_count;

    showResultPanel(true);
}

function renderHistory(records) {
    const historyList = document.getElementById("historyList");
    historyList.innerHTML = "";

    if (!records.length) {
        historyList.innerHTML = '<p class="history-empty">当前房屋还没有检测历史。</p>';
        showHistoryPanel(true);
        return;
    }

    records.forEach((record) => {
        const item = document.createElement("article");
        item.className = "history-item";
        item.innerHTML = `
            <div class="history-header">
                <strong>${record.original_filename}</strong>
                <span>${record.created_at || "未知时间"}</span>
            </div>
            <div class="history-meta">
                <span>裂缝占比：${record.crack_area_ratio ?? "-"}</span>
                <span>像素数：${record.crack_pixel_count ?? "-"}</span>
                <span>设备：${record.inference_device ?? "-"}</span>
            </div>
            <div class="history-links">
                <a href="${record.source_image_url}" target="_blank" rel="noopener noreferrer">原图</a>
                <a href="${record.mask_url}" target="_blank" rel="noopener noreferrer">掩码</a>
                <a href="${record.overlay_url}" target="_blank" rel="noopener noreferrer">叠加图</a>
            </div>
            <div class="history-actions">
                <button type="button" class="secondary-button" onclick="rerunDetection(${record.id})">重新检测</button>
                <button type="button" class="danger-button" onclick="deleteDetection(${record.id})">删除记录</button>
            </div>
        `;
        historyList.appendChild(item);
    });

    showHistoryPanel(true);
}

async function loadHouseDetections(houseId) {
    if (!houseId) {
        showHistoryPanel(false);
        return;
    }

    try {
        const response = await fetch(`/house/${houseId}/detections`);
        const result = await response.json();

        if (!response.ok) {
            showHistoryPanel(false);
            return;
        }

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

    if (!houseNumber) {
        setStatus("infoStatus", "error", "请填写房屋编号");
        setGlobalStatus("error", "请填写房屋编号");
        return;
    }

    if (!houseType) {
        setStatus("infoStatus", "error", "请选择或填写房屋类型");
        setGlobalStatus("error", "请选择或填写房屋类型");
        return;
    }

    if (!crackLocation) {
        setStatus("infoStatus", "error", "请选择或填写裂缝位置");
        setGlobalStatus("error", "请选择或填写裂缝位置");
        return;
    }

    if (!detectionType) {
        setStatus("infoStatus", "error", "请选择或填写检测类型");
        setGlobalStatus("error", "请选择或填写检测类型");
        return;
    }

    const data = {
        house_number: houseNumber,
        house_type: houseType,
        crack_location: crackLocation,
        detection_type: detectionType
    };

    setButtonState("submitInfoButton", true, "提交中...");
    setStatus("infoStatus", "loading", "正在提交房屋信息...");
    setGlobalStatus("loading", "正在提交房屋信息，一般会在几秒内返回结果。");

    try {
        const response = await fetch("/submit_house_info", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();

        if (response.status === 409 && result.house_id) {
            updateHouseBinding(result.house_id, houseNumber);
            await loadHouseDetections(result.house_id);
            setStatus("infoStatus", "success", `房屋编号已存在，已自动绑定到 house_id: ${result.house_id}`);
            setGlobalStatus("success", `房屋编号已存在，已绑定到现有 house_id: ${result.house_id}`);
            return;
        }

        if (!response.ok) {
            setStatus("infoStatus", "error", result.message || "提交房屋信息失败");
            setGlobalStatus("error", result.message || "提交房屋信息失败");
            return;
        }

        updateHouseBinding(result.house_id, houseNumber);
        await loadHouseDetections(result.house_id);
        setStatus("infoStatus", "success", `房屋信息提交成功，house_id: ${result.house_id}`);
        setGlobalStatus("success", `房屋信息提交成功，house_id: ${result.house_id}`);
    } catch (error) {
        setStatus("infoStatus", "error", `提交房屋信息失败: ${error}`);
        setGlobalStatus("error", `提交房屋信息失败: ${error}`);
    } finally {
        setButtonState("submitInfoButton", false, "提交房屋信息");
    }
}

async function uploadPhoto() {
    const formData = new FormData();
    const fileInput = document.getElementById("photo");

    if (!fileInput.files.length) {
        setStatus("uploadStatus", "error", "请先选择图片");
        setGlobalStatus("error", "请先选择图片后再上传。");
        return;
    }

    formData.append("file", fileInput.files[0]);

    if (currentHouseId) {
        formData.append("house_id", currentHouseId);
    }

    showResultPanel(false);
    setButtonState("uploadButton", true, "处理中...");
    setStatus("uploadStatus", "loading", "图片已上传，正在执行裂缝检测，请不要刷新页面...");
    setGlobalStatus(
        "loading",
        currentHouseId
            ? `裂缝检测已开始，结果将自动绑定到房屋 ID ${currentHouseId}。CPU 模式下大图通常需要 1 到 2 分钟。`
            : "裂缝检测已开始。当前未绑定房屋信息，结果只会保存在文件系统。CPU 模式下大图通常需要 1 到 2 分钟。"
    );

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });
        const result = await response.json();

        if (!response.ok) {
            setStatus("uploadStatus", "error", result.segmentation_error || result.message || "裂缝检测失败");
            setGlobalStatus("error", result.segmentation_error || result.message || "裂缝检测失败");
            return;
        }

        renderUploadResult(result);
        if (currentHouseId) {
            await loadHouseDetections(currentHouseId);
        }
        setStatus("uploadStatus", "success", "裂缝检测完成，结果已显示在下方。");
        setGlobalStatus("success", "裂缝检测完成，结果已显示在页面下方。");
    } catch (error) {
        setStatus("uploadStatus", "error", `上传失败: ${error}`);
        setGlobalStatus("error", `上传失败: ${error}`);
    } finally {
        setButtonState("uploadButton", false, "上传图片");
    }
}

async function deleteDetection(recordId) {
    if (!confirm("确认删除这条检测记录吗？")) {
        return;
    }

    setGlobalStatus("loading", "正在删除检测记录...");

    try {
        const response = await fetch(`/detections/${recordId}`, {
            method: "DELETE"
        });
        const result = await response.json();

        if (!response.ok) {
            setGlobalStatus("error", result.message || "删除检测记录失败");
            return;
        }

        if (currentHouseId) {
            await loadHouseDetections(currentHouseId);
        }
        setGlobalStatus("success", "检测记录已删除。");
    } catch (error) {
        setGlobalStatus("error", `删除检测记录失败: ${error}`);
    }
}

async function rerunDetection(recordId) {
    setGlobalStatus("loading", "正在重新执行裂缝检测，请等待...");

    try {
        const response = await fetch(`/detections/${recordId}/rerun`, {
            method: "POST"
        });
        const result = await response.json();

        if (!response.ok) {
            setGlobalStatus("error", result.segmentation_error || result.message || "重新检测失败");
            return;
        }

        if (currentHouseId) {
            await loadHouseDetections(currentHouseId);
        }
        setGlobalStatus("success", "重新检测完成，历史记录已更新。");
    } catch (error) {
        setGlobalStatus("error", `重新检测失败: ${error}`);
    }
}
