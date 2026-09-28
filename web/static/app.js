// ===== 高寒地区道路智能养护系统 - 前端交互 =====

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const btnDetect = document.getElementById('btnDetect');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
let selectedFile = null;

// 上传区域点击
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// 文件选择
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        uploadArea.querySelector('.upload-box p').textContent = `已选择: ${selectedFile.name}`;
        btnDetect.disabled = false;
    }
});

// 拖拽上传
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.querySelector('.upload-box').classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.querySelector('.upload-box').classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.querySelector('.upload-box').classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        selectedFile = e.dataTransfer.files[0];
        uploadArea.querySelector('.upload-box p').textContent = `已选择: ${selectedFile.name}`;
        btnDetect.disabled = false;
    }
});

// 检测按钮
btnDetect.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('temperature', document.getElementById('temperature').value);
    formData.append('humidity', document.getElementById('humidity').value);
    formData.append('freeze_thaw_cycles', document.getElementById('freeze_thaw_cycles').value);
    formData.append('ice_thickness', document.getElementById('ice_thickness').value);
    formData.append('road_age', document.getElementById('road_age').value);
    formData.append('avg_freeze_days', document.getElementById('avg_freeze_days').value);
    formData.append('permafrost_depth', document.getElementById('permafrost_depth').value);
    formData.append('salt_usage', document.getElementById('salt_usage').value);
    formData.append('road_section', document.getElementById('road_section').value);

    // 显示加载
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    btnDetect.disabled = true;

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        loading.style.display = 'none';

        if (data.error) {
            alert('检测失败: ' + data.error);
            btnDetect.disabled = false;
            return;
        }

        renderResults(data);
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        loading.style.display = 'none';
        alert('请求失败: ' + error.message);
    }

    btnDetect.disabled = false;
});

// 渲染结果
function renderResults(data) {
    renderSummary(data.summary);
    renderResultImage(data.result_image);
    renderDiseaseTable(data.detections);
    renderRisk(data.risk_assessment);
    renderPlan(data.maintenance_plan);
    renderRecommendations(data.maintenance_plan.recommendations);
}

// 渲染摘要
function renderSummary(summary) {
    const grid = document.getElementById('summaryGrid');
    const diseases = summary.diseases || {};
    const diseaseList = Object.entries(diseases).map(([k, v]) => `${k}(${v})`).join(' / ') || '无';
    grid.innerHTML = `
        <div class="summary-card">
            <div class="value">${summary.total || 0}</div>
            <div class="label">病害总数</div>
        </div>
        <div class="summary-card">
            <div class="value" style="font-size:18px;">${diseaseList}</div>
            <div class="label">病害类型</div>
        </div>
        <div class="summary-card">
            <div class="value">${summary.coverage || 0}%</div>
            <div class="label">病害覆盖率</div>
        </div>
        <div class="summary-card">
            <div class="value">${summary.max_severity || '无'}</div>
            <div class="label">最大严重程度</div>
        </div>
    `;
}

// 渲染结果图片
function renderResultImage(src) {
    document.getElementById('resultImage').src = src;
}

// 渲染病害详情表
function renderDiseaseTable(detections) {
    const tbody = document.getElementById('diseaseTableBody');
    if (!detections || detections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;">未检测到病害</td></tr>';
        return;
    }
    tbody.innerHTML = detections.map((d, i) => `
        <tr>
            <td>${i + 1}</td>
            <td>${d.disease_name}</td>
            <td>${(d.confidence * 100).toFixed(1)}%</td>
            <td>${(d.area_ratio * 100).toFixed(2)}%</td>
            <td>(${d.bbox[0]},${d.bbox[1]}) - (${d.bbox[2]},${d.bbox[3]})</td>
        </tr>
    `).join('');
}

// 渲染风险评估
function renderRisk(risk) {
    const grid = document.getElementById('riskGrid');
    const items = [
        { title: '冻融循环风险', value: risk.freeze_thaw_risk },
        { title: '冻胀损坏风险', value: risk.frost_damage_risk },
        { title: '冰雪覆盖风险', value: risk.ice_snow_risk },
        { title: '盐冻剥蚀风险', value: risk.salt_erosion_risk },
        { title: '综合风险', value: risk.overall_risk },
        { title: '综合风险评分', value: risk.risk_score + '/100' },
    ];
    grid.innerHTML = items.map(item => `
        <div class="risk-card">
            <div class="risk-title">${item.title}</div>
            <div class="risk-value risk-${item.value}">${item.value}</div>
        </div>
    `).join('');

    // 风险因素列表
    if (risk.risk_factors && risk.risk_factors.length > 0) {
        grid.innerHTML += `
            <div class="risk-card" style="grid-column: 1 / -1;">
                <div class="risk-title">风险因素</div>
                <div style="font-size:14px; color:#4a5568; margin-top:6px;">
                    ${risk.risk_factors.map(f => `<div>▸ ${f}</div>`).join('')}
                </div>
            </div>
        `;
    }
}

// 渲染养护方案
function renderPlan(plan) {
    // 方案摘要
    const summary = document.getElementById('planSummary');
    summary.innerHTML = `
        <div class="plan-card">
            <div class="value">${plan.road_section}</div>
            <div class="label">路段</div>
        </div>
        <div class="plan-card">
            <div class="value">${plan.total_tasks}</div>
            <div class="label">养护任务数</div>
        </div>
        <div class="plan-card">
            <div class="value">${plan.priority_tasks}</div>
            <div class="label">高优先级任务</div>
        </div>
        <div class="plan-card">
            <div class="value">¥${plan.total_cost.toFixed(0)}</div>
            <div class="label">预估总成本</div>
        </div>
        <div class="plan-card">
            <div class="value">${plan.total_duration}</div>
            <div class="label">总工期(天)</div>
        </div>
        <div class="plan-card">
            <div class="value">${plan.overall_risk}</div>
            <div class="label">综合风险等级</div>
        </div>
    `;

    // 任务表
    const tbody = document.getElementById('maintenanceTableBody');
    if (!plan.tasks || plan.tasks.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;">无需养护任务</td></tr>';
        return;
    }
    tbody.innerHTML = plan.tasks.map(t => `
        <tr ${t.urgent ? 'style="background:#fff3cd;"' : ''}>
            <td>${t.task_id}</td>
            <td>${t.disease_name}</td>
            <td><span class="severity-badge severity-${t.severity}">${t.severity}</span></td>
            <td>${t.area}</td>
            <td>${t.method}</td>
            <td class="priority-${t.priority}">${'★'.repeat(t.priority)}(${t.priority})</td>
            <td>¥${t.estimated_cost.toFixed(2)}</td>
            <td>${t.estimated_duration}</td>
            <td style="font-size:13px; color:#e67e22;">${t.cold_region_note || '-'}</td>
        </tr>
    `).join('');
}

// 渲染建议
function renderRecommendations(recommendations) {
    const container = document.getElementById('recommendations');
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<div class="recommendation-item">暂无建议</div>';
        return;
    }
    container.innerHTML = recommendations.map(r => `
        <div class="recommendation-item">${r}</div>
    `).join('');
}
