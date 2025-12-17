// Wait for DOM to load
document.addEventListener("DOMContentLoaded", function() {
    fetchDashboardData();
    // Refresh data every 30 seconds (optional)
    // setInterval(fetchDashboardData, 30000);
});

function fetchDashboardData() {
    fetch('/api/dashboard-data')
        .then(response => response.json())
        .then(data => {
            updateCards(data.cards);
            initDailyChart(data.chart_daily);
            initQualityChart(data.chart_quality);
            updateActivityList(data.recent_activity);
            updatePatientBadge(data.current_patient_id);
        })
        .catch(error => console.error('Error loading dashboard data:', error));
}

function updateCards(cards) {
    document.getElementById('total-scans').innerText = cards.total_scans;
    document.getElementById('today-scans').innerText = cards.today_scans;
    document.getElementById('avg-clarity').innerText = cards.avg_clarity;
    document.getElementById('pending-review').innerText = cards.pending_review;
}

function updatePatientBadge(id) {
    document.getElementById('current-patient-badge').innerText = "Viewing: " + id;
}

function updateActivityList(activities) {
    const list = document.getElementById('activity-list');
    list.innerHTML = ''; // Clear existing
    activities.forEach(act => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center px-0';
        li.innerHTML = `
            <div>
                <span class="fw-bold">${act.event}</span><br>
                <small class="text-muted">${act.time}</small>
            </div>
            <span class="badge rounded-pill bg-light text-dark">${act.status}</span>
        `;
        list.appendChild(li);
    });
}

// Image switcher function
function changeImage(route) {
    fetch(route, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        if(data.success) {
            // Refresh image and re-fetch data to update patient ID badge
            const img = document.getElementById('dashboard-display');
            img.src = "/dashboard_image?" + new Date().getTime(); // Force cache refresh
            fetchDashboardData(); 
        }
    });
}

// --- CHART CONFIGURATIONS ---
let dailyChartInstance = null;
let qualityChartInstance = null;

function initDailyChart(chartData) {
    const ctx = document.getElementById('dailyChart').getContext('2d');
    if (dailyChartInstance) dailyChartInstance.destroy(); // Prevent duplicate charts

    dailyChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: 'Scans Processed',
                data: chartData.data,
                backgroundColor: 'rgba(67, 167, 197, 0.1)',
                borderColor: '#43a7c5',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: '#f0f0f0' } },
                x: { grid: { display: false } }
            }
        }
    });
}

function initQualityChart(chartData) {
    const ctx = document.getElementById('qualityChart').getContext('2d');
    if (qualityChartInstance) qualityChartInstance.destroy();

    qualityChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartData.labels,
            datasets: [{
                data: chartData.data,
                backgroundColor: ['#43a7c5', '#f7c35f', '#e76c6c'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            cutout: '70%',
            plugins: {
                legend: { position: 'bottom', labels: { usePointStyle: true, padding: 20 } }
            }
        }
    });
}