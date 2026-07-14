const failureCanvas = document.getElementById("failureChart");

if (failureCanvas) {
    new Chart(failureCanvas, {
        type: "line",
        data: {
            labels: ["Jun 1", "Jun 5", "Jun 10", "Jun 15", "Jun 20", "Jun 25", "Jun 30"],
            datasets: [{
                label: "Failure Trend",
                data: [15, 32, 80, 95, 60, 42, 20],
                borderColor: "#38bdf8",
                backgroundColor: "rgba(56,189,248,0.2)",
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            plugins: { legend: { labels: { color: "white" } } },
            scales: {
                x: { ticks: { color: "#cbd5e1" } },
                y: { ticks: { color: "#cbd5e1" } }
            }
        }
    });
}

const healthCanvas = document.getElementById("healthChart");

if (healthCanvas) {
    new Chart(healthCanvas, {
        type: "doughnut",
        data: {
            labels: ["Healthy", "At Risk", "Failed"],
            datasets: [{
                data: [9660, 279, 61],
                backgroundColor: ["#22c55e", "#facc15", "#ef4444"]
            }]
        }
    });
}

const riskCanvas = document.getElementById("riskChart");

if (riskCanvas) {
    new Chart(riskCanvas, {
        type: "bar",
        data: {
            labels: ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            datasets: [{
                label: "Machines",
                data: [6128, 2361, 944, 409, 158],
                backgroundColor: ["#22c55e", "#84cc16", "#facc15", "#f97316", "#ef4444"]
            }]
        }
    });
}