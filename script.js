// script.js - Versi Final dengan Semua Fungsi Maksimal

const CSV_FILE = 'data.csv';

// Variabel Global
let data = [];
let modelLinear = { m: 0, c: 0, r2: 0, mse: 0, mae: 0, rmse: 0, maxError: 0 };
let modelPoly = { c0: 0, c1: 0, c2: 0, r2: 0, mse: 0, mae: 0, rmse: 0, maxError: 0 };
let avgSuhu = 0;
let chartInstances = { linear: null, poly: null, residual: null };
let showPolyResidual = false;
let errorStats = { mean: 0, std: 0, maxPos: 0, maxNeg: 0 };

// =================================================================
// 1. FUNGSI LSM (Least Squares Method)
// =================================================================

function linearRegression(x, y) {
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }

    const m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const c = (sumY - m * sumX) / n;

    return { m, c };
}

function polynomialRegression(x, y) {
    const n = x.length;
    
    let sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
    let sumY = 0, sumXY = 0, sumX2Y = 0;

    for (let i = 0; i < n; i++) {
        const xi = x[i];
        const xi2 = xi * xi;
        const xi3 = xi2 * xi;
        const xi4 = xi2 * xi2;
        const yi = y[i];

        sumX += xi;
        sumX2 += xi2;
        sumX3 += xi3;
        sumX4 += xi4;
        
        sumY += yi;
        sumXY += xi * yi;
        sumX2Y += xi2 * yi;
    }

    const A = [
        [n, sumX, sumX2],
        [sumX, sumX2, sumX3],
        [sumX2, sumX3, sumX4]
    ];
    
    const B = [sumY, sumXY, sumX2Y];
    
    return solveLinearSystem(A, B);
}

function solveLinearSystem(A, B) {
    const n = B.length;
    const augmented = A.map((row, i) => [...row, B[i]]);
    
    // Eliminasi Gauss
    for (let i = 0; i < n; i++) {
        // Pivot
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }
        
        [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
        
        // Eliminasi
        for (let k = i + 1; k < n; k++) {
            const factor = augmented[k][i] / augmented[i][i];
            for (let j = i; j <= n; j++) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }
    
    // Substitusi mundur
    const coeff = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
        coeff[i] = augmented[i][n];
        for (let j = i + 1; j < n; j++) {
            coeff[i] -= augmented[i][j] * coeff[j];
        }
        coeff[i] /= augmented[i][i];
    }
    
    return coeff;
}

function calculateMetrics(actual, predicted) {
    const n = actual.length;
    let sumSST = 0;
    let sumSSE = 0;
    let sumAE = 0;
    let maxError = 0;
    let sumError = 0;
    let sumSqError = 0;
    
    const meanActual = actual.reduce((a, b) => a + b, 0) / n;

    for (let i = 0; i < n; i++) {
        const error = actual[i] - predicted[i];
        sumSST += (actual[i] - meanActual) ** 2;
        sumSSE += error ** 2;
        sumAE += Math.abs(error);
        maxError = Math.max(maxError, Math.abs(error));
        sumError += error;
        sumSqError += error * error;
    }

    const r2 = Math.max(0, 1 - (sumSSE / sumSST));
    const mae = sumAE / n;
    const mse = sumSSE / n;
    const rmse = Math.sqrt(mse);
    const meanError = sumError / n;
    const stdError = Math.sqrt(sumSqError / n - meanError * meanError);

    return { r2, mae, mse, rmse, maxError, meanError, stdError };
}

// =================================================================
// 2. FUNGSI UTAMA (INIT)
// =================================================================

async function initDashboard() {
    try {
        const response = await fetch(CSV_FILE);
        const text = await response.text();
        
        // Parsing CSV
        const rows = text.trim().split('\n').slice(1);
        const rhData = [];
        const tempData = [];
        data = rows.map(row => {
            const [tanggal, kelembabanStr, suhuStr] = row.split(',');
            const kelembaban = parseFloat(kelembabanStr);
            const suhu = parseFloat(suhuStr);
            rhData.push(kelembaban);
            tempData.push(suhu);
            return { tanggal, kelembaban, suhu };
        });

        // Hitung statistik kelembaban
        const avgHum = rhData.reduce((a, b) => a + b, 0) / rhData.length;
        const minHum = Math.min(...rhData);
        const maxHum = Math.max(...rhData);
        
        // Update statistik kelembaban
        document.getElementById('avgHum').textContent = avgHum.toFixed(1) + '%';
        document.getElementById('rangeHum').textContent = minHum + '% - ' + maxHum + '%';

        avgSuhu = tempData.reduce((a, b) => a + b, 0) / data.length;
        document.getElementById('dataCount').textContent = data.length;

        // Pelatihan Model
        const linearCoeff = linearRegression(rhData, tempData);
        const linearPred = rhData.map(rh => linearCoeff.m * rh + linearCoeff.c);
        const linearMetrics = calculateMetrics(tempData, linearPred);
        modelLinear = { ...linearCoeff, ...linearMetrics };

        const polyCoeff = polynomialRegression(rhData, tempData);
        const polyPred = rhData.map(rh => 
            polyCoeff[2] * rh * rh + polyCoeff[1] * rh + polyCoeff[0]
        );
        const polyMetrics = calculateMetrics(tempData, polyPred);
        modelPoly = { 
            c0: polyCoeff[0], 
            c1: polyCoeff[1], 
            c2: polyCoeff[2], 
            ...polyMetrics 
        };

        // Update Tampilan
        updateEquations();
        updateAccuracyTable();
        updateSampleTable(linearPred, tempData);
        
        // Inisialisasi chart
        initChartLinear(rhData, tempData);
        initChartPoly(rhData, tempData);
        initChartResidual(rhData, tempData, linearPred, polyPred);
        
        updatePrediction(document.getElementById('humSlider').value);

        console.log("Dashboard berhasil dimuat!");
        console.log("Model Linear:", modelLinear);
        console.log("Model Polynomial:", modelPoly);

    } catch (error) {
        console.error("Gagal memuat data:", error);
        alert("Gagal memuat data. Pastikan file data.csv ada di folder yang sama dan formatnya benar.");
    }
}

// =================================================================
// 3. FUNGSI UPDATE TAMPILAN
// =================================================================

function updateEquations() {
    const m = modelLinear.m.toFixed(4);
    const c = modelLinear.c.toFixed(4);
    const r2Lin = (modelLinear.r2 * 100).toFixed(1);
    document.getElementById('linearEq').innerHTML = 
        `<b>LINEAR:</b> T = ${m} × RH + ${c} &nbsp;&nbsp;(R² = ${modelLinear.r2.toFixed(3)} | ${r2Lin}%)`;
    
    const c2 = modelPoly.c2.toFixed(6);
    const c1 = modelPoly.c1.toFixed(4);
    const c0 = modelPoly.c0.toFixed(3);
    const r2Poly = (modelPoly.r2 * 100).toFixed(1);
    document.getElementById('polyEq').innerHTML = 
        `<b>POLYNOMIAL (deg 2):</b> T = ${c2}RH² ${c1 > 0 ? '+' : ''} ${c1}RH + ${c0} &nbsp;&nbsp;(R² = ${modelPoly.r2.toFixed(3)} | ${r2Poly}%)`;

    document.getElementById('mCoeff').textContent = `${m}°C`;
    document.getElementById('rSqLinear').textContent = `${r2Lin}%`;
}

function updateAccuracyTable() {
    const tableBody = document.querySelector('#accuracyTable tbody');
    tableBody.innerHTML = `
        <tr>
            <td>R-squared (R²)</td>
            <td>${modelLinear.r2.toFixed(4)}</td>
            <td>${modelPoly.r2.toFixed(4)}</td>
        </tr>
        <tr>
            <td>Mean Absolute Error (MAE)</td>
            <td>${modelLinear.mae.toFixed(3)}°C</td>
            <td>${modelPoly.mae.toFixed(3)}°C</td>
        </tr>
        <tr>
            <td>Mean Squared Error (MSE)</td>
            <td>${modelLinear.mse.toFixed(3)}</td>
            <td>${modelPoly.mse.toFixed(3)}</td>
        </tr>
        <tr>
            <td>Root MSE (RMSE)</td>
            <td>${modelLinear.rmse.toFixed(3)}°C</td>
            <td>${modelPoly.rmse.toFixed(3)}°C</td>
        </tr>
        <tr>
            <td>Max Error</td>
            <td>${modelLinear.maxError.toFixed(3)}°C</td>
            <td>${modelPoly.maxError.toFixed(3)}°C</td>
        </tr>
    `;
}

function updateSampleTable(linearPred, tempData) {
    const tableBody = document.querySelector('#sampleTable tbody');
    tableBody.innerHTML = '';
    
    // Menampilkan 5 data pertama dan terakhir
    const indices = [0, 1, Math.floor(data.length/2), data.length-2, data.length-1];
    
    indices.forEach(idx => {
        const d = data[idx];
        const pred = linearPred[idx];
        const error = tempData[idx] - pred;
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${d.tanggal}</td>
            <td>${d.kelembaban}%</td>
            <td>${d.suhu.toFixed(1)}°C</td>
            <td>${pred.toFixed(1)}°C</td>
            <td class="${error > 0 ? 'error-positive' : error < 0 ? 'error-negative' : 'error-neutral'}">
                ${error > 0 ? '+' : ''}${error.toFixed(1)}°C
            </td>
        `;
        tableBody.appendChild(row);
    });
}

function updatePrediction(rh) {
    const rhVal = parseFloat(rh);
    document.getElementById('humVal').innerText = rhVal;
    document.getElementById('currentRH').innerText = rhVal;

    // Prediksi Linear
    const predLin = modelLinear.m * rhVal + modelLinear.c;
    const errorLin = predLin - avgSuhu;
    document.getElementById('predLinear').innerText = predLin.toFixed(1) + '°C';
    document.getElementById('errorLinear').innerText = `${errorLin > 0 ? '+' : ''}${errorLin.toFixed(2)}°C`;
    document.getElementById('errorLinear').className = errorLin > 0.5 ? 'error-positive' : errorLin < -0.5 ? 'error-negative' : 'error-neutral';
    const statusLinear = Math.abs(errorLin) < 0.5 ? '✅ AKURAT' : Math.abs(errorLin) < 1.0 ? '⚠️ NETRAL' : '🔴 PERHATIAN';
    document.getElementById('statusLinear').innerText = statusLinear;
    document.getElementById('statusLinear').className = Math.abs(errorLin) < 0.5 ? 'status-badge akurasi' : 
                                                        Math.abs(errorLin) < 1.0 ? 'status-badge netral' : 'status-badge perhatian';

    // Prediksi Polinomial
    const predPoly = modelPoly.c2 * rhVal * rhVal + modelPoly.c1 * rhVal + modelPoly.c0;
    const errorPoly = predPoly - avgSuhu;
    document.getElementById('predPoly').innerText = predPoly.toFixed(1) + '°C';
    document.getElementById('errorPoly').innerText = `${errorPoly > 0 ? '+' : ''}${errorPoly.toFixed(2)}°C`;
    document.getElementById('errorPoly').className = errorPoly > 0.5 ? 'error-positive' : errorPoly < -0.5 ? 'error-negative' : 'error-neutral';
    const statusPoly = Math.abs(errorPoly) < 0.5 ? '✅ AKURAT' : Math.abs(errorPoly) < 1.0 ? '⚠️ NETRAL' : '🔴 PERHATIAN';
    document.getElementById('statusPoly').innerText = statusPoly;
    document.getElementById('statusPoly').className = Math.abs(errorPoly) < 0.5 ? 'status-badge akurasi' : 
                                                       Math.abs(errorPoly) < 1.0 ? 'status-badge netral' : 'status-badge perhatian';
    
    // Rata-rata
    document.getElementById('predAvg').innerText = avgSuhu.toFixed(1) + '°C';
    document.getElementById('errorAvg').innerText = '±0.00°C';
    document.getElementById('statusAvg').innerText = '⚠️ NETRAL';

    updatePredictionPointer(rhVal, predLin, predPoly);
}

// =================================================================
// 4. FUNGSI CHART.JS DENGAN GARIS NOL
// =================================================================

function initChartLinear(rhData, tempData) {
    if (chartInstances.linear) chartInstances.linear.destroy();
    
    const ctx = document.getElementById('chartLinear').getContext('2d');
    const dataPoints = rhData.map((rh, i) => ({ x: rh, y: tempData[i] }));
    
    const minRh = Math.min(...rhData);
    const maxRh = Math.max(...rhData);
    const linearLine = [];
    for (let rh = minRh; rh <= maxRh; rh += 0.5) {
        linearLine.push({ x: rh, y: modelLinear.m * rh + modelLinear.c });
    }
    
    chartInstances.linear = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Data Aktual BMKG',
                data: dataPoints,
                backgroundColor: 'rgba(229, 115, 115, 0.8)',
                borderColor: '#e53935',
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Regresi Linear',
                data: linearLine,
                type: 'line',
                borderColor: '#2196f3',
                backgroundColor: 'transparent',
                borderWidth: 3,
                pointRadius: 0,
                fill: false,
                tension: 0
            }]
        },
        options: getChartOptions('Kelembaban (%)', 'Suhu (°C)')
    });
}

function initChartPoly(rhData, tempData) {
    if (chartInstances.poly) chartInstances.poly.destroy();
    
    const ctx = document.getElementById('chartPoly').getContext('2d');
    const dataPoints = rhData.map((rh, i) => ({ x: rh, y: tempData[i] }));
    
    const minRh = Math.min(...rhData);
    const maxRh = Math.max(...rhData);
    const polyCurve = [];
    for (let rh = minRh; rh <= maxRh; rh += 0.5) {
        polyCurve.push({ 
            x: rh, 
            y: modelPoly.c2 * rh * rh + modelPoly.c1 * rh + modelPoly.c0 
        });
    }
    
    chartInstances.poly = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Data Aktual BMKG',
                data: dataPoints,
                backgroundColor: 'rgba(76, 175, 80, 0.8)',
                borderColor: '#43a047',
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Regresi Polinomial',
                data: polyCurve,
                type: 'line',
                borderColor: '#9c27b0',
                backgroundColor: 'transparent',
                borderWidth: 3,
                pointRadius: 0,
                fill: false,
                tension: 0.3
            }]
        },
        options: getChartOptions('Kelembaban (%)', 'Suhu (°C)')
    });
}

function initChartResidual(rhData, tempData, linearPred, polyPred) {
    if (chartInstances.residual) chartInstances.residual.destroy();
    
    const ctx = document.getElementById('chartResidual').getContext('2d');
    
    // Siapkan data residual
    const residualData = rhData.map((rh, i) => {
        const error = tempData[i] - linearPred[i];
        return { 
            x: rh, 
            y: error,
            // Tentukan bentuk point berdasarkan error
            pointStyle: error > 0 ? 'triangle' : error < 0 ? 'triangle' : 'circle',
            rotation: error > 0 ? 0 : 180 // Putar segitiga untuk positif/negatif
        };
    });
    
    // Hitung statistik error
    calculateErrorStats(residualData);
    
    // Garis nol (y = 0)
    const zeroLineData = [
        { x: Math.min(...rhData), y: 0 },
        { x: Math.max(...rhData), y: 0 }
    ];
    
    chartInstances.residual = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Residual Error Linear',
                data: residualData,
                backgroundColor: (context) => {
                    const error = context.raw.y;
                    return error > 0 ? 'rgba(76, 175, 80, 0.8)' : 
                           error < 0 ? 'rgba(244, 67, 54, 0.8)' : 'rgba(33, 150, 243, 0.8)';
                },
                borderColor: (context) => {
                    const error = context.raw.y;
                    return error > 0 ? '#388e3c' : 
                           error < 0 ? '#d32f2f' : '#1565c0';
                },
                pointRadius: 6,
                pointHoverRadius: 8,
                pointStyle: (context) => context.raw.pointStyle || 'circle',
                rotation: (context) => context.raw.rotation || 0
            }, {
                label: 'Garis Nol (y=0)',
                data: zeroLineData,
                type: 'line',
                borderColor: 'rgba(0, 0, 0, 0.7)',
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
                tension: 0
            }]
        },
        options: getResidualChartOptions()
    });
}

function calculateErrorStats(residualData) {
    const errors = residualData.map(d => d.y);
    const mean = errors.reduce((a, b) => a + b, 0) / errors.length;
    const std = Math.sqrt(errors.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / errors.length);
    const maxPos = Math.max(...errors.filter(e => e > 0));
    const maxNeg = Math.min(...errors.filter(e => e < 0));
    
    errorStats = { mean, std, maxPos, maxNeg };
    
    // Update statistik di UI
    document.getElementById('meanError').textContent = mean.toFixed(2) + '°C';
    document.getElementById('stdError').textContent = std.toFixed(2) + '°C';
    document.getElementById('maxPosError').textContent = '+' + maxPos.toFixed(2) + '°C';
    document.getElementById('maxNegError').textContent = maxNeg.toFixed(2) + '°C';
}

function getChartOptions(xLabel, yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { 
                position: 'top',
                labels: {
                    color: '#333',
                    font: { size: 12, family: 'Arial' }
                }
            },
            tooltip: {
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: '#fff',
                bodyColor: '#fff',
                callbacks: {
                    label: function(context) {
                        if (context.datasetIndex === 0) {
                            return `Suhu: ${context.parsed.y.toFixed(1)}°C, RH: ${context.parsed.x}%`;
                        }
                        return `Prediksi: ${context.parsed.y.toFixed(1)}°C`;
                    }
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: { 
                    display: true, 
                    text: xLabel,
                    color: '#333',
                    font: { size: 13, weight: 'bold', family: 'Arial' }
                },
                grid: { 
                    color: 'rgba(0, 0, 0, 0.1)',
                    drawBorder: true
                },
                ticks: { 
                    color: '#333',
                    font: { size: 11, family: 'Arial' }
                }
            },
            y: {
                title: { 
                    display: true, 
                    text: yLabel,
                    color: '#333',
                    font: { size: 13, weight: 'bold', family: 'Arial' }
                },
                grid: { 
                    color: 'rgba(0, 0, 0, 0.1)',
                    drawBorder: true
                },
                ticks: { 
                    color: '#333',
                    font: { size: 11, family: 'Arial' }
                }
            }
        },
        animation: {
            duration: 1000,
            easing: 'easeOutQuart'
        }
    };
}

function getResidualChartOptions() {
    const options = getChartOptions('Kelembaban (%)', 'Error Residual (°C)');
    
    // Custom untuk residual plot
    options.scales.y.ticks.callback = function(value) {
        return value + '°C';
    };
    
    options.plugins.tooltip.callbacks.label = function(context) {
        const error = context.parsed.y;
        const type = error > 0 ? 'Positif' : error < 0 ? 'Negatif' : 'Nol';
        return `Error ${type}: ${error.toFixed(2)}°C, RH: ${context.parsed.x}%`;
    };
    
    return options;
}

function toggleResidualType() {
    showPolyResidual = !showPolyResidual;
    
    const rhData = data.map(d => d.kelembaban);
    const tempData = data.map(d => d.suhu);
    
    // Hitung prediksi
    const linearPred = rhData.map(rh => modelLinear.m * rh + modelLinear.c);
    const polyPred = rhData.map(rh => modelPoly.c2 * rh * rh + modelPoly.c1 * rh + modelPoly.c0);
    
    // Update data chart residual
    if (chartInstances.residual) {
        const residualData = rhData.map((rh, i) => {
            const error = showPolyResidual ? tempData[i] - polyPred[i] : tempData[i] - linearPred[i];
            return { 
                x: rh, 
                y: error,
                pointStyle: error > 0 ? 'triangle' : error < 0 ? 'triangle' : 'circle',
                rotation: error > 0 ? 0 : 180
            };
        });
        
        // Update dataset
        chartInstances.residual.data.datasets[0].data = residualData;
        chartInstances.residual.data.datasets[0].label = showPolyResidual ? 
            'Residual Error Polynomial' : 'Residual Error Linear';
        
        // Hitung dan update statistik baru
        calculateErrorStats(residualData);
        
        chartInstances.residual.update();
        
        // Update tombol
        const button = document.querySelector('button[onclick="toggleResidualType()"]');
        button.textContent = showPolyResidual ? '🔄 Tampilkan Error Linear' : '🔄 Tampilkan Error Poly';
        button.title = showPolyResidual ? 'Klik untuk melihat error model linear' : 'Klik untuk melihat error model polynomial';
    }
}

function updatePredictionPointer(rh, predLin, predPoly) {
    // Update chart linear
    if (chartInstances.linear) {
        const linearChart = chartInstances.linear;
        
        // Hapus dataset prediksi lama
        linearChart.data.datasets = linearChart.data.datasets.filter(ds => 
            !ds.id || ds.id !== 'prediction'
        );
        
        // Tambahkan dataset baru
        linearChart.data.datasets.push({
            id: 'prediction',
            label: 'Prediksi Saat Ini',
            data: [{ x: rh, y: predLin }],
            backgroundColor: '#ff9800',
            borderColor: '#ff5722',
            pointStyle: 'star',
            pointRadius: 10,
            pointHoverRadius: 12,
            borderWidth: 2
        });
        
        linearChart.update();
    }
    
    // Update chart polynomial
    if (chartInstances.poly) {
        const polyChart = chartInstances.poly;
        
        // Hapus dataset prediksi lama
        polyChart.data.datasets = polyChart.data.datasets.filter(ds => 
            !ds.id || ds.id !== 'prediction'
        );
        
        // Tambahkan dataset baru
        polyChart.data.datasets.push({
            id: 'prediction',
            label: 'Prediksi Saat Ini',
            data: [{ x: rh, y: predPoly }],
            backgroundColor: '#ff9800',
            borderColor: '#ff5722',
            pointStyle: 'star',
            pointRadius: 10,
            pointHoverRadius: 12,
            borderWidth: 2
        });
        
        polyChart.update();
    }
}

// =================================================================
// 5. FUNGSI TAMBAHAN
// =================================================================

function resetSlider() {
    const slider = document.getElementById('humSlider');
    slider.value = 85;
    updatePrediction(85);
    
    // Reset tombol toggle jika perlu
    if (showPolyResidual) {
        toggleResidualType(); // Kembali ke linear
    }
}

function downloadCSV() {
    if (!data.length) {
        alert("Data belum dimuat!");
        return;
    }
    
    let csvContent = "Tanggal,Kelembaban(%),Suhu_Aktual(°C),Prediksi_Linear(°C),Prediksi_Polynomial(°C),Error_Linear,Error_Polynomial,Status_Linear,Status_Polynomial\n";
    
    data.forEach((row) => {
        const predLin = modelLinear.m * row.kelembaban + modelLinear.c;
        const predPoly = modelPoly.c2 * row.kelembaban * row.kelembaban + modelPoly.c1 * row.kelembaban + modelPoly.c0;
        const errorLin = row.suhu - predLin;
        const errorPoly = row.suhu - predPoly;
        const statusLin = Math.abs(errorLin) < 0.5 ? 'AKURAT' : Math.abs(errorLin) < 1.0 ? 'NETRAL' : 'PERHATIAN';
        const statusPoly = Math.abs(errorPoly) < 0.5 ? 'AKURAT' : Math.abs(errorPoly) < 1.0 ? 'NETRAL' : 'PERHATIAN';
        
        csvContent += `${row.tanggal},${row.kelembaban},${row.suhu},${predLin.toFixed(2)},${predPoly.toFixed(2)},${errorLin.toFixed(2)},${errorPoly.toFixed(2)},${statusLin},${statusPoly}\n`;
    });
    
    // Tambahkan metadata
    csvContent += `\n# METADATA\n`;
    csvContent += `# Model Linear, T = ${modelLinear.m.toFixed(4)} × RH + ${modelLinear.c.toFixed(4)}\n`;
    csvContent += `# Model Polynomial, T = ${modelPoly.c2.toFixed(6)}RH² ${modelPoly.c1 > 0 ? '+' : ''}${modelPoly.c1.toFixed(4)}RH + ${modelPoly.c0.toFixed(3)}\n`;
    csvContent += `# R² Linear,${modelLinear.r2.toFixed(4)}\n`;
    csvContent += `# R² Polynomial,${modelPoly.r2.toFixed(4)}\n`;
    csvContent += `# MAE Linear,${modelLinear.mae.toFixed(3)}\n`;
    csvContent += `# MAE Polynomial,${modelPoly.mae.toFixed(3)}\n`;
    csvContent += `# Dibuat pada,${new Date().toLocaleString()}\n`;
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `prediksi_suhu_bmkg_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    alert("File CSV berhasil diunduh!");
}

function showErrorDistribution() {
    if (!chartInstances.residual) return;
    
    const ctx = document.getElementById('chartResidual').getContext('2d');
    const currentData = chartInstances.residual.data.datasets[0].data;
    
    // Hitung distribusi error
    const errors = currentData.map(d => d.y);
    const bins = 10;
    const minError = Math.min(...errors);
    const maxError = Math.max(...errors);
    const binSize = (maxError - minError) / bins;
    
    const histogram = new Array(bins).fill(0);
    errors.forEach(error => {
        const binIndex = Math.min(Math.floor((error - minError) / binSize), bins - 1);
        histogram[binIndex]++;
    });
    
    // Tampilkan alert dengan distribusi
    let distributionText = "📊 DISTRIBUSI ERROR:\n\n";
    histogram.forEach((count, i) => {
        const binStart = minError + i * binSize;
        const binEnd = binStart + binSize;
        const bar = '█'.repeat(Math.ceil(count / errors.length * 50));
        distributionText += `${binStart.toFixed(1)}°C - ${binEnd.toFixed(1)}°C: ${count} data ${bar}\n`;
    });
    
    distributionText += `\nTotal Data: ${errors.length}`;
    distributionText += `\nMean Error: ${errorStats.mean.toFixed(2)}°C`;
    distributionText += `\nStd Dev: ${errorStats.std.toFixed(2)}°C`;
    
    alert(distributionText);
}

// Validasi semua fungsi saat load
function validateAllFunctions() {
    console.log("🔄 Validasi semua fungsi...");
    
    // Validasi data
    if (data.length === 0) {
        console.error("❌ Data tidak dimuat");
        return false;
    }
    
    // Validasi model
    if (modelLinear.m === 0 || modelPoly.c2 === 0) {
        console.error("❌ Model tidak terinisialisasi");
        return false;
    }
    
    // Validasi chart
    if (!chartInstances.linear || !chartInstances.poly || !chartInstances.residual) {
        console.error("❌ Chart tidak terinisialisasi");
        return false;
    }
    
    console.log("✅ Semua fungsi berjalan normal");
    return true;
}

// Jalankan dashboard
document.addEventListener('DOMContentLoaded', () => {
    initDashboard().then(() => {
        setTimeout(() => {
            validateAllFunctions();
        }, 1000);
    });
});