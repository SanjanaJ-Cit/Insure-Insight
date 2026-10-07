// MediCost AI Frontend Client Logic

let appMeta = null;
let currentPredictionResult = null;
let currentFormData = null;
let shapChartInstance = null;
let activeShapTab = "actionable"; // 'actionable' or 'all'
let simulationDebounceTimer = null;

// Format Currency to Indian Rupee (e.g. ₹27,150)
function formatINR(val) {
  if (val === undefined || val === null || isNaN(val)) return "₹0";
  const num = Math.round(Number(val));
  return "₹" + num.toLocaleString("en-IN");
}

// Format number with commas
function formatNumber(val) {
  if (val === undefined || val === null || isNaN(val)) return "0";
  return Number(val).toLocaleString("en-IN");
}

// Sync slider and badges
function syncWeight(val) {
  document.getElementById("weight-val-badge").innerText = val;
  // Estimate BMI = weight / (1.75^2) as default helper if user hasn't modified it
  const currentBmi = document.getElementById("in_bmi").value;
  if (!currentBmi || Math.abs(parseFloat(currentBmi) - (parseFloat(val) / 2.7)) > 8) {
    document.getElementById("in_bmi").value = (parseFloat(val) / 2.45).toFixed(1);
  }
}

function syncSteps(val) {
  document.getElementById("steps-val-badge").innerText = formatNumber(val);
}

function toggleAdmissionYear(val) {
  const container = document.getElementById("admitted_year_container");
  if (val === "1") {
    container.classList.remove("opacity-40", "pointer-events-none");
  } else {
    container.classList.add("opacity-40", "pointer-events-none");
    document.getElementById("in_Year_last_admitted").value = 0;
  }
}

// Fetch Metadata on startup
async function initApp() {
  try {
    const res = await fetch("/api/meta");
    appMeta = await res.json();
    
    // Populate locations
    const locSelect = document.getElementById("in_Location");
    locSelect.innerHTML = "";
    if (appMeta.categories && appMeta.categories.Location) {
      appMeta.categories.Location.forEach(loc => {
        const opt = document.createElement("option");
        opt.value = loc;
        opt.textContent = loc;
        if (loc === "Mumbai") opt.selected = true;
        locSelect.appendChild(opt);
      });
    }

    // Initialize Lucide icons
    if (window.lucide) {
      lucide.createIcons();
    }
  } catch (err) {
    console.error("Failed to load metadata:", err);
  }
}

// Load Preset Profile
function loadPreset(presetId) {
  if (!appMeta || !appMeta.presets) return;
  const preset = appMeta.presets.find(p => p.id === presetId);
  if (!preset) return;

  const d = preset.data;
  document.getElementById("in_age").value = d.age;
  document.getElementById("in_Gender").value = d.Gender;
  document.getElementById("in_Occupation").value = d.Occupation;
  document.getElementById("in_Location").value = d.Location;
  
  document.getElementById("in_weight").value = d.weight;
  syncWeight(d.weight);

  document.getElementById("in_bmi").value = d.bmi;
  document.getElementById("in_fat_percentage").value = d.fat_percentage;
  document.getElementById("in_avg_glucose_level").value = d.avg_glucose_level;
  document.getElementById("in_cholesterol_level").value = d.cholesterol_level;

  document.getElementById("in_daily_avg_steps").value = d.daily_avg_steps;
  syncSteps(d.daily_avg_steps);

  document.getElementById("in_exercise").value = d.exercise;
  document.getElementById("in_smoking_status").value = d.smoking_status;
  document.getElementById("in_Alcohol").value = d.Alcohol;
  document.getElementById("in_adventure_sports").value = d.adventure_sports;

  document.getElementById("in_heart_decs_history").value = d.heart_decs_history;
  document.getElementById("in_other_major_decs_history").value = d.other_major_decs_history;
  document.getElementById("in_regular_checkup_lasy_year").value = d.regular_checkup_lasy_year;
  document.getElementById("in_visited_doctor_last_1_year").value = d.visited_doctor_last_1_year;
  document.getElementById("in_weight_change_in_last_one_year").value = d.weight_change_in_last_one_year;
  document.getElementById("in_years_of_insurance_with_us").value = d.years_of_insurance_with_us;
  document.getElementById("in_covered_by_any_other_company").value = d.covered_by_any_other_company;

  if (d.Year_last_admitted && d.Year_last_admitted > 0) {
    document.getElementById("in_admitted_flag").value = "1";
    document.getElementById("in_Year_last_admitted").value = d.Year_last_admitted;
    toggleAdmissionYear("1");
  } else {
    document.getElementById("in_admitted_flag").value = "0";
    toggleAdmissionYear("0");
  }

  // Trigger prediction calculation automatically
  document.getElementById("insurance-form").requestSubmit();
}

function resetDefaults() {
  loadPreset("elevated_risk");
}

// Gather form data
function getFormData() {
  const admittedFlag = document.getElementById("in_admitted_flag").value;
  const admittedYear = admittedFlag === "1" ? parseFloat(document.getElementById("in_Year_last_admitted").value) : 0.0;

  return {
    age: parseInt(document.getElementById("in_age").value),
    Gender: document.getElementById("in_Gender").value,
    Occupation: document.getElementById("in_Occupation").value,
    Location: document.getElementById("in_Location").value,
    weight: parseFloat(document.getElementById("in_weight").value),
    bmi: parseFloat(document.getElementById("in_bmi").value),
    fat_percentage: parseFloat(document.getElementById("in_fat_percentage").value),
    daily_avg_steps: parseInt(document.getElementById("in_daily_avg_steps").value),
    exercise: document.getElementById("in_exercise").value,
    smoking_status: document.getElementById("in_smoking_status").value,
    Alcohol: document.getElementById("in_Alcohol").value,
    cholesterol_level: document.getElementById("in_cholesterol_level").value,
    avg_glucose_level: parseFloat(document.getElementById("in_avg_glucose_level").value),
    regular_checkup_lasy_year: parseInt(document.getElementById("in_regular_checkup_lasy_year").value),
    visited_doctor_last_1_year: parseInt(document.getElementById("in_visited_doctor_last_1_year").value),
    weight_change_in_last_one_year: parseFloat(document.getElementById("in_weight_change_in_last_one_year").value),
    years_of_insurance_with_us: parseInt(document.getElementById("in_years_of_insurance_with_us").value),
    covered_by_any_other_company: document.getElementById("in_covered_by_any_other_company").value,
    adventure_sports: parseInt(document.getElementById("in_adventure_sports").value),
    heart_decs_history: parseInt(document.getElementById("in_heart_decs_history").value),
    other_major_decs_history: parseInt(document.getElementById("in_other_major_decs_history").value),
    Year_last_admitted: admittedYear
  };
}

// Handle Form Submit
async function handleFormSubmit(e) {
  e.preventDefault();

  const submitBtn = document.getElementById("submit-btn");
  const originalBtnHtml = submitBtn.innerHTML;
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="inline-block animate-spin mr-2">&#9696;</span> Computing SHAP Explainability...`;

  try {
    currentFormData = getFormData();
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentFormData)
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}: ${await res.text()}`);
    }

    currentPredictionResult = await res.json();
    renderResults(currentPredictionResult);
  } catch (err) {
    alert("Error calculating insurance cost: " + err.message);
    console.error(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalBtnHtml;
    if (window.lucide) lucide.createIcons();
  }
}

// Render Results
function renderResults(data) {
  document.getElementById("placeholder-view").classList.add("hidden");
  const resultsView = document.getElementById("results-view");
  resultsView.classList.remove("hidden");

  // 1. Costs
  document.getElementById("display-cost").innerText = formatINR(data.predicted_cost);
  document.getElementById("display-monthly").innerText = formatINR(data.monthly_cost);

  const diff = data.cost_difference_from_average;
  const diffBadge = document.getElementById("display-diff-badge");
  if (diff > 0) {
    diffBadge.className = "badge badge-danger font-mono";
    diffBadge.innerText = `+${formatINR(diff)} Above Avg`;
  } else {
    diffBadge.className = "badge badge-success font-mono";
    diffBadge.innerText = `${formatINR(diff)} Below Avg (Discount)`;
  }

  // 2. Health Gauge
  const scoreObj = data.health_score;
  document.getElementById("score-val").innerText = scoreObj.score;
  const scoreGrade = document.getElementById("score-grade");
  scoreGrade.innerText = scoreObj.grade;
  scoreGrade.style.color = scoreObj.color;
  scoreGrade.style.borderColor = scoreObj.color + "40";
  scoreGrade.style.backgroundColor = scoreObj.color + "15";

  // Progress circle (circumference = 2 * PI * 40 = 251.2)
  const circle = document.getElementById("gauge-circle");
  const offset = 251.2 - (251.2 * scoreObj.score) / 100;
  circle.style.strokeDashoffset = offset;
  circle.style.stroke = scoreObj.color;

  // 3. Potential Savings Banner
  const comb = data.combined_scenario;
  document.getElementById("display-total-savings").innerText = formatINR(comb.total_potential_savings);
  document.getElementById("display-savings-subtitle").innerText = 
    `Lower your premium by ${comb.total_savings_pct}% by adopting recommended lifestyle habits`;
  document.getElementById("display-optimized-cost").innerText = `${formatINR(comb.optimized_cost)} / yr`;

  // 4. SHAP Factor Breakdown Lists
  renderShapLists(data.shap_breakdown);

  // 5. SHAP Chart
  renderShapChart(data.shap_breakdown);

  // 6. Actionable Recommendations
  renderRecommendations(data.recommendations);

  // 7. Initialize Simulator with current applicant values
  initSimulator(currentFormData, data.predicted_cost);

  if (window.lucide) lucide.createIcons();
}

function renderShapLists(shapData) {
  // Pushers
  const pushersEl = document.getElementById("cost-pushers-list");
  pushersEl.innerHTML = "";
  if (shapData.cost_pushers.length === 0) {
    pushersEl.innerHTML = `<p class="text-slate-400 italic">No significant surcharge drivers detected.</p>`;
  } else {
    shapData.cost_pushers.forEach(item => {
      pushersEl.innerHTML += `
        <div class="flex items-center justify-between p-2 rounded-lg bg-white border border-rose-200/70 shadow-sm">
          <div>
            <div class="font-semibold text-slate-800">${item.name}</div>
            <div class="text-[11px] text-slate-500">${item.summary}</div>
          </div>
          <span class="font-mono font-bold text-rose-600 shrink-0 ml-2">+${formatINR(item.impact_inr)}</span>
        </div>
      `;
    });
  }

  // Savers
  const saversEl = document.getElementById("cost-savers-list");
  saversEl.innerHTML = "";
  if (shapData.cost_savers.length === 0) {
    saversEl.innerHTML = `<p class="text-slate-400 italic">No healthy discount credits currently applied.</p>`;
  } else {
    shapData.cost_savers.forEach(item => {
      saversEl.innerHTML += `
        <div class="flex items-center justify-between p-2 rounded-lg bg-white border border-emerald-200/70 shadow-sm">
          <div>
            <div class="font-semibold text-slate-800">${item.name}</div>
            <div class="text-[11px] text-slate-500">${item.summary}</div>
          </div>
          <span class="font-mono font-bold text-emerald-600 shrink-0 ml-2">-${formatINR(Math.abs(item.impact_inr))}</span>
        </div>
      `;
    });
  }
}

function setShapTab(tab) {
  activeShapTab = tab;
  document.getElementById("tab-actionable").classList.toggle("active", tab === "actionable");
  document.getElementById("tab-all").classList.toggle("active", tab === "all");
  if (currentPredictionResult) {
    renderShapChart(currentPredictionResult.shap_breakdown);
  }
}

function renderShapChart(shapData) {
  const ctx = document.getElementById("shapChart").getContext("2d");
  
  let features = shapData.all_features;
  if (activeShapTab === "actionable") {
    features = features.filter(f => f.is_actionable);
  }
  
  // Pick top 8 features
  features = features.slice(0, 8);
  
  const labels = features.map(f => f.name);
  const values = features.map(f => f.shap_value);
  const backgroundColors = values.map(v => v >= 0 ? "rgba(239, 68, 68, 0.85)" : "rgba(16, 185, 129, 0.85)");
  const borderColors = values.map(v => v >= 0 ? "#dc2626" : "#059669");

  if (shapChartInstance) {
    shapChartInstance.destroy();
  }

  shapChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Impact on Premium (₹ INR)",
        data: values,
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              const val = context.raw;
              const prefix = val >= 0 ? "+₹" : "-₹";
              return ` Impact: ${prefix}${Math.abs(Math.round(val)).toLocaleString("en-IN")}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(226, 232, 240, 0.8)" },
          ticks: {
            callback: function(value) {
              return (value >= 0 ? "+" : "") + "₹" + Math.round(value).toLocaleString("en-IN");
            },
            font: { family: "monospace", size: 10 }
          }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11, weight: "500" } }
        }
      }
    }
  });
}

function renderRecommendations(recs) {
  const container = document.getElementById("recommendations-container");
  container.innerHTML = "";

  if (!recs || recs.length === 0) {
    container.innerHTML = `
      <div class="p-4 bg-emerald-50 border border-emerald-200 rounded-xl text-center text-emerald-800 text-xs">
        🎉 Outstanding! Your health and lifestyle markers are already optimized for minimum health insurance pricing.
      </div>
    `;
    return;
  }

  recs.forEach(rec => {
    container.innerHTML += `
      <div class="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
          <div class="flex items-center space-x-2">
            <span class="badge badge-primary text-[10px] uppercase font-bold">${rec.category}</span>
            <span class="badge badge-warning text-[10px] font-semibold">${rec.badge}</span>
          </div>
          <div class="font-mono font-extrabold text-emerald-600 text-sm flex items-center">
            <i data-lucide="tag" class="w-3.5 h-3.5 mr-1 text-emerald-500"></i>
            <span>Save ${formatINR(rec.savings_inr)} / yr (${rec.savings_pct}%)</span>
          </div>
        </div>
        <h4 class="text-sm font-bold text-slate-800 mb-1">${rec.title}</h4>
        <p class="text-xs text-slate-600 mb-2 leading-relaxed">${rec.advice}</p>
        <div class="flex items-center text-[11px] text-blue-700 font-medium bg-blue-50/70 px-3 py-1.5 rounded-lg border border-blue-100">
          <i data-lucide="arrow-right-circle" class="w-3.5 h-3.5 mr-1.5 shrink-0"></i>
          <span>Action: ${rec.action}</span>
        </div>
      </div>
    `;
  });
}

// Live What-If Simulator
function initSimulator(profile, currentCost) {
  const targetWeight = Math.max(54, Math.round(profile.weight * 0.9));
  document.getElementById("sim_weight").value = targetWeight;
  document.getElementById("sim-weight-badge").innerText = targetWeight;

  const targetSteps = Math.min(11000, Math.max(8000, profile.daily_avg_steps + 3000));
  document.getElementById("sim_steps").value = targetSteps;
  document.getElementById("sim-steps-badge").innerText = formatNumber(targetSteps);

  document.getElementById("sim_smoking").value = "never smoked";
  document.getElementById("sim_exercise").value = "Moderate";

  runLiveSimulation();
}

function runLiveSimulation() {
  const w = parseFloat(document.getElementById("sim_weight").value);
  const s = parseInt(document.getElementById("sim_steps").value);
  const smk = document.getElementById("sim_smoking").value;
  const ex = document.getElementById("sim_exercise").value;

  document.getElementById("sim-weight-badge").innerText = w;
  document.getElementById("sim-steps-badge").innerText = formatNumber(s);

  if (simulationDebounceTimer) clearTimeout(simulationDebounceTimer);
  simulationDebounceTimer = setTimeout(async () => {
    if (!currentFormData) return;

    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile: currentFormData,
          adjustments: {
            weight: w,
            daily_avg_steps: s,
            smoking_status: smk,
            exercise: ex
          }
        })
      });

      if (!res.ok) return;
      const simData = await res.json();

      document.getElementById("sim-display-cost").innerText = formatINR(simData.simulated_cost);
      const savingsEl = document.getElementById("sim-display-savings");
      if (simData.savings_inr >= 0) {
        savingsEl.className = "text-xl font-bold font-mono text-emerald-400 mt-0.5";
        savingsEl.innerText = `Save ${formatINR(simData.savings_inr)} / yr (${simData.savings_pct}%)`;
      } else {
        savingsEl.className = "text-xl font-bold font-mono text-rose-400 mt-0.5";
        savingsEl.innerText = `+${formatINR(Math.abs(simData.savings_inr))} Premium Increase`;
      }
    } catch (e) {
      console.error("Simulation error:", e);
    }
  }, 120);
}

// Kick off
window.addEventListener("DOMContentLoaded", () => {
  initApp();
  // Auto-run high-risk profile demo on initial page load
  setTimeout(() => {
    loadPreset("elevated_risk");
  }, 300);
});
