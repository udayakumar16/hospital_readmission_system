const form = document.getElementById('predict-form');
const resultCard = document.getElementById('result-card');
const probabilityEl = document.getElementById('probability');
const riskBadgeEl = document.getElementById('risk-badge');
const recommendationEl = document.getElementById('recommendation');
const metaEl = document.getElementById('meta');
const demoBtn = document.getElementById('demo-btn');

function setBadge(risk) {
  riskBadgeEl.classList.remove('low', 'medium', 'high');
  const r = (risk || '').toUpperCase();
  if (r === 'LOW') riskBadgeEl.classList.add('low');
  if (r === 'MEDIUM') riskBadgeEl.classList.add('medium');
  if (r === 'HIGH') riskBadgeEl.classList.add('high');
  riskBadgeEl.textContent = r || '--';
}

function formToPayload() {
  const fd = new FormData(form);
  const obj = Object.fromEntries(fd.entries());

  // Cast numeric fields (backend also coerces, but this helps cleanliness)
  const numericFields = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
  ];

  for (const key of numericFields) {
    if (obj[key] === '') continue;
    obj[key] = Number(obj[key]);
  }

  // Trim text fields
  ['diag_1', 'diag_2', 'diag_3'].forEach((k) => {
    if (typeof obj[k] === 'string') obj[k] = obj[k].trim();
    if (obj[k] === '') obj[k] = null;
  });

  return obj;
}

async function predict(payload) {
  const btn = form.querySelector('button[type="submit"]');
  btn.disabled = true;
  btn.textContent = 'Predicting...';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error || 'Prediction failed');
    }

    resultCard.hidden = false;
    probabilityEl.textContent = `${data.probability_percent}%`;
    setBadge(data.risk);
    recommendationEl.textContent = data.recommendation;
    metaEl.textContent = `Scored at ${data.created_at}`;
  } catch (err) {
    resultCard.hidden = false;
    probabilityEl.textContent = '--';
    setBadge('');
    recommendationEl.textContent = err?.message || String(err);
    metaEl.textContent = '';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict Risk';
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  predict(formToPayload());
});

// A quick preset that often pushes higher risk based on utilization patterns
// (This is just a demo; real clinical decisions should use validated workflows.)
demoBtn.addEventListener('click', () => {
  // This preset is taken from a top-scoring row in the dataset for the currently trained model.
  // It should usually produce a HIGH risk score (p > 0.60).
  document.getElementById('race').value = 'Caucasian';
  document.getElementById('gender').value = 'Female';
  document.getElementById('age').value = '[60-70)';

  document.getElementById('time_in_hospital').value = 3;
  document.getElementById('num_lab_procedures').value = 41;
  document.getElementById('num_procedures').value = 2;
  document.getElementById('num_medications').value = 16;

  document.getElementById('number_outpatient').value = 3;
  document.getElementById('number_emergency').value = 0;
  document.getElementById('number_inpatient').value = 15;
  document.getElementById('number_diagnoses').value = 6;

  document.getElementById('diag_1').value = 'V58';
  document.getElementById('diag_2').value = '202';
  document.getElementById('diag_3').value = '250';

  // A1Cresult is unknown in this example
  document.getElementById('A1Cresult').value = '';
  document.getElementById('insulin').value = 'No';
  document.getElementById('change').value = 'No';
  document.getElementById('diabetesMed').value = 'Yes';

  document.getElementById('admission_type_id').value = 3;
  document.getElementById('discharge_disposition_id').value = 1;
  document.getElementById('admission_source_id').value = 2;
});
