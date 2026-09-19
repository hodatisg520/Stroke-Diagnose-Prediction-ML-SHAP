
import { useEffect, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  AlertTriangle,
  Check,
  Footprints,
  HeartPulse,
  Info,
  Moon,
  RefreshCw,
  Sparkles,
  User,
  Watch,
} from 'lucide-react';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const DEFAULT_HEALTH_SNAPSHOT = {
  heart_rate: 72,
  systolic_bp: 124,
  diastolic_bp: 79,
  sleep_hours: 7.2,
  daily_steps: 6800,
  active_minutes: 32,
  weight_kg: 72.0,
  source: 'Manual entry',
};

const DEMO_SOURCES = [
  {
    id: 'apple-health',
    name: 'Apple Health',
    description: 'iPhone + Apple Watch',
    snapshot: {
      heart_rate: 72,
      systolic_bp: 124,
      diastolic_bp: 79,
      sleep_hours: 7.4,
      daily_steps: 7500,
      active_minutes: 38,
      weight_kg: 72.4,
    },
    history: [
      { date: '2026-09-16', systolic_bp: 122, diastolic_bp: 78, sleep_hours: 7.1, daily_steps: 6900 },
      { date: '2026-09-17', systolic_bp: 126, diastolic_bp: 80, sleep_hours: 7.6, daily_steps: 8100 },
      { date: '2026-09-18', systolic_bp: 124, diastolic_bp: 79, sleep_hours: 7.4, daily_steps: 7500 },
    ],
  },
  {
    id: 'fitbit',
    name: 'Fitbit',
    description: 'Activity + sleep tracker',
    snapshot: {
      heart_rate: 78,
      systolic_bp: 132,
      diastolic_bp: 84,
      sleep_hours: 6.3,
      daily_steps: 5100,
      active_minutes: 22,
      weight_kg: 76.1,
    },
    history: [
      { date: '2026-09-16', systolic_bp: 130, diastolic_bp: 83, sleep_hours: 6.0, daily_steps: 4800 },
      { date: '2026-09-17', systolic_bp: 134, diastolic_bp: 86, sleep_hours: 6.4, daily_steps: 5300 },
      { date: '2026-09-18', systolic_bp: 132, diastolic_bp: 84, sleep_hours: 6.3, daily_steps: 5100 },
    ],
  },
  {
    id: 'garmin',
    name: 'Garmin',
    description: 'Watch + training metrics',
    snapshot: {
      heart_rate: 68,
      systolic_bp: 118,
      diastolic_bp: 76,
      sleep_hours: 7.8,
      daily_steps: 10400,
      active_minutes: 55,
      weight_kg: 70.3,
    },
    history: [
      { date: '2026-09-16', systolic_bp: 120, diastolic_bp: 77, sleep_hours: 7.5, daily_steps: 9600 },
      { date: '2026-09-17', systolic_bp: 117, diastolic_bp: 75, sleep_hours: 8.0, daily_steps: 11200 },
      { date: '2026-09-18', systolic_bp: 118, diastolic_bp: 76, sleep_hours: 7.8, daily_steps: 10400 },
    ],
  },
];

const METRIC_DEFINITIONS = [
  { key: 'heart_rate', label: 'Heart rate', unit: 'bpm', icon: HeartPulse, format: value => `${Math.round(value)} bpm` },
  { key: 'systolic_bp', label: 'Blood pressure', unit: 'mmHg', icon: Activity, format: (value, snapshot) => `${Math.round(value)}/${Math.round(snapshot.diastolic_bp)} mmHg` },
  { key: 'sleep_hours', label: 'Sleep', unit: 'hours', icon: Moon, format: value => `${Number(value).toFixed(1)} h` },
  { key: 'daily_steps', label: 'Steps today', unit: 'steps', icon: Footprints, format: value => `${Math.round(value).toLocaleString()} steps` },
];

function App() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [connectedSource, setConnectedSource] = useState(null);
  const [syncingSource, setSyncingSource] = useState(null);
  const [healthSnapshot, setHealthSnapshot] = useState(DEFAULT_HEALTH_SNAPSHOT);
  const [healthHistory, setHealthHistory] = useState([]);
  const [aiAdvice, setAiAdvice] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState(null);

  const [formData, setFormData] = useState({
    model_name: 'Voting Ensemble',
    gender: 'Female',
    age: 55,
    hypertension: false,
    heart_disease: false,
    ever_married: true,
    avg_glucose_level: 105.0,
    bmi: 28.5,
    work_type: 'Private',
    Residence_type: 'Urban',
    smoking_status: 'never smoked',
  });

  useEffect(() => {
    axios.get(`${API_URL}/models`)
      .then(res => {
        setModels(res.data.models);
        if (res.data.models.length > 0) {
          setFormData(prev => ({
            ...prev,
            model_name: res.data.models.includes('Voting Ensemble') ? 'Voting Ensemble' : res.data.models[0],
          }));
        }
      })
      .catch(err => {
        console.error('Error fetching models:', err);
        setError('Failed to connect to backend server.');
      });
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : (type === 'range' || type === 'number' ? Number(value) : value),
    }));
  };

  const handleHealthChange = (e) => {
    const { name, value } = e.target;
    setHealthSnapshot(prev => ({
      ...prev,
      [name]: Number(value),
      source: 'Manual entry',
    }));
    setConnectedSource(null);
    setHealthHistory([]);
  };

  const handleSourceConnect = (source) => {
    if (syncingSource) return;
    setSyncingSource(source.id);
    setAiAdvice(null);
    setAiError(null);

    // Simulate the permission/sync delay of a real health platform connection.
    window.setTimeout(() => {
      setHealthSnapshot({ ...source.snapshot, source: `${source.name} demo sync` });
      setHealthHistory(source.history);
      setConnectedSource(source);
      setSyncingSource(null);
    }, 650);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setAiAdvice(null);
    setAiError(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setError('Failed to get prediction. Check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const handleAiDoctor = async () => {
    if (!result) return;
    setAiLoading(true);
    setAiError(null);
    try {
      const res = await axios.post(`${API_URL}/ai-doctor`, {
        patient_profile: formData,
        prediction: result,
        health_snapshot: healthSnapshot,
        history: healthHistory,
        source: connectedSource ? `${connectedSource.name} demo` : 'Manual entry',
      });
      setAiAdvice(res.data);
    } catch (err) {
      console.error(err);
      setAiError('AI Doctor is unavailable. Start the updated backend and try again.');
    } finally {
      setAiLoading(false);
    }
  };

  const getRiskClass = (category) => {
    if (!category) return '';
    if (category.includes('Low')) return 'low';
    if (category.includes('Moderate')) return 'moderate';
    return 'high';
  };

  const formatFeatureName = (name) => {
    const map = {
      age: 'Age',
      avg_glucose_level: 'Glucose Level',
      bmi: 'BMI',
      hypertension: 'Hypertension',
      heart_disease: 'Heart Disease',
      ever_married: 'Ever Married',
      gender: 'Gender',
      work: 'Work Type',
      Residence: 'Residence Area',
      smoking: 'Smoking Status',
    };
    if (map[name]) return map[name];
    if (name.startsWith('work_type_')) return `Work: ${name.replace('work_type_', '')}`;
    if (name.startsWith('Residence_type_')) return `Residence: ${name.replace('Residence_type_', '')}`;
    if (name.startsWith('smoking_status_')) return `Smoking: ${name.replace('smoking_status_', '')}`;
    return name.replaceAll('_', ' ');
  };

  const maxAbsImpact = result?.shap_explanation?.length
    ? Math.max(...result.shap_explanation.map(item => Math.abs(item.impact)))
    : 1;

  return (
    <div className="container animate-fade-in">
      <header className="app-header">
        <h1>
          <Activity size={32} color="var(--accent-main)" />
          STROKEGUARD Health Console
        </h1>
        <p>
          Combine manual clinical information with a simulated health-data stream to explore stroke-risk screening and personalized prevention guidance.
        </p>
      </header>

      <div className="prototype-banner">
        <Info color="#2563eb" size={22} />
        <div>
          <strong>Prototype mode:</strong> the tracker connections below simulate Apple Health, Fitbit, and Garmin sync. They do not request real OAuth permissions or access a real wearable.
        </div>
      </div>

      {models.length === 0 && !error && (
        <div className="initializing-banner">
          <Info color="#2563eb" size={24} />
          <div>
            <strong>System initializing:</strong> the model backend may need 30–50 seconds to wake up. Refresh the page if the diagnostic button remains disabled.
          </div>
        </div>
      )}

      {error && (
        <div className="error-banner">
          <Info color="#b91c1c" />
          <span>{error}</span>
        </div>
      )}

      <section className="glass-panel source-panel">
        <div className="section-heading">
          <div>
            <div className="section-kicker">Connected health data</div>
            <h2>Connect a tracker or health app</h2>
            <p>Choose a simulated source to populate the live snapshot and a three-day trend for the AI Doctor prototype.</p>
          </div>
          <div className={`source-status ${connectedSource ? 'connected' : ''}`}>
            <span className="status-dot"></span>
            {connectedSource ? `${connectedSource.name} synced` : 'Manual entry'}
          </div>
        </div>

        <div className="source-grid">
          {DEMO_SOURCES.map(source => {
            const isConnected = connectedSource?.id === source.id;
            const isSyncing = syncingSource === source.id;
            return (
              <button
                type="button"
                className={`source-card ${isConnected ? 'selected' : ''}`}
                key={source.id}
                onClick={() => handleSourceConnect(source)}
                disabled={Boolean(syncingSource)}
              >
                <span className="source-card-icon"><Watch size={20} /></span>
                <span className="source-card-copy">
                  <strong>{source.name}</strong>
                  <small>{source.description}</small>
                </span>
                <span className="source-card-action">
                  {isSyncing ? <RefreshCw size={16} className="spin" /> : isConnected ? <Check size={18} /> : 'Connect'}
                </span>
              </button>
            );
          })}
        </div>

        <div className="telemetry-grid">
          {METRIC_DEFINITIONS.map(metric => {
            const MetricIcon = metric.icon;
            return (
              <div className="telemetry-card" key={metric.key}>
                <MetricIcon size={17} color="var(--accent-main)" />
                <strong>{metric.format(healthSnapshot[metric.key], healthSnapshot)}</strong>
                <span>{metric.label}</span>
              </div>
            );
          })}
        </div>
        {healthHistory.length > 0 && (
          <div className="trend-panel">
            <div className="trend-heading">
              <div><strong>Recent trend</strong><span>Simulated history from the selected source</span></div>
              <span>3 days</span>
            </div>
            <div className="trend-grid">
              {healthHistory.map(day => (
                <div className="trend-day" key={day.date}>
                  <span className="trend-date">{day.date}</span>
                  <strong>{day.systolic_bp}/{day.diastolic_bp} <small>mmHg</small></strong>
                  <span>{day.sleep_hours} h sleep · {day.daily_steps.toLocaleString()} steps</span>
                </div>
              ))}
            </div>
          </div>
        )}
        <p className="source-note">The current ML model still uses the clinical fields in the manual form. The additional wearable metrics enrich the AI Doctor context and can later support a time-series model.</p>
      </section>

      <div className="dashboard-layout">
        <form className="glass-panel form-panel" onSubmit={handleSubmit}>
          <div className="form-heading">
            <h2><User size={18} /> Manual clinical check</h2>
            <select name="model_name" value={formData.model_name} onChange={handleChange} className="form-control model-select">
              {models.map(model => <option key={model} value={model}>{model}</option>)}
            </select>
          </div>

          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Gender</label>
              <select name="gender" value={formData.gender} onChange={handleChange} className="form-control">
                <option value="Female">Female</option>
                <option value="Male">Male</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label range-label"><span>Age (Years)</span><span className="range-value">{formData.age}</span></label>
              <input type="range" name="age" min="0" max="120" value={formData.age} onChange={handleChange} />
            </div>

            <div className="form-group">
              <label className="form-label">Work Category</label>
              <select name="work_type" value={formData.work_type} onChange={handleChange} className="form-control">
                <option value="Private">Private Sector</option>
                <option value="Self-employed">Self Employed</option>
                <option value="Govt_job">Government</option>
                <option value="children">Children</option>
                <option value="Never_worked">Never Worked</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Residence</label>
              <select name="Residence_type" value={formData.Residence_type} onChange={handleChange} className="form-control">
                <option value="Urban">Urban</option>
                <option value="Rural">Rural</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label range-label"><span>Average Glucose (mg/dL)</span><span className="range-value">{formData.avg_glucose_level}</span></label>
              <input type="range" name="avg_glucose_level" min="50" max="300" step="0.1" value={formData.avg_glucose_level} onChange={handleChange} />
            </div>

            <div className="form-group">
              <label className="form-label range-label"><span>Body Mass Index (BMI)</span><span className="range-value">{formData.bmi}</span></label>
              <input type="range" name="bmi" min="10" max="60" step="0.1" value={formData.bmi} onChange={handleChange} />
            </div>

            <div className="form-group full-width">
              <label className="form-label">Smoking History</label>
              <select name="smoking_status" value={formData.smoking_status} onChange={handleChange} className="form-control">
                <option value="never smoked">Never Smoked</option>
                <option value="formerly smoked">Formerly Smoked</option>
                <option value="smokes">Currently Smokes</option>
                <option value="Unknown">Unknown</option>
              </select>
            </div>

            <div className="form-section-title full-width">
              <span className="section-kicker">Wearable / manual telemetry</span>
              <p>These fields are used by the AI Doctor context in this prototype.</p>
            </div>

            <div className="form-group">
              <label className="form-label">Heart rate (bpm)</label>
              <input className="form-control" type="number" name="heart_rate" min="30" max="220" value={healthSnapshot.heart_rate} onChange={handleHealthChange} />
            </div>
            <div className="form-group">
              <label className="form-label">Systolic BP (mmHg)</label>
              <input className="form-control" type="number" name="systolic_bp" min="60" max="260" value={healthSnapshot.systolic_bp} onChange={handleHealthChange} />
            </div>
            <div className="form-group">
              <label className="form-label">Diastolic BP (mmHg)</label>
              <input className="form-control" type="number" name="diastolic_bp" min="30" max="180" value={healthSnapshot.diastolic_bp} onChange={handleHealthChange} />
            </div>
            <div className="form-group">
              <label className="form-label">Sleep (hours)</label>
              <input className="form-control" type="number" name="sleep_hours" min="0" max="24" step="0.1" value={healthSnapshot.sleep_hours} onChange={handleHealthChange} />
            </div>
            <div className="form-group">
              <label className="form-label">Steps today</label>
              <input className="form-control" type="number" name="daily_steps" min="0" max="100000" value={healthSnapshot.daily_steps} onChange={handleHealthChange} />
            </div>
            <div className="form-group">
              <label className="form-label">Active minutes</label>
              <input className="form-control" type="number" name="active_minutes" min="0" max="1440" value={healthSnapshot.active_minutes} onChange={handleHealthChange} />
            </div>
            <div className="form-group">
              <label className="form-label">Weight (kg)</label>
              <input className="form-control" type="number" name="weight_kg" min="20" max="300" step="0.1" value={healthSnapshot.weight_kg} onChange={handleHealthChange} />
            </div>

            <div className="form-group full-width clinical-history">
              <label className="form-label">Clinical History</label>
              <div className="checkbox-row">
                <label className="custom-checkbox">
                  <input type="checkbox" name="hypertension" checked={formData.hypertension} onChange={handleChange} />
                  <span className="checkmark"></span><span>Hypertension</span>
                </label>
                <label className="custom-checkbox">
                  <input type="checkbox" name="heart_disease" checked={formData.heart_disease} onChange={handleChange} />
                  <span className="checkmark"></span><span>Heart Disease</span>
                </label>
                <label className="custom-checkbox">
                  <input type="checkbox" name="ever_married" checked={formData.ever_married} onChange={handleChange} />
                  <span className="checkmark"></span><span>Ever Married</span>
                </label>
              </div>
            </div>
          </div>

          <div className="form-actions">
            <button type="submit" className="btn-primary" disabled={loading || models.length === 0}>
              {loading ? 'Processing analysis...' : 'Run diagnostic'}
            </button>
          </div>
        </form>

        <div className="glass-panel result-card">
          <h2>Diagnostic result</h2>

          {result ? (
            <div className="animate-fade-in result-content">
              <div className={`risk-display-modern ${getRiskClass(result.risk_category)}`}>
                <div className="risk-value">{(result.stroke_probability * 100).toFixed(1)}<span>%</span></div>
                <div className="risk-label">Screening probability</div>
                <div className={`status-badge ${getRiskClass(result.risk_category)}`}>{result.risk_category}</div>
              </div>

              {result.shap_explanation?.length > 0 && (
                <div className="shap-bar-container">
                  <h3>Model explanation</h3>
                  {result.shap_explanation.map((item, idx) => {
                    const widthPct = (Math.abs(item.impact) / maxAbsImpact) * 100;
                    const isPositive = item.impact > 0;
                    return (
                      <div className="shap-item" key={`${item.feature}-${idx}`}>
                        <div className="shap-label" title={formatFeatureName(item.feature)}>{formatFeatureName(item.feature)}</div>
                        <div className="shap-track"><div className={`shap-fill ${isPositive ? 'pos' : 'neg'}`} style={{ width: `${Math.max(widthPct, 2)}%` }}></div></div>
                      </div>
                    );
                  })}
                  <div className="shap-legend">
                    <span><i className="legend-dot decrease"></i> Decreases risk</span>
                    <span><i className="legend-dot increase"></i> Increases risk</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-result">
              <Info size={40} color="var(--text-muted)" />
              <p>Complete the profile and run the diagnostic to see screening probability and feature impact.</p>
            </div>
          )}
        </div>
      </div>

      {result && (
        <section className="glass-panel ai-doctor-card animate-fade-in">
          <div className="ai-doctor-header">
            <div className="ai-title-wrap">
              <div className="ai-icon"><Sparkles size={23} /></div>
              <div>
                <div className="section-kicker">Personalized prevention assistant</div>
                <h2>AI Doctor <span className="prototype-tag">Prototype</span></h2>
                <p>Combines the model result, the latest manual/device snapshot, and the simulated trend to suggest practical next steps.</p>
              </div>
            </div>
            <button type="button" className="btn-secondary" onClick={handleAiDoctor} disabled={aiLoading}>
              {aiLoading ? <><RefreshCw size={17} className="spin" /> Generating...</> : <><Sparkles size={17} /> Ask AI Doctor</>}
            </button>
          </div>

          {aiError && <div className="ai-error"><AlertTriangle size={18} /> {aiError}</div>}

          {aiAdvice ? (
            <div className="advice-content">
              <div className="advice-meta">
                <span className={`advice-mode ${aiAdvice.mode === 'gemini' ? 'gemini' : 'demo'}`}>
                  {aiAdvice.mode === 'gemini' ? 'Gemini response' : 'Local demo response'}
                </span>
                <span>Source: {aiAdvice.source}</span>
              </div>
              <p className="advice-summary">{aiAdvice.summary}</p>
              {aiAdvice.notice && <div className="demo-notice"><Info size={17} /> {aiAdvice.notice}</div>}

              <div className="advice-grid">
                <div>
                  <h3>Signals to notice</h3>
                  <div className="signal-list">
                    {aiAdvice.risk_signals?.map((signal, index) => (
                      <div className={`signal-card ${signal.severity}`} key={`${signal.title}-${index}`}>
                        <div className="signal-card-heading"><AlertTriangle size={17} /><strong>{signal.title}</strong></div>
                        <p>{signal.message}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3>Small actions</h3>
                  <div className="recommendation-list">
                    {aiAdvice.recommendations?.map((recommendation, index) => (
                      <div className="recommendation-card" key={`${recommendation.title}-${index}`}>
                        <div className="recommendation-number">{index + 1}</div>
                        <div><strong>{recommendation.title}</strong><p>{recommendation.action}</p><small>{recommendation.why}</small></div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="next-steps">
                <h3>Next steps</h3>
                <ul>{aiAdvice.next_steps?.map((step, index) => <li key={`${step}-${index}`}><Check size={16} /> {step}</li>)}</ul>
              </div>
              <p className="medical-disclaimer">{aiAdvice.disclaimer}</p>
            </div>
          ) : (
            <div className="ai-empty-state"><Sparkles size={22} /><p>Ask the prototype AI Doctor to turn this result into understandable signals and habit suggestions.</p></div>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
