import { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, User, Info, Check } from 'lucide-react';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
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
    smoking_status: 'never smoked'
  });

  useEffect(() => {
    axios.get(`${API_URL}/models`)
      .then(res => {
        setModels(res.data.models);
        if (res.data.models.length > 0) {
          setFormData(prev => ({ ...prev, model_name: res.data.models.includes('Voting Ensemble') ? 'Voting Ensemble' : res.data.models[0] }));
        }
      })
      .catch(err => {
        console.error("Error fetching models:", err);
        setError("Failed to connect to backend server.");
      });
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : (type === 'range' || type === 'number' ? Number(value) : value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setError("Failed to get prediction. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  const getRiskClass = (category) => {
    if (!category) return '';
    if (category.includes('Low')) return 'low';
    if (category.includes('Moderate')) return 'moderate';
    return 'high';
  };

  // Process SHAP for display
  const maxAbsImpact = result?.shap_explanation 
    ? Math.max(...result.shap_explanation.map(s => Math.abs(s.impact))) 
    : 1;

  const formatFeatureName = (name) => {
    const map = {
      'age': 'Age',
      'avg_glucose_level': 'Glucose Level',
      'bmi': 'BMI',
      'hypertension': 'Hypertension',
      'heart_disease': 'Heart Disease',
      'ever_married': 'Ever Married',
      'work': 'Work Type',
      'Residence': 'Residence Area',
      'smoking': 'Smoking Status',
      'gender': 'Gender'
    };
    return map[name] || name;
  };

  return (
    <div className="container animate-fade-in">
      <header style={{ marginBottom: '2.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1.5rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Activity size={32} color="var(--accent-main)" /> 
          Clinical Risk Assessment
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1rem', marginTop: '0.5rem' }}>
          Predictive analytics platform for stroke probability estimation and feature impact interpretation.
        </p>
      </header>

      {models.length === 0 && !error && (
        <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', color: '#1e3a8a', padding: '1rem 1.5rem', borderRadius: '12px', marginBottom: '2rem', display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
          <Info color="#2563eb" size={24} style={{ flexShrink: 0 }} />
          <div style={{ fontSize: '0.95rem', lineHeight: 1.5 }}>
            <strong>System Initializing:</strong> The AI backend is hosted on a free cloud service and may take 30-50 seconds to wake up from sleep mode. If the "Run Diagnostic" button is disabled, please wait a moment and <strong>refresh the page</strong>.
          </div>
        </div>
      )}

      {error && (
        <div className="error-banner">
          <Info color="#b91c1c" />
          <span>{error}</span>
        </div>
      )}

      <div className="dashboard-layout">
        {/* Form Panel */}
        <form className="glass-panel" style={{ padding: '2.5rem' }} onSubmit={handleSubmit}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
            <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.125rem' }}>
              <User size={18} /> Patient Parameters
            </h2>
            
            <select name="model_name" value={formData.model_name} onChange={handleChange} className="form-control" style={{ width: 'auto', padding: '0.375rem 2rem 0.375rem 0.75rem', fontSize: '0.85rem' }}>
              {models.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
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
              <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Age (Years)</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{formData.age}</span>
              </label>
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
              <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Average Glucose (mg/dL)</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{formData.avg_glucose_level}</span>
              </label>
              <input type="range" name="avg_glucose_level" min="50" max="300" step="0.1" value={formData.avg_glucose_level} onChange={handleChange} />
            </div>

            <div className="form-group">
              <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Body Mass Index (BMI)</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{formData.bmi}</span>
              </label>
              <input type="range" name="bmi" min="10" max="60" step="0.1" value={formData.bmi} onChange={handleChange} />
            </div>

            <div className="form-group" style={{ gridColumn: '1 / -1' }}>
              <label className="form-label">Smoking History</label>
              <select name="smoking_status" value={formData.smoking_status} onChange={handleChange} className="form-control">
                <option value="never smoked">Never Smoked</option>
                <option value="formerly smoked">Formerly Smoked</option>
                <option value="smokes">Currently Smokes</option>
                <option value="Unknown">Unknown</option>
              </select>
            </div>

            <div className="form-group" style={{ gridColumn: '1 / -1', marginTop: '0.5rem' }}>
              <label className="form-label" style={{ marginBottom: '1rem' }}>Clinical History</label>
              <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                <label className="custom-checkbox">
                  <input type="checkbox" name="hypertension" checked={formData.hypertension} onChange={handleChange} />
                  <span className="checkmark"></span>
                  <span style={{ color: 'var(--text-secondary)' }}>Hypertension</span>
                </label>
                
                <label className="custom-checkbox">
                  <input type="checkbox" name="heart_disease" checked={formData.heart_disease} onChange={handleChange} />
                  <span className="checkmark"></span>
                  <span style={{ color: 'var(--text-secondary)' }}>Heart Disease</span>
                </label>
                
                <label className="custom-checkbox">
                  <input type="checkbox" name="ever_married" checked={formData.ever_married} onChange={handleChange} />
                  <span className="checkmark"></span>
                  <span style={{ color: 'var(--text-secondary)' }}>Ever Married</span>
                </label>
              </div>
            </div>
          </div>

          <div style={{ marginTop: '2.5rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-color)' }}>
            <button type="submit" className="btn-primary" disabled={loading || models.length === 0}>
              {loading ? "Processing Analysis..." : "Run Diagnostic"}
            </button>
          </div>
        </form>

        {/* Result & Explanation Panel */}
        <div className="glass-panel result-card">
          <h2 style={{ fontSize: '1.125rem', marginBottom: '2rem' }}>Diagnostic Result</h2>

          {result ? (
            <div className="animate-fade-in" style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div className={`risk-display-modern ${getRiskClass(result.risk_category)}`}>
                <div className="risk-value">{(result.stroke_probability * 100).toFixed(1)}<span style={{ fontSize: '2rem', opacity: 0.8 }}>%</span></div>
                <div className="risk-label">Probability</div>
                <div className={`status-badge ${getRiskClass(result.risk_category)}`}>
                  {result.risk_category}
                </div>
              </div>

              {/* SHAP Explanation Section */}
              {result.shap_explanation && result.shap_explanation.length > 0 && (
                <div className="shap-bar-container">
                  <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Model Explanation (Impact)
                  </h3>
                  
                  {result.shap_explanation.map((item, idx) => {
                    const widthPct = (Math.abs(item.impact) / maxAbsImpact) * 100;
                    const isPositive = item.impact > 0;
                    return (
                      <div className="shap-item" key={idx}>
                        <div className="shap-label" title={formatFeatureName(item.feature)}>
                          {formatFeatureName(item.feature)}
                        </div>
                        <div className="shap-track">
                          <div 
                            className={`shap-fill ${isPositive ? 'pos' : 'neg'}`} 
                            style={{ width: `${Math.max(widthPct, 2)}%` }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><div style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--success)' }}></div> Decreases Risk</span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><div style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--danger)' }}></div> Increases Risk</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', opacity: 0.5, marginTop: '2rem' }}>
              <Info size={40} color="var(--text-muted)" style={{ marginBottom: '1rem' }} />
              <p style={{ color: 'var(--text-muted)', textAlign: 'center', maxWidth: '200px', fontSize: '0.9rem' }}>
                Complete the profile and run the diagnostic to see risk probability and feature impact.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
