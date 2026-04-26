import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const performanceData = [
  { date: '2026-04-01', accuracy: 0.88, drift: 0.08 },
  { date: '2026-04-08', accuracy: 0.91, drift: 0.10 },
  { date: '2026-04-15', accuracy: 0.90, drift: 0.12 },
  { date: '2026-04-22', accuracy: 0.92, drift: 0.09 },
  { date: '2026-04-29', accuracy: 0.93, drift: 0.07 },
]

const metrics = [
  { label: 'Prediction Accuracy', value: '93.0%', description: 'Latest end-to-end model accuracy' },
  { label: 'F1 Score', value: '0.86', description: 'Balanced precision / recall' },
  { label: 'Data Drift', value: '0.12', description: 'Drift score over the latest week' },
  { label: 'AUC', value: '0.91', description: 'Link-prediction separability' },
]

export function ModelMonitoringDashboard() {
  return (
    <section style={{ display: 'grid', gap: 24 }}>
      <div style={{ display: 'grid', gap: 16, gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
        {metrics.map((metric) => (
          <div
            key={metric.label}
            style={{
              padding: 20,
              borderRadius: 16,
              background: '#fff',
              boxShadow: '0 2px 14px rgba(0, 0, 0, 0.06)',
              border: '1px solid #ececec',
            }}
          >
            <p style={{ margin: 0, fontSize: 14, color: '#666' }}>{metric.label}</p>
            <p style={{ margin: '12px 0', fontSize: 28, fontWeight: 700 }}>{metric.value}</p>
            <p style={{ margin: 0, fontSize: 12, color: '#888' }}>{metric.description}</p>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gap: 24, gridTemplateColumns: '1.5fr 1fr' }}>
        <div style={{ minHeight: 320, padding: 20, borderRadius: 16, background: '#fff', boxShadow: '0 2px 14px rgba(0, 0, 0, 0.06)', border: '1px solid #ececec' }}>
          <h2 style={{ marginTop: 0 }}>Prediction Accuracy Trend</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={performanceData} margin={{ top: 8, right: 24, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis domain={[0.7, 1.0]} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
              <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
              <Line type="monotone" dataKey="accuracy" stroke="#3f8efc" strokeWidth={3} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ minHeight: 320, padding: 20, borderRadius: 16, background: '#fff', boxShadow: '0 2px 14px rgba(0, 0, 0, 0.06)', border: '1px solid #ececec' }}>
          <h2 style={{ marginTop: 0 }}>Drift Detection</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={performanceData} margin={{ top: 8, right: 24, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis tickFormatter={(value) => value.toFixed(2)} />
              <Tooltip formatter={(value: number) => value.toFixed(2)} />
              <Legend />
              <Bar dataKey="drift" fill="#f65d5d" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p style={{ marginTop: 12, fontSize: 14, color: '#555' }}>
            Overall model drift is moderate. Watch for sudden deviations in feature distributions.
          </p>
        </div>
      </div>
    </section>
  )
}
