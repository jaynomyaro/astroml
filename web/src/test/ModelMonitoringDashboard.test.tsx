import { render, screen } from '@testing-library/react'
import { ModelMonitoringDashboard } from '../components/ModelMonitoringDashboard'

test('renders model monitoring dashboard', () => {
  render(<ModelMonitoringDashboard />)

  expect(screen.getByText(/Prediction Accuracy Trend/i)).toBeInTheDocument()
  expect(screen.getByText(/Drift Detection/i)).toBeInTheDocument()
  expect(screen.getByText(/Data Drift/i)).toBeInTheDocument()
})
