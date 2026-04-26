import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { FraudDetectionPanel } from '../components/LoyaltyDashboard/FraudDetectionPanel'

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient()
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

test('renders fraud stats and recent alerts', async () => {
  renderWithClient(<FraudDetectionPanel />)
  await waitFor(() => expect(screen.getByText(/Fraud Detection/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Total Alerts/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/High Risk/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Recent Alerts/i)).toBeInTheDocument())
})

test('renders risk distribution labels', async () => {
  renderWithClient(<FraudDetectionPanel />)
  await waitFor(() => expect(screen.getByText(/Risk Distribution/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Risk Score/i)).toBeInTheDocument())
})
