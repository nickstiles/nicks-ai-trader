import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Card, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import logo from '../assets/logo.png';
import { TradesTable } from '../components/TradesTable';
import { ProfitLossBadge } from '../components/ProfitLossBadge';

export default function Dashboard() {
  const [openTrades, setOpenTrades] = useState([]);
  const [closedTrades, setClosedTrades] = useState([]);
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchDashboardData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [openRes, closedRes, signalRes] = await Promise.all([
        axios.get('/api/trade/open'),
        axios.get('/api/trade/closed'),
        axios.get('/api/signal/latest'),
      ]);
      setOpenTrades(openRes.data);
      setClosedTrades(closedRes.data);
      setSignals(signalRes.data);
    } catch {
      setError('Failed to load data. Please try again.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  useEffect(() => {
    const es = new EventSource('/api/trade/stream');
    es.onmessage = () => fetchDashboardData();
    es.onerror = () => es.close();
    return () => es.close();
  }, [fetchDashboardData]);

  const openColumns = [
    { header: 'Symbol', accessor: 'option_symbol', sticky: 'left' },
    { header: 'Entry', accessor: 'entry_price', cell: v => `$${v.toFixed(2)}` },
    { header: 'Pred %', accessor: 'pred_return', cell: v => `${(v*100).toFixed(1)}%` },
    { header: 'Current', accessor: 'current_price', cell: v => `$${v.toFixed(2)}`, 
      style: { textAlign: 'right' } },
    { 
      header: 'P&L %', 
      accessor: 'pnl_pct', 
      cell: val => <ProfitLossBadge value={val} /> 
    },
    { header: 'Cash Used', accessor: 'cash_balance_on_entry', cell: v => `–$${v.toLocaleString()}` },
  ];

  const closedColumns = [
    { header: 'Symbol', accessor: 'option_symbol' },
    { header: 'Exit', accessor: 'exit_price', cell: val => `$${(val ?? 0).toFixed(2)}` },
    { header: 'PnL $', accessor: 'pnl_dollars', cell: val => `$${(val ?? 0).toFixed(2)}`, },
    { header: 'PnL %', accessor: 'pnl_pct', cell: val => `${((val ?? 0) * 100).toFixed(2)}%` }
  ];

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      {/* Header */}
      <header className="flex items-center mb-8">
        <img src={logo} alt="AI Trader Logo" className="h-12 w-12 mr-4" />
        <h1 className="text-3xl font-extrabold text-gray-800">AI Trader Dashboard</h1>
      </header>

      {/* Controls */}
      <div className="flex items-center justify-between mb-6">
        <Button onClick={fetchDashboardData} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh Dashboard"}
        </Button>
        {error && <p className="text-red-600 font-medium">{error}</p>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Signals */}
        <Card className="shadow-lg">
          <CardContent className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Recent Signals</h2>
            {signals.length === 0 ? (
              <p className="text-gray-500 italic">No recent signals found.</p>
            ) : (
              <div className="space-y-3">
                {signals.map((s) => (
                  <Card key={s.option_symbol} className="border">
                    <CardContent className="p-4 flex justify-between items-center">
                      <div>
                        <p className="font-bold text-lg">{s.ticker}</p>
                        <p className="text-sm text-gray-600">{s.option_symbol}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xl font-mono">{(s.pred_return * 100).toFixed(1)}%</p>
                        <p className="text-sm text-gray-500">${s.option_price.toFixed(2)}</p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Trades */}
        <div className="space-y-6">
          <TradesTable
            title="Open Trades"
            data={openTrades}
            columns={openColumns}
            badgeColor="bg-blue-100 text-blue-800"
          />
          <TradesTable
            title="Closed Trades"
            data={closedTrades}
            columns={closedColumns}
            badgeColor="bg-green-100 text-green-800"
          />
        </div>
      </div>
    </div>
  );
}
