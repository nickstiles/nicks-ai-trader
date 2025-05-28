# 🧠 AI-Powered Stock Trading Platform

**An end-to-end, full-stack platform for AI-driven stock trading.**
Built using **Go**, **Python (ML)**, **Node.js**, and **React**.
The system fetches market data, generates ML-based trading signals, executes trades via brokerage APIs, and visualizes performance in a clean dashboard.

---

## ✨ Features

* 📈 **Market Data Ingestion**: Real-time and historical stock data collection
* 🧠 **AI Trading Engine**: ML models for signal generation and strategy evaluation (Python)
* ⚙️ **Trade Execution**: Go-based service to manage orders through brokerage APIs (e.g. Alpaca, Interactive Brokers)
* 🛡️ **Authentication & API Gateway**: Node.js server to handle user auth, requests, and routing
* 📊 **Frontend Dashboard**: React-based interface for portfolio monitoring, strategy metrics, and visualizations
* 🐳 **Containerized**: Docker support for local and cloud deployments

---

## 🧱 Tech Stack

| Layer            | Technology                          |
| ---------------- | ----------------------------------- |
| Backend Logic    | Go                                  |
| Machine Learning | Python (Pandas, Scikit-learn, etc.) |
| API Gateway      | Node.js                             |
| Frontend         | React                               |
| DevOps           | Docker, GitHub Actions (planned)    |

---

## 📁 Project Structure

```
nicks-ai-trader/
├── backend/
│   ├── go/           # Trading engine and brokerage APIs
│   └── python/       # ML models, training, inference
├── frontend/
│   ├── client/       # React app for dashboard
│   └── server/       # Node.js API and auth
├── deployments/      # Docker, Kubernetes configs
├── scripts/          # Dev tools and automation
├── docs/             # Architecture, API docs
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

* Go 1.21+
* Python 3.10+
* Node.js 18+
* Docker (optional, for containerized development)

### Clone the Repo

```bash
git clone https://github.com/nickstiles/nicks-ai-trader.git
cd nicks-ai-trader
```

### Setup (WIP)

Each service has its own README for detailed setup:

* `backend/go`: `go run ./cmd/trading-api`
* `backend/python`: `python -m api.main`
* `frontend/client`: `npm install && npm start`
* `frontend/server`: `npm install && npm run dev`

---

## 📌 Status

This project is under active development.
Planned enhancements include:

* Strategy backtesting module
* Kubernetes deployment templates
* Real-time WebSocket feeds
* Authentication with OAuth

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🧑‍💻 Author

Built by [Nick Stiles](https://www.linkedin.com/in/nicholas-h-stiles) — *AI developer & full-stack engineer passionate about fintech and trading systems.*
