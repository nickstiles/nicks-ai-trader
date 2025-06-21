const express = require("express")
const { createProxyMiddleware } = require("http-proxy-middleware")
const cors = require("cors")

const app = express()
const PORT = 3001

app.use(cors())
app.use(express.json())

// simple request logger
app.use("/api", (req, res, next) => {
  console.log(`🔁 [Node] ${req.method} ${req.originalUrl}`)
  next()
})

// trade_manager proxy
app.use(
  "/api/trade",
  createProxyMiddleware({
    target: "http://trade_manager:8002/trade",  // note the `/trade` suffix
    changeOrigin: true,
    pathRewrite: { "^/api/trade": "" },          // /api/trade/open → /open on target
    ws: true,
    logLevel: "debug",
  })
);

// signal_generator proxy
app.use(
  "/api/signal",
  createProxyMiddleware({
    target: "http://signal_generator:8001/signal", // note the `/signal`
    changeOrigin: true,
    pathRewrite: { "^/api/signal": "" },            // /api/signals/latest → /latest
    ws: true,
    logLevel: "debug",
  })
);

app.listen(PORT, () => {
  console.log(`Node proxy server running at http://localhost:${PORT}`)
})