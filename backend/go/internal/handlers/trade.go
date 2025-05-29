package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"time"
)

type OrderRequest struct {
	Ticker   string `json:"ticker"`
	Action   string `json:"action"`   // "buy" or "sell"
	Quantity int    `json:"quantity"` // number of shares
}

type OrderResponse struct {
	OrderID string `json:"order_id"`
	Status  string `json:"status"`
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

func SubmitTradeHandler(w http.ResponseWriter, r *http.Request) {
	var req OrderRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("❌ [Go] Invalid order request: %v", err)
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}
	log.Printf("🔁 [Go] Received trade order: %+v", req)

	// Simulate order execution
	orderID := fmt.Sprintf("ORD-%d", rand.Intn(1e6))
	resp := OrderResponse{
		OrderID: orderID,
		Status:  "executed",
	}
	log.Printf("✅ [Go] Order executed: %v", resp)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
