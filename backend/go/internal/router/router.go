package router

import (
	"github.com/gorilla/mux"
	"github.com/nickstiles/nicks-ai-trader/backend/go/internal/handlers"
)

func NewRouter() *mux.Router {
	r := mux.NewRouter()
	r.HandleFunc("/ping", handlers.PingHandler).Methods("GET")
	r.HandleFunc("/status", handlers.StatusHandler).Methods("GET")
	r.HandleFunc("/submit-trade", handlers.SubmitTradeHandler).Methods("POST")
	return r
}
