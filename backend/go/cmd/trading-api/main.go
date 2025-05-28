package main

import (
	"log"
	"net/http"

	"github.com/nickstiles/nicks-ai-trader/backend/go/internal/router"
)

func main() {
	r := router.NewRouter()
	log.Println("Starting trading-api on :8080")
	log.Fatal(http.ListenAndServe(":8080", r))
}
