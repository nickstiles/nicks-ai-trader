FROM golang:1.24.3
WORKDIR /app
COPY . .
RUN go mod tidy
CMD ["go", "run", "./cmd/trading-api"]