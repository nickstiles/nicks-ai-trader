export async function loadTickerOptions(inputValue) {
  const res = await fetch(`http://localhost:3001/api/tickers?q=${inputValue}`);
  return res.json(); // [{ value, label }, …]
}