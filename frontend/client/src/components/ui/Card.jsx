import React from "react";

export function Card({ children }) {
  return <div className="bg-white shadow rounded-xl">{children}</div>;
}

export function CardContent({ children, className = "" }) {
  return <div className={className}>{children}</div>;
}
