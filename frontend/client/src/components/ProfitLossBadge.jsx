import React from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';

/**
 * Renders a small badge showing profit or loss percentage
 * with an up/down arrow and color coding.
 *
 * Props:
 *  - value: number | null  // fraction (e.g. 0.05 for 5%)
 */
export function ProfitLossBadge({ value }) {
  if (value == null) {
    return <span className="text-gray-500">—</span>;
  }

  const isPositive = value >= 0;
  const formatted = `${(value * 100).toFixed(2)}%`;

  // fixed width & height, centered content
  const baseClasses = "inline-flex items-center justify-start text-sm font-medium w-24 h-8 rounded whitespace-nowrap pl-3 pr-2";

  const colorClasses = isPositive
    ? "bg-green-100 text-green-800"
    : "bg-red-100 text-red-800";

  const Icon = isPositive ? ArrowUp : ArrowDown;

  return (
    <span className={`${baseClasses} ${colorClasses}`}>
      <Icon className="w-4 h-4 mr-1 align-middle" />
      <span className="align-middle">{formatted}</span>
    </span>
  );
}