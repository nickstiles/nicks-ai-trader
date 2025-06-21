import React from 'react';

export function TradesTable({
  title,
  data,
  columns,
  badgeColor = 'bg-blue-100 text-blue-800',
  showCount = true,
}) {
  return (
    <div className="shadow-lg bg-white rounded-lg overflow-hidden">
      <div className="p-6 border-b flex justify-between items-center">
        <h2 className="text-2xl font-semibold">{title}</h2>
        {showCount && (
          <span className={`${badgeColor} rounded-full px-3 py-1 text-sm`}>
            {data.length}
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b text-gray-700">
              {columns.map(({ header, accessor }, colIndex) => (
                <th
                  key={`${accessor}-${colIndex}`}
                  className="py-2 px-4 align-middle"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, rowIndex) => (
              <tr
                key={row.db_id ?? rowIndex}
                className="border-b hover:bg-gray-50"
              >
                {columns.map(({ accessor, cell }, colIndex) => {
                  const value = row[accessor];
                  return (
                    <td
                      key={`${accessor}-${rowIndex}-${colIndex}`}
                      className="py-2 px-4 align-middle"
                    >
                      {cell ? cell(value, row) : value}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
