const StatusCard = ({ status }) => {
  if (!status) return <p className="text-center text-gray-500">Loading status...</p>;

  return (
    <div className="mt-4 p-4 bg-white border border-gray-300 rounded-lg shadow text-center">
      <h2 className="text-xl font-semibold text-gray-800">Go Service Status</h2>
      <p className="text-green-600 font-medium mt-2">{status.status}</p>
    </div>
  );
};

export default StatusCard;