const StatusCard = ({ status }) => {
    if (!status) return <p>Loading status...</p>;
    return <p>Status: {status.status}</p>;
  };
  
  export default StatusCard;