interface Participant {
  id: string;
  name?: string;
}

interface Room {
  id: string;
  name: string;
  objective: string;
  participants: Participant[];
  status: 'active' | 'completed' | 'failed';
  createdAt: string;
  lastMessage?: string;
}

interface RoomCardProps {
  room: Room;
  onClick: () => void;
}

const getAvatarUrl = (id: string) => {
  return `https://api.dicebear.com/7.x/bottts/svg?seed=${id}`;
};

export default function RoomCard({ room, onClick }: RoomCardProps) {
  return (
    <div
      className="bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-200 cursor-pointer border border-gray-200 hover:border-blue-300 group overflow-hidden"
      onClick={onClick}
    >
      {/* Card Header */}
      <div className="p-6 pb-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xl font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
            {room.name}
          </h3>
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              room.status === 'active'
                ? 'bg-green-100 text-green-800'
                : room.status === 'completed'
                  ? 'bg-blue-100 text-blue-800'
                  : 'bg-red-100 text-red-800'
            }`}
          >
            {room.status.charAt(0).toUpperCase() + room.status.slice(1)}
          </span>
        </div>
        <p className="text-gray-600 line-clamp-2 mb-4">{room.objective}</p>

        {/* Participants */}
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {room.participants.map((participant, index) => (
              <div
                key={participant.id}
                className="flex items-center gap-2 px-2 py-1 rounded-md bg-gray-50"
                title={participant.id}
              >
                <img
                  alt={participant.name || `SecondMe ${index + 1}`}
                  className="w-6 h-6 rounded-full border border-white"
                  src={getAvatarUrl(participant.id)}
                />
                <span className="text-xs text-gray-600">
                  {participant.name || `SecondMe ${index + 1}`}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Card Footer */}
      <div className="border-t border-gray-100 px-6 py-4 bg-gray-50 flex items-center justify-between">
        <div className="flex items-center space-x-2 text-sm text-gray-500">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
            />
          </svg>
          <span>{room.createdAt}</span>
        </div>
        <span className="text-sm text-blue-600 group-hover:text-blue-700 font-medium flex items-center space-x-1">
          <span>View Room</span>
          <svg
            className="w-4 h-4 transform group-hover:translate-x-1 transition-transform"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
          </svg>
        </span>
      </div>
    </div>
  );
}
