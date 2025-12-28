'use client';

interface TaskCardProps {
  task: {
    id: string;
    description: string;
    participants: { name: string }[];
    status: 'running' | 'completed' | 'failed';
    createdAt: string;
  };
  onClick: () => void;
}

export default function TaskCard({ task, onClick }: TaskCardProps) {
  const statusColors = {
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800'
  };

  return (
    <button
      className="w-full text-left border rounded-lg p-4 hover:shadow-md transition-shadow bg-white"
      onClick={onClick}
    >
      <div className="flex justify-between items-start mb-3">
        <h3 className="text-lg font-medium line-clamp-2">{task.description}</h3>
        <span className={`px-2 py-1 rounded-full text-sm font-medium ${statusColors[task.status]}`}>
          {task.status.charAt(0).toUpperCase() + task.status.slice(1)}
        </span>
      </div>

      <div className="text-sm text-gray-500 mb-3">Created {task.createdAt}</div>

      <div className="flex flex-wrap gap-2">
        {task.participants.map((participant, index) => (
          <span key={index} className="px-2 py-1 bg-gray-100 rounded-full text-sm">
            {participant.name}
          </span>
        ))}
      </div>
    </button>
  );
}
