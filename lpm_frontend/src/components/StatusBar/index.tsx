import { type ModelStatus } from '@/store/useTrainingStore';

interface StatusBarProps {
  status: ModelStatus;
}

export function StatusBar({ status }: StatusBarProps) {
  const steps = [
    { title: 'Identity', status: 'seed_identity', icon: '🌱' },
    { title: 'Memory Upload', status: 'memory_upload', icon: '📝' },
    { title: 'Training', status: 'training', icon: '⚡' },
    { title: 'Trained', status: 'trained', icon: '✓' }
  ] as const;

  const getStepState = (stepStatus: (typeof steps)[number]['status']) => {
    const statusOrder = ['seed_identity', 'memory_upload', 'training', 'trained'];
    const currentIndex = statusOrder.indexOf(status);
    const stepIndex = statusOrder.indexOf(stepStatus);

    // If current status is trained, all previous steps should be completed
    if (status === 'trained') {
      return {
        isActive: stepStatus === 'trained',
        isCompleted: stepStatus !== 'trained'
      };
    }

    // If current status is training, previous steps should be completed
    if (
      status === 'training' &&
      (stepStatus === 'seed_identity' || stepStatus === 'memory_upload')
    ) {
      return {
        isActive: false,
        isCompleted: true
      };
    }

    return {
      isActive: stepStatus === status,
      isCompleted: currentIndex > stepIndex
    };
  };

  return (
    <div className="inline-flex items-center">
      {steps.map((step, index) => {
        const state = getStepState(step.status);
        const currentStepIndex = steps.findIndex((s) => s.status === status);

        return (
          <div key={step.status} className="flex items-center">
            {index > 0 && (
              <div
                className={`w-9 h-[1px] transition-colors ${index <= currentStepIndex ? 'bg-blue-600' : 'bg-gray-200'}`}
              />
            )}
            <div
              className={`flex items-center gap-1.5 px-2 py-1 text-sm font-medium transition-colors whitespace-nowrap
                ${state.isActive || state.isCompleted ? 'text-blue-600' : 'text-gray-300'}`}
            >
              <span className={state.isActive || state.isCompleted ? 'text-blue-600' : ''}>
                {step.icon}
              </span>
              <span>{step.title}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
