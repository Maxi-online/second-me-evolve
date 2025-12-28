'use client';

import { useEffect } from 'react';

interface IContextSettings {
  enableL0Retrieval: boolean;
  enableL1Retrieval: boolean;
  enableHelperModel: boolean;
  selectedModel: string;
  apiKey: string;
  systemPrompt: string;
  temperature: number;
}

interface ContextSettingsProps {
  settings: IContextSettings;
  onSettingsChange: (settings: IContextSettings) => void;
}

export default function ContextSettings({ settings, onSettingsChange }: ContextSettingsProps) {
  const handleToggle = (
    key: keyof Omit<IContextSettings, 'systemPrompt' | 'selectedModel' | 'apiKey'>
  ) => {
    onSettingsChange({
      ...settings,
      [key]: !settings[key]
    });
  };

  useEffect(() => {
    if (!localStorage.getItem('playgroundSettings')) {
      onSettingsChange(settings);
    }
  }, [settings]);

  useEffect(() => {
    if (!settings.selectedModel) {
      onSettingsChange({
        ...settings,
        selectedModel: 'ollama',
        apiKey: 'http://localhost:11434'
      });
    }
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 space-y-6 h-full">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold mb-0">Chat Settings</h2>
          <button
            className="p-1.5 rounded-full hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
            onClick={() => {
              const modal = document.createElement('div');

              modal.className =
                'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
              modal.innerHTML = `
                <div class="bg-white rounded-xl max-w-2xl p-6 m-4 space-y-4 relative shadow-xl">
                  <h3 class="text-xl font-semibold">Hybrid Chat Architecture</h3>
                  <div class="space-y-4 text-gray-600">
                    <p>Your SecondMe uses a hybrid architecture that combines personal memory with advanced AI capabilities:</p>
                    
                    <div class="space-y-2">
                      <h4 class="font-medium text-gray-900">1. Memory Retrieval (L0/L1)</h4>
                      <p>Access your personal memories to provide responses that reflect your experiences and knowledge:</p>
                      <ul class="list-disc pl-5 space-y-1">
                        <li>L0: Quick lookup of relevant memories</li>
                        <li>L1: Deep semantic search for complex context</li>
                      </ul>
                    </div>

                    <div class="space-y-2">
                      <h4 class="font-medium text-gray-900">2. Support Model</h4>
                      <p>For complex tasks, SecondMe can consult a more powerful AI model to:</p>
                      <ul class="list-disc pl-5 space-y-1">
                        <li>Break down complex problems</li>
                        <li>Provide step-by-step reasoning</li>
                        <li>Handle specialized knowledge domains</li>
                      </ul>
                    </div>

                    <p class="text-sm italic">By combining your memories with support model capabilities, Second Me can tackle both personal conversations and complex problem-solving tasks effectively.</p>
                  </div>
                  <button class="absolute top-4 right-4 p-2 text-gray-400 hover:text-gray-600" onclick="this.parentElement.parentElement.remove()">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              `;
              document.body.appendChild(modal);
              modal.onclick = (e) => {
                if (e.target === modal) modal.remove();
              };
            }}
            title="Learn about Hybrid Architecture"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </button>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-start justify-between">
          <div className="max-w-[235px]">
            <label className="font-medium">Memory Retrieval</label>
            <p className="text-sm text-gray-500">Configure how Second Me accesses your memories</p>
          </div>
          <div className="pt-1">
            <button
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.enableL0Retrieval ? 'bg-blue-600' : 'bg-gray-200'
              }`}
              onClick={() => handleToggle('enableL0Retrieval')}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings.enableL0Retrieval ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>

        {/* Support Model section hidden as requested */}

        <div className="space-y-2">
          <label className="font-medium">System Prompt</label>
          <p className="text-sm text-gray-500">Configure the base behavior of your SecondMe</p>
          <textarea
            className="w-full h-32 px-3 py-2 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            onChange={(e) =>
              onSettingsChange({
                ...settings,
                systemPrompt: e.target.value
              })
            }
            placeholder="Enter system prompt..."
            value={settings.systemPrompt}
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <label className="font-medium">Temperature</label>
              <p className="text-sm text-gray-500">Adjust creativity (0 = precise, 1 = creative)</p>
            </div>
            <div className="flex items-center gap-2">
              <input
                className="w-16 px-2 py-1 border rounded text-center"
                max="1"
                min="0"
                onChange={(e) => {
                  const value = parseFloat(e.target.value);

                  if (!isNaN(value) && value >= 0 && value <= 1) {
                    onSettingsChange({
                      ...settings,
                      temperature: value
                    });
                  }
                }}
                step="0.01"
                type="number"
                value={settings.temperature}
              />
              <button
                className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded"
                onClick={() =>
                  onSettingsChange({
                    ...settings,
                    temperature: 0.1
                  })
                }
              >
                Reset
              </button>
            </div>
          </div>
          <input
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            max="1"
            min="0"
            onChange={(e) =>
              onSettingsChange({
                ...settings,
                temperature: parseFloat(e.target.value)
              })
            }
            step="0.01"
            type="range"
            value={settings.temperature}
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>Precise</span>
            <span>Creative</span>
          </div>
        </div>
      </div>
    </div>
  );
}
