'use client';

import { useState } from 'react';

interface Result {
  id: string;
  input: string;
  output: string;
  type: 'enhancement' | 'critic';
  timestamp: string;
}

export default function BridgeMode() {
  const [results, setResults] = useState<Result[]>([]);
  const [enhancementInput, setEnhancementInput] = useState('');
  const [criticInput, setCriticInput] = useState('');
  const [loading, setLoading] = useState({
    enhancement: false,
    critic: false
  });

  const handleEnhancement = async () => {
    if (!enhancementInput.trim()) return;

    setLoading((prev) => ({ ...prev, enhancement: true }));

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));

    const newResult: Result = {
      id: Date.now().toString(),
      input: enhancementInput,
      output: `Enhanced version: ${enhancementInput}\n\nAdditional context-aware details have been integrated into the response.`,
      type: 'enhancement',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setResults((prev) => [newResult, ...prev]);
    setEnhancementInput('');
    setLoading((prev) => ({ ...prev, enhancement: false }));
  };

  const handleCritic = async () => {
    if (!criticInput.trim()) return;

    setLoading((prev) => ({ ...prev, critic: true }));

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));

    const newResult: Result = {
      id: Date.now().toString(),
      input: criticInput,
      output:
        'Analysis:\n\n1. Accuracy: The response appears to be [assessment]\n2. Relevance: [evaluation of relevance]\n3. Completeness: [evaluation of completeness]\n4. Suggestions: [recommendations for improvement]',
      type: 'critic',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setResults((prev) => [newResult, ...prev]);
    setCriticInput('');
    setLoading((prev) => ({ ...prev, critic: false }));
  };

  return (
    <div className="h-full w-full flex flex-col">
      <div className="flex-1 p-6 overflow-auto">
        <div className="max-w-6xl mx-auto">
          <div className="mb-6">
            <h1 className="text-2xl font-semibold text-gray-900">Bridge Mode</h1>
            <p className="text-sm text-gray-600 mt-1">
              Bridge Mode acts as a context-aware bridge between you and other AI systems or
              services. Your Second Me personalizes both outgoing and incoming information, creating
              a seamless experience tailored uniquely to you. This bidirectional context enhancement
              is a core function of Second Me.
            </p>
          </div>

          {/* Content area with frosted glass overlay */}
          <div className="relative min-h-[600px] rounded-lg">
            {/* Frosted glass overlay */}
            <div className="absolute inset-0 backdrop-blur-md bg-white/70 flex flex-col items-center justify-center z-20 rounded-lg">
              <div className="text-3xl font-bold text-gray-800">Coming Soon</div>
              <p className="text-sm text-gray-600 mt-2 text-center px-4">
                {`We're working on this feature. Stay tuned!`}
              </p>
            </div>

            {/* Content that will be blurred */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4">
              {/* Enhancement Section */}
              <div className="space-y-4">
                <div className="bg-white rounded-lg shadow-sm p-4">
                  <h3 className="text-lg font-semibold mb-4">Context Enhancement</h3>
                  <div className="space-y-4">
                    <textarea
                      className="w-full h-32 px-3 py-2 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      onChange={(e) => setEnhancementInput(e.target.value)}
                      placeholder="Enter text to enhance with your personal context..."
                      value={enhancementInput}
                    />
                    <button
                      className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                      disabled={loading.enhancement}
                      onClick={handleEnhancement}
                    >
                      {loading.enhancement ? (
                        <span className="animate-pulse">Enhancing...</span>
                      ) : (
                        'Enhance with Context'
                      )}
                    </button>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm p-4">
                  <h3 className="text-lg font-semibold mb-4">Enhancement Results</h3>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {results
                      .filter((result) => result.type === 'enhancement')
                      .map((result) => (
                        <div key={result.id} className="border rounded-lg p-4 space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">{result.timestamp}</span>
                          </div>
                          <div className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                            Input: {result.input}
                          </div>
                          <div className="text-sm whitespace-pre-wrap">{result.output}</div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>

              {/* Critic Section */}
              <div className="space-y-4">
                <div className="bg-white rounded-lg shadow-sm p-4">
                  <h3 className="text-lg font-semibold mb-4">Context Critic</h3>
                  <div className="space-y-4">
                    <textarea
                      className="w-full h-32 px-3 py-2 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      onChange={(e) => setCriticInput(e.target.value)}
                      placeholder="Enter AI response to analyze with your personal context..."
                      value={criticInput}
                    />
                    <button
                      className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                      disabled={loading.critic}
                      onClick={handleCritic}
                    >
                      {loading.critic ? (
                        <span className="animate-pulse">Analyzing...</span>
                      ) : (
                        'Analyze with Context'
                      )}
                    </button>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm p-4">
                  <h3 className="text-lg font-semibold mb-4">Analysis Results</h3>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {results
                      .filter((result) => result.type === 'critic')
                      .map((result) => (
                        <div key={result.id} className="border rounded-lg p-4 space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">{result.timestamp}</span>
                          </div>
                          <div className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                            Input: {result.input}
                          </div>
                          <div className="text-sm whitespace-pre-wrap">{result.output}</div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
