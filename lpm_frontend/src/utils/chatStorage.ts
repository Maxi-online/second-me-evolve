// Storage keys for different chat types
const STORAGE_KEYS = {
  PLAYGROUND: 'playgroundChat',
  CHAT_WITH_UPLOAD: 'chatWithUpload',
  ROLEPLAY: 'roleplayChat'
} as const;

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string;
}

export interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: string;
  messages: ChatMessage[];
}

// Type for playground chat which only stores messages
interface PlaygroundChatData {
  messages: ChatMessage[];
}

// Type for chat with upload which stores sessions and their messages
interface ChatWithUploadData {
  sessions: ChatSession[];
}

// Type for roleplay chat which stores chats by role_id
interface RoleplayChatData {
  [role_id: string]: {
    messages: ChatMessage[];
  };
}

// Helper function to safely parse JSON from localStorage
function getStorageData<T>(key: string, defaultValue: T): T {
  try {
    const data = localStorage.getItem(key);

    if (!data) return defaultValue;

    // Convert old format data to new format
    const parsedData = JSON.parse(data);

    if (key === STORAGE_KEYS.PLAYGROUND) {
      // If it's old format data, convert it
      if (parsedData.messages?.[0]?.isUser !== undefined) {
        parsedData.messages = parsedData.messages.map((msg: any) => ({
          id: msg.id,
          content: msg.content,
          role: msg.isUser ? 'user' : 'assistant',
          timestamp: msg.timestamp
        }));
      }
    } else if (key === STORAGE_KEYS.CHAT_WITH_UPLOAD) {
      // If it's old format data, convert it
      if (parsedData.sessions?.[0]?.messages?.[0]?.isUser !== undefined) {
        parsedData.sessions = parsedData.sessions.map((session: any) => ({
          ...session,
          messages: session.messages.map((msg: any) => ({
            id: msg.id,
            content: msg.content,
            role: msg.isUser ? 'user' : 'assistant',
            timestamp: msg.timestamp
          }))
        }));
      }
    } else if (key === STORAGE_KEYS.ROLEPLAY) {
      // If it's old format data, convert it
      Object.keys(parsedData).forEach((role_id: string) => {
        if (parsedData[role_id].messages?.[0]?.isUser !== undefined) {
          parsedData[role_id].messages = parsedData[role_id].messages.map((msg: any) => ({
            id: msg.id,
            content: msg.content,
            role: msg.isUser ? 'user' : 'assistant',
            timestamp: msg.timestamp
          }));
        }
      });
    }

    return parsedData;
  } catch (error) {
    console.error(`Error parsing data for key ${key}:`, error);

    return defaultValue;
  }
}

// Helper function to safely save JSON to localStorage
function setStorageData(key: string, data: any) {
  try {
    localStorage.setItem(key, JSON.stringify(data));
  } catch (error) {
    console.error(`Error saving data for key ${key}:`, error);
  }
}

// Playground Chat Storage Functions
export const playgroundChatStorage = {
  getMessages: (): ChatMessage[] => {
    const data = getStorageData<PlaygroundChatData>(STORAGE_KEYS.PLAYGROUND, { messages: [] });

    return data.messages;
  },

  addMessage: (message: ChatMessage) => {
    const messages = playgroundChatStorage.getMessages();

    setStorageData(STORAGE_KEYS.PLAYGROUND, {
      messages: [...messages, message]
    });
  },

  saveMessages: (messages: ChatMessage[]) => {
    setStorageData(STORAGE_KEYS.PLAYGROUND, { messages });
  },

  clearMessages: () => {
    setStorageData(STORAGE_KEYS.PLAYGROUND, { messages: [] });
  }
};

// Chat with Upload Storage Functions
export const chatWithUploadStorage = {
  getSessions: (): ChatSession[] => {
    const data = getStorageData<ChatWithUploadData>(STORAGE_KEYS.CHAT_WITH_UPLOAD, {
      sessions: []
    });

    return data.sessions;
  },

  createSession: (title: string = 'New Conversation'): ChatSession => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title,
      lastMessage: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      messages: []
    };

    const sessions = chatWithUploadStorage.getSessions();

    setStorageData(STORAGE_KEYS.CHAT_WITH_UPLOAD, {
      sessions: [newSession, ...sessions]
    });

    return newSession;
  },

  updateSession: (sessionId: string, updates: Partial<ChatSession>) => {
    const sessions = chatWithUploadStorage.getSessions();
    const updatedSessions = sessions.map((session) =>
      session.id === sessionId ? { ...session, ...updates } : session
    );

    setStorageData(STORAGE_KEYS.CHAT_WITH_UPLOAD, { sessions: updatedSessions });
  },

  addMessage: (sessionId: string, message: ChatMessage) => {
    const sessions = chatWithUploadStorage.getSessions();
    const updatedSessions = sessions.map((session) =>
      session.id === sessionId
        ? {
            ...session,
            messages: [...session.messages, message],
            lastMessage: message.content,
            timestamp: message.timestamp
          }
        : session
    );

    setStorageData(STORAGE_KEYS.CHAT_WITH_UPLOAD, { sessions: updatedSessions });
  },

  saveSessionMessages: (sessionId: string, messages: ChatMessage[]) => {
    const sessions = chatWithUploadStorage.getSessions();
    const lastMessage = messages.length > 0 ? messages[messages.length - 1] : undefined;
    const updatedSessions = sessions.map((session) =>
      session.id === sessionId
        ? {
            ...session,
            messages,
            title: messages.length > 0 ? session.title : 'New Conversation',
            lastMessage: lastMessage?.content || '',
            timestamp: lastMessage?.timestamp || session.timestamp
          }
        : session
    );

    setStorageData(STORAGE_KEYS.CHAT_WITH_UPLOAD, { sessions: updatedSessions });
  },

  getSessionMessages: (sessionId: string): ChatMessage[] => {
    const sessions = chatWithUploadStorage.getSessions();
    const session = sessions.find((s) => s.id === sessionId);

    return session?.messages || [];
  },

  deleteSession: (sessionId: string) => {
    const sessions = chatWithUploadStorage.getSessions();

    setStorageData(STORAGE_KEYS.CHAT_WITH_UPLOAD, {
      sessions: sessions.filter((s) => s.id !== sessionId)
    });
  }
};

// Roleplay Chat Storage Functions
export const roleplayChatStorage = {
  getMessages: (role_id: string): ChatMessage[] => {
    const data = getStorageData<RoleplayChatData>(STORAGE_KEYS.ROLEPLAY, {});

    return data[role_id]?.messages || [];
  },

  saveMessages: (role_id: string, messages: ChatMessage[]) => {
    const data = getStorageData<RoleplayChatData>(STORAGE_KEYS.ROLEPLAY, {});
    const updatedData = {
      ...data,
      [role_id]: {
        messages
      }
    };

    setStorageData(STORAGE_KEYS.ROLEPLAY, updatedData);
  },

  addMessage: (role_id: string, message: ChatMessage) => {
    const data = getStorageData<RoleplayChatData>(STORAGE_KEYS.ROLEPLAY, {});
    const roleData = data[role_id] || { messages: [] };

    const updatedData = {
      ...data,
      [role_id]: {
        messages: [...roleData.messages, message]
      }
    };

    setStorageData(STORAGE_KEYS.ROLEPLAY, updatedData);
  },

  clearMessages: (role_id: string) => {
    const data = getStorageData<RoleplayChatData>(STORAGE_KEYS.ROLEPLAY, {});

    const updatedData = {
      ...data,
      [role_id]: {
        messages: []
      }
    };

    setStorageData(STORAGE_KEYS.ROLEPLAY, updatedData);
  },

  deleteRole: (role_id: string) => {
    const data = getStorageData<RoleplayChatData>(STORAGE_KEYS.ROLEPLAY, {});
    const { [role_id]: _, ...rest } = data;

    setStorageData(STORAGE_KEYS.ROLEPLAY, rest);
  }
};
