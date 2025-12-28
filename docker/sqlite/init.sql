-- Document Table
CREATE TABLE IF NOT EXISTS document (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL DEFAULT '',
    title VARCHAR(511) NOT NULL DEFAULT '',
    extract_status TEXT CHECK(extract_status IN ('INITIALIZED', 'SUCCESS', 'FAILED')) NOT NULL DEFAULT 'INITIALIZED',
    embedding_status TEXT CHECK(embedding_status IN ('INITIALIZED', 'SUCCESS', 'FAILED')) NOT NULL DEFAULT 'INITIALIZED',
    analyze_status TEXT CHECK(analyze_status IN ('INITIALIZED', 'SUCCESS', 'FAILED')) NOT NULL DEFAULT 'INITIALIZED',
    mime_type VARCHAR(50) NOT NULL DEFAULT '',
    raw_content TEXT DEFAULT NULL,
    user_description VARCHAR(255) NOT NULL DEFAULT '',
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    url VARCHAR(1023) NOT NULL DEFAULT '',
    document_size INTEGER NOT NULL DEFAULT 0,
    insight TEXT DEFAULT NULL,  -- JSON data stored as TEXT
    summary TEXT DEFAULT NULL,  -- JSON data stored as TEXT
    keywords TEXT DEFAULT NULL
);

-- Document table indexes
CREATE INDEX IF NOT EXISTS idx_extract_status ON document(extract_status);
CREATE INDEX IF NOT EXISTS idx_name ON document(name);
CREATE INDEX IF NOT EXISTS idx_create_time ON document(create_time);

-- Chunk Table
CREATE TABLE IF NOT EXISTS chunk (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    has_embedding BOOLEAN NOT NULL DEFAULT 0,
    tags TEXT DEFAULT NULL,  -- JSON data stored as TEXT
    topic VARCHAR(255) DEFAULT NULL,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES document(id)
);

-- Chunk table indexes
CREATE INDEX IF NOT EXISTS idx_document_id ON chunk(document_id);
CREATE INDEX IF NOT EXISTS idx_has_embedding ON chunk(has_embedding);

-- L1 Version Table
CREATE TABLE IF NOT EXISTS l1_versions (
    version INTEGER PRIMARY KEY,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    description VARCHAR(500)
);

-- L1 Bio Table
CREATE TABLE IF NOT EXISTS l1_bios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    content TEXT,
    content_third_view TEXT,
    summary TEXT,
    summary_third_view TEXT,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version) REFERENCES l1_versions(version)
);

-- L1 Shade Table
CREATE TABLE IF NOT EXISTS l1_shades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    name VARCHAR(200),
    aspect VARCHAR(200),
    icon VARCHAR(100),
    desc_third_view TEXT,
    content_third_view TEXT,
    desc_second_view TEXT,
    content_second_view TEXT,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version) REFERENCES l1_versions(version)
);

-- L1 Cluster Table
CREATE TABLE IF NOT EXISTS l1_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    cluster_id VARCHAR(100),
    memory_ids TEXT,  -- JSON data stored as TEXT
    cluster_center TEXT,  -- JSON data stored as TEXT
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version) REFERENCES l1_versions(version)
);

-- L1 Chunk Topic Table
CREATE TABLE IF NOT EXISTS l1_chunk_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    chunk_id VARCHAR(100),
    topic TEXT,
    tags TEXT,  -- JSON data stored as TEXT
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version) REFERENCES l1_versions(version)
);

-- Status Biography Table
CREATE TABLE IF NOT EXISTS status_biography (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_third_view TEXT NOT NULL,
    summary TEXT NOT NULL,
    summary_third_view TEXT NOT NULL,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Personal Load Table
CREATE TABLE IF NOT EXISTS loads (
    id VARCHAR(36) PRIMARY KEY,    -- UUID
    name VARCHAR(255) NOT NULL,    -- load name
    description TEXT,             -- load description
    email VARCHAR(255) NOT NULL DEFAULT '',  -- load email
    avatar_data TEXT,             -- load avatar base64 encoded data
    instance_id VARCHAR(255),     -- upload instance ID
    instance_password VARCHAR(255),  -- upload instance password
    status TEXT CHECK(status IN ('active', 'inactive', 'deleted')) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_loads_status ON loads(status);
CREATE INDEX IF NOT EXISTS idx_loads_created_at ON loads(created_at);

-- Memory Files Table
CREATE TABLE IF NOT EXISTS memories (
    id VARCHAR(36) NOT NULL,
    name VARCHAR(255) NOT NULL,
    size INTEGER NOT NULL,
    type VARCHAR(50) NOT NULL,
    path VARCHAR(1024) NOT NULL,
    meta_data TEXT,  -- JSON data stored as TEXT
    document_id VARCHAR(36),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT CHECK(status IN ('active', 'deleted')) NOT NULL DEFAULT 'active',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_memories_document_id ON memories(document_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);

-- Roles Table
CREATE TABLE IF NOT EXISTS roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid VARCHAR(64) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL UNIQUE,
    description VARCHAR(500),
    system_prompt TEXT NOT NULL,
    icon VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT 1,
    enable_l0_retrieval BOOLEAN NOT NULL DEFAULT 1,
    enable_l1_retrieval BOOLEAN NOT NULL DEFAULT 1,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name);
CREATE INDEX IF NOT EXISTS idx_roles_uuid ON roles(uuid);
CREATE INDEX IF NOT EXISTS idx_roles_is_active ON roles(is_active);

-- Insert predefined Roles (only if they don't exist)
INSERT OR IGNORE INTO roles (uuid, name, description, system_prompt, icon) VALUES 
('role_interviewer_8f3a1c2e4b5d6f7a9e0b1d2c3f4e5d6b', 
 'Interviewer (a test case)', 
 'Professional interviewer who asks insightful questions to learn about people', 
 
 'You are a professional interviewer with expertise in asking insightful questions to understand people deeply, and you are facing the interviewee, and you dont know his/her background. Your responsibilities include:\n1. Asking thoughtful, open-ended questions\n2. Following up on interesting points\n3. sharing what you know to attract the interviewee.',
 'interview-icon');

-- User LLM Configuration Table
CREATE TABLE IF NOT EXISTS user_llm_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_type VARCHAR(50) NOT NULL DEFAULT 'openai',
    key VARCHAR(200),
    
    -- Chat configuration
    chat_endpoint VARCHAR(200),
    chat_api_key VARCHAR(200),
    chat_model_name VARCHAR(200),
    
    -- Embedding configuration
    embedding_endpoint VARCHAR(200),
    embedding_api_key VARCHAR(200),
    embedding_model_name VARCHAR(200),
    
    -- Thinking configuration
    thinking_model_name VARCHAR(200),
    thinking_endpoint VARCHAR(200),
    thinking_api_key VARCHAR(200),

    -- Judge configuration (auto reinforcement / mood judge)
    judge_model_name VARCHAR(200),
    judge_endpoint VARCHAR(200),
    judge_api_key VARCHAR(200),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- User LLM Configuration table indexes
CREATE INDEX IF NOT EXISTS idx_user_llm_configs_created_at ON user_llm_configs(created_at);

-- Spaces Table
CREATE TABLE IF NOT EXISTS spaces (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    objective TEXT NOT NULL,
    participants TEXT NOT NULL,  -- JSON array stored as TEXT
    host VARCHAR(255) NOT NULL,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status INTEGER DEFAULT 1,
    conclusion TEXT,
    space_share_id VARCHAR(255)
);

-- Space Messages Table
CREATE TABLE IF NOT EXISTS space_messages (
    id VARCHAR(255) PRIMARY KEY,
    space_id VARCHAR(255) NOT NULL,
    sender_endpoint VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    message_type VARCHAR(50) NOT NULL,
    round INTEGER DEFAULT 0,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    role VARCHAR(50) DEFAULT 'participant',
    FOREIGN KEY (space_id) REFERENCES spaces(id)
);

-- Space Messages Table Indexes
CREATE INDEX IF NOT EXISTS idx_space_messages_space_id ON space_messages(space_id);
CREATE INDEX IF NOT EXISTS idx_space_messages_round ON space_messages(round);
CREATE INDEX IF NOT EXISTS idx_space_messages_create_time ON space_messages(create_time);

-- Space Table Indexes
CREATE INDEX IF NOT EXISTS idx_spaces_create_time ON spaces(create_time);
CREATE INDEX IF NOT EXISTS idx_spaces_status ON spaces(status);

-- =========================
-- Chat learning (RL/DPO) tables
-- =========================

CREATE TABLE IF NOT EXISTS chat_experiences (
    id VARCHAR(64) PRIMARY KEY,
    source VARCHAR(50) NOT NULL DEFAULT 'kernel2',
    model VARCHAR(200),
    prompt_messages TEXT NOT NULL, -- JSON stored as TEXT
    temperature INTEGER,           -- temperature * 1000
    max_tokens INTEGER,
    seed INTEGER,
    completion TEXT,
    finish_reason VARCHAR(50),
    meta_data TEXT NOT NULL DEFAULT '{}', -- JSON stored as TEXT
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_experiences_created_at ON chat_experiences(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_experiences_source ON chat_experiences(source);

CREATE TABLE IF NOT EXISTS chat_feedback (
    id VARCHAR(64) PRIMARY KEY,
    experience_id VARCHAR(64) NOT NULL,
    rating INTEGER NOT NULL DEFAULT 0,
    comment TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experience_id) REFERENCES chat_experiences(id)
);

CREATE INDEX IF NOT EXISTS idx_chat_feedback_experience_id ON chat_feedback(experience_id);
CREATE INDEX IF NOT EXISTS idx_chat_feedback_created_at ON chat_feedback(created_at);

CREATE TABLE IF NOT EXISTS chat_preferences (
    id VARCHAR(64) PRIMARY KEY,
    source VARCHAR(50) NOT NULL DEFAULT 'kernel2',
    model VARCHAR(200),
    prompt_messages TEXT NOT NULL, -- JSON stored as TEXT
    chosen TEXT NOT NULL,
    rejected TEXT NOT NULL,
    meta_data TEXT NOT NULL DEFAULT '{}', -- JSON stored as TEXT
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_preferences_created_at ON chat_preferences(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_preferences_source ON chat_preferences(source);