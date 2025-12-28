export enum ROUTER_PATH {
  HOME = '/',

  DASHBOARD = '/dashboard',

  TRAIN = '/dashboard/train',
  TRAIN_IDENTITY = '/dashboard/train/identity',
  TRAIN_MEMORIES = '/dashboard/train/memories',
  TRAIN_TRAINING = '/dashboard/train/training',

  PLAYGROUND = '/dashboard/playground',
  PLAYGROUND_CHAT = '/dashboard/playground/chat',
  PLAYGROUND_BRIDGE = '/dashboard/playground/bridge',

  APPLICATIONS = '/dashboard/applications',
  APPLICATIONS_ROLEPLAY = '/dashboard/applications/roleplay-apps',
  APPLICATIONS_TASK = '/dashboard/applications/task-apps',
  APPLICATIONS_SECOND_X = '/dashboard/applications/second-x',
  APPLICATIONS_NETWORK = '/dashboard/applications/network-apps',
  APPLICATIONS_INTEGRATIONS = '/dashboard/applications/integrations',
  APPLICATIONS_API_MCP = '/dashboard/applications/api-mcp',

  STANDALONE = '/standalone',
  STANDALONE_ROLE = '/standalone/role/:role_id',
  // STANDALONE_ROOM = '/standalone/room/:roomId',
  // STANDALONE_TASK = '/standalone/task/:taskId',
  STANDALONE_SPACE = '/standalone/space/:space_id'
}

export const DEFAULT_ROUTER = ROUTER_PATH.HOME;
