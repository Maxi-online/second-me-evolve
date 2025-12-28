import { create } from 'zustand';
import type { SpaceInfo } from '@/service/space';
import {
  getAllSpaces,
  createSpace,
  getSpaceDetail,
  startSpace,
  deleteSpace as deleteSpaceApi
} from '@/service/space';

interface ISpaceStore {
  spaces: SpaceInfo[];
  currentSpace: SpaceInfo | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchAllSpaces: () => Promise<void>;
  fetchSpaceById: (space_id: string) => Promise<SpaceInfo | null>;
  addSpace: (spaceData: {
    title: string;
    objective: string;
    host: string;
    participants: string[];
    participants_info?: {
      url: string;
      role_description?: string;
    }[];
  }) => Promise<SpaceInfo | null>;
  startSpace: (space_id: string) => Promise<SpaceInfo | null>;
  deleteSpace: (space_id: string) => Promise<boolean>;
  updateSpaceStatus: (space_id: string, status: number) => void;
  setCurrentSpace: (space: SpaceInfo | null) => void;
  clearError: () => void;
}

export const useSpaceStore = create<ISpaceStore>((set, get) => ({
  spaces: [],
  currentSpace: null,
  loading: false,
  error: null,

  fetchAllSpaces: async () => {
    set({ loading: true, error: null });

    try {
      const response = await getAllSpaces();

      console.log('API Response:', response);

      if (response.data.code === 0) {
        set({ spaces: response.data.data, loading: false });
      } else {
        console.error('API Error:', response.data);
        set({ error: response.data.message || 'Failed to fetch spaces', loading: false });
      }
    } catch (error) {
      console.error('Fetch Error:', error);
      set({ error: 'An error occurred while fetching spaces', loading: false });
    }
  },

  fetchSpaceById: async (space_id: string) => {
    set({ loading: true, error: null });

    try {
      const response = await getSpaceDetail(space_id);

      console.log('Space Detail API Response:', response);

      if (response.data.code === 0) {
        const spaceData = response.data.data;

        set({ currentSpace: spaceData, loading: false });

        return spaceData;
      }

      console.error('API Error:', response.data);
      set({ error: response.data.message || 'Failed to fetch space', loading: false });

      return null;
    } catch (error) {
      console.error('Fetch Error:', error);
      set({ error: 'An error occurred while fetching space details', loading: false });

      return null;
    }
  },

  addSpace: async (spaceData) => {
    set({ loading: true, error: null });

    try {
      const response = await createSpace(spaceData);

      console.log('API Response:', response);

      if (response.data.code === 0) {
        const newSpace = response.data.data;

        set((state) => ({
          spaces: [newSpace, ...state.spaces],
          currentSpace: newSpace,
          loading: false
        }));

        return newSpace;
      }

      console.error('API Error:', response.data);
      set({ error: response.data.message || 'Failed to create space', loading: false });

      return null;
    } catch (error) {
      console.error('Fetch Error:', error);
      set({ error: 'An error occurred while creating the space', loading: false });

      return null;
    }
  },

  startSpace: async (space_id: string) => {
    set({ loading: true, error: null });

    try {
      const response = await startSpace(space_id);

      console.log('Start Space API Response:', response);

      if (response.data.code === 0) {
        const updatedSpace = response.data.data;

        set((state) => ({
          spaces: state.spaces.map((space) =>
            space.id === updatedSpace.id ? updatedSpace : space
          ),
          currentSpace:
            state.currentSpace?.id === updatedSpace.id ? updatedSpace : state.currentSpace,
          loading: false
        }));

        return updatedSpace;
      }

      console.error('API Error:', response.data);
      set({ error: response.data.message || 'Failed to start space', loading: false });

      return null;
    } catch (error) {
      console.error('Fetch Error:', error);
      set({ error: 'An error occurred while starting the space', loading: false });

      return null;
    }
  },

  updateSpaceStatus: (space_id, status) => {
    const spaces = get().spaces;
    const newSpaces = spaces.map((space) => {
      if (space.id === space_id) {
        return { ...space, status };
      }

      return space;
    });

    set({ spaces: newSpaces });
  },

  deleteSpace: async (space_id: string) => {
    set({ loading: true, error: null });

    try {
      const response = await deleteSpaceApi(space_id);

      if (response.data.code === 0) {
        set((state) => ({
          spaces: state.spaces.filter((space) => space.id !== space_id),
          loading: false
        }));

        return true;
      }

      set({ error: response.data.message || 'Failed to delete space', loading: false });

      return false;
    } catch (error) {
      console.error('Delete Error:', error);
      set({ error: 'An error occurred while deleting the space', loading: false });

      return false;
    }
  },

  setCurrentSpace: (space) => set({ currentSpace: space }),

  clearError: () => set({ error: null })
}));
