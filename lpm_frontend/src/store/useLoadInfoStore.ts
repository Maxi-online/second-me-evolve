import { create } from 'zustand';
import { getCurrentInfo, type ILoadInfo } from '@/service/info';
import { EVENT } from '@/utils/event';

interface ILoadInfoState {
  loadInfo: ILoadInfo | null;
  fetchLoadInfo: () => void;
  setLoadInfo: (info: ILoadInfo | null) => void;
  clearLoadInfo: () => void;
}

export const useLoadInfoStore = create<ILoadInfoState>((set) => ({
  loadInfo: null,
  fetchLoadInfo: () => {
    getCurrentInfo().then((res) => {
      if (res.data.code === 0) {
        set({ loadInfo: res.data.data });
        localStorage.setItem('upload', JSON.stringify(res.data.data));
      } else if (res.data.code === 404) {
        dispatchEvent(new Event(EVENT.LOGOUT));
      }
    });
  },
  setLoadInfo: (info) => set({ loadInfo: info }),
  clearLoadInfo: () => set({ loadInfo: null })
}));
