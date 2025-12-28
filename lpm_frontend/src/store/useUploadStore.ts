import { create } from 'zustand';
import { getUploadList, type IUploadInfo } from '@/service/upload';

interface UploadState {
  uploads: IUploadInfo[];
  total: number;
  loading: boolean;
  fetchUploadList: (refresh?: boolean) => Promise<void>;
  paginationRef: React.MutableRefObject<{ page_no: number; page_size: number }>;
  addUpload: (upload: IUploadInfo) => void;
  updateUploadStatus: (instance_id: string, status: IUploadInfo) => void;
  removeUpload: (instanceId: string) => void;
  reorderUploads: (instance_id: string | null) => void;
}

export const useUploadStore = create<UploadState>((set, get) => ({
  uploads: [],
  total: 0,
  loading: false,
  fetchUploadList: (refresh = true) => {
    set({ loading: true });

    if (refresh) {
      set({ uploads: [], total: 0 });
      set({ paginationRef: { current: { page_no: 1, page_size: 8 } } });
    } else {
      const pageNo = get().paginationRef.current.page_no;

      set({ paginationRef: { current: { page_no: pageNo + 1, page_size: 8 } } });
    }

    return getUploadList({ ...get().paginationRef.current })
      .then((res) => {
        if (res.data.code === 0) {
          const _data = res.data.data;

          if (refresh) {
            set({
              uploads: _data.items || [],
              total: _data.pagination.total
            });
          } else {
            const _list = get().uploads;
            const _newList = _data.items || [];

            _newList.forEach((item) => {
              const index = _list.findIndex((oldItem) => oldItem.instance_id === item.instance_id);

              if (index !== -1) {
                _list[index] = item;
              } else {
                _list.push(item);
              }
            });

            set({
              uploads: [..._list],
              total: _data.pagination.total
            });
          }
        }
      })
      .catch((err) => {
        console.error('Failed to fetch upload list:', err);
      })
      .finally(() => {
        set({ loading: false });
      });
  },
  paginationRef: { current: { page_no: 1, page_size: 8 } },
  addUpload: (upload) =>
    set((state) => {
      // If an upload with the same upload_name exists, replace it
      const index = state.uploads.findIndex((u) => u.instance_id === upload.instance_id);

      if (index !== -1) {
        const newUploads = [...state.uploads];

        newUploads[index] = upload;

        return { uploads: newUploads };
      }

      // If it doesn't exist, add it to the array
      return { uploads: [...state.uploads, upload] };
    }),
  updateUploadStatus: (instance_id, status) =>
    set((state) => ({
      uploads: state.uploads.map((upload) => (upload.instance_id === instance_id ? status : upload))
    })),
  removeUpload: (instanceId) => {
    set((state) => ({
      uploads: state.uploads.filter((upload) => !(upload.instance_id === instanceId))
    }));
  },
  reorderUploads: (instance_id: string | null) => {
    if (!instance_id) return;

    try {
      set((state) => {
        // Find the current user's upload
        const currentUserUploadIndex = state.uploads.findIndex(
          (upload) => upload.instance_id === instance_id
        );

        // If not found or already at the beginning, return the original state
        if (currentUserUploadIndex === -1 || currentUserUploadIndex === 0) return state;

        // Create a new array with the current user's upload at the beginning
        const reorderedUploads = [...state.uploads];
        const [currentUserUpload] = reorderedUploads.splice(currentUserUploadIndex, 1);

        reorderedUploads.unshift(currentUserUpload);

        return { uploads: reorderedUploads };
      });
    } catch (error) {
      console.error('Error reordering uploads:', error);
    }
  }
}));
