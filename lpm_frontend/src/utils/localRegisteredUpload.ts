import type { Upload } from '@/service/upload';

export const updateRegisteredUpload = (upload: Upload) => {
  const registeredUpload = JSON.parse(localStorage.getItem('registeredUpload') || '{}');
  const newRegisteredUpload = {
    ...registeredUpload,
    ...upload
  };

  localStorage.setItem('registeredUpload', JSON.stringify(newRegisteredUpload));
};
