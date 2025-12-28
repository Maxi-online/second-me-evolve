const manualCopyFallback = (textToCopy: string): void => {
  const textArea = document.createElement('textarea');

  textArea.value = textToCopy;

  textArea.style.position = 'absolute';
  textArea.style.left = '-999999px';

  document.body.prepend(textArea);
  textArea.select();

  try {
    document.execCommand('copy');
  } catch (error) {
    console.error(error);
  } finally {
    textArea.remove();
  }
};

export const copyToClipboard = (textToCopy: string) => {
  return new Promise<void>((resolve, reject) => {
    if (navigator.clipboard && window.isSecureContext) {
      setTimeout(() => {
        navigator.clipboard
          .writeText(textToCopy)
          .then(() => {
            resolve();
          })
          .catch(() => {
            try {
              manualCopyFallback(textToCopy);
              resolve();
            } catch (manualError) {
              reject(manualError);
            }
          });
      });
    } else {
      try {
        manualCopyFallback(textToCopy);
        resolve();
      } catch {
        reject();
      }
    }
  });
};
