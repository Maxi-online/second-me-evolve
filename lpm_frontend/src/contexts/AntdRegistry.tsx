'use client';

import { StyleProvider, createCache, extractStyle } from '@ant-design/cssinjs';
import { useServerInsertedHTML } from 'next/navigation';
import type React from 'react';

const StyledComponentsRegistry = ({ children }: { children: React.ReactNode }): JSX.Element => {
  const cache = createCache();

  useServerInsertedHTML(() => (
    <style dangerouslySetInnerHTML={{ __html: extractStyle(cache, true) }} id="antd" />
  ));

  return <StyleProvider cache={cache}>{children}</StyleProvider>;
};

export default StyledComponentsRegistry;
