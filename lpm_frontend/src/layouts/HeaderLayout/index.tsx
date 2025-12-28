import { Header } from '@/layouts/HeaderLayout/Header';
import StyledComponentsRegistry from '@/contexts/AntdRegistry';

const HeaderLayout = ({ children }: { children: React.ReactNode }) => {
  return (
    <StyledComponentsRegistry>
      <Header />
      <div className="flex-1 overflow-hidden">{children}</div>
    </StyledComponentsRegistry>
  );
};

export default HeaderLayout;
