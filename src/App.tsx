import { Layout } from './components/Layout';
import { UploadPage } from './pages/UploadPage';
import { SmoothingPage } from './pages/SmoothingPage';
import { ParametersPage } from './pages/ParametersPage';
import { ANNPage } from './pages/ANNPage';
import { ExportPage } from './pages/ExportPage';
import { useAppStore } from './stores/useAppStore';

function App() {
  const activeTab = useAppStore((s) => s.activeTab);

  return (
    <Layout>
      {activeTab === 'upload' && <UploadPage />}
      {activeTab === 'smoothing' && <SmoothingPage />}
      {activeTab === 'parameters' && <ParametersPage />}
      {activeTab === 'ann' && <ANNPage />}
      {activeTab === 'export' && <ExportPage />}
    </Layout>
  );
}

export default App;
