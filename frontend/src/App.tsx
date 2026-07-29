import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import DashboardLayout from "./layouts/DashboardLayout";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import Detection from "./pages/Detection";
import Batch from "./pages/Batch";
import Mapping from "./pages/Mapping";
import Calibration from "./pages/Calibration";
import Reports from "./pages/Reports";
import Logs from "./pages/Logs";
import Settings from "./pages/Settings";
import About from "./pages/About";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<DashboardLayout />}>
            <Route index element={<Home />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="detection" element={<Detection />} />
            <Route path="batch" element={<Batch />} />
            <Route path="mapping" element={<Mapping />} />
            <Route path="calibration" element={<Calibration />} />
            <Route path="reports" element={<Reports />} />
            <Route path="logs" element={<Logs />} />
            <Route path="settings" element={<Settings />} />
            <Route path="about" element={<About />} />
          </Route>
        </Routes>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 3500,
            style: {
              background: "#18181b",
              color: "#f4f4f5",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              fontSize: "12px",
              fontFamily: "monospace",
            },
          }}
        />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
