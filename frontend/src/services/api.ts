import axios from "axios";
import { toast } from "react-hot-toast";
import type {
  HealthResponse,
  ModelInfoResponse,
  DetectionResponse,
  FolderDetectionResponse,
  CalibrationResponse,
  MappingResponse,
  ProjectStatisticsResponse
} from "../types";

// Get backend URL from LocalStorage or default to 127.0.0.1:8000
export const getBackendUrl = (): string => {
  return localStorage.getItem("backend_url") || "http://127.0.0.1:8000";
};

// Create axios instance with interceptors for error handling
const createApiClient = () => {
  const instance = axios.create({
    baseURL: getBackendUrl(),
    timeout: 15000, // 15 seconds timeout
  });

  // Request interceptor to update URL dynamically
  instance.interceptors.request.use((config) => {
    config.baseURL = getBackendUrl();
    return config;
  });

  // Response interceptor to handle common errors
  instance.interceptors.response.use(
    (response) => response,
    (error) => {
      let message = "An error occurred with the API connection.";
      if (error.response) {
        // Server responded with a status code other than 2xx
        message = error.response.data?.detail || error.response.data?.message || `Error ${error.response.status}`;
      } else if (error.request) {
        // Request was made but no response was received
        message = "Could not connect to the backend server. Verify the API is running and CORS is enabled.";
      } else {
        message = error.message;
      }
      toast.error(message, { id: "api-error" }); // Avoid duplicate toasts using id
      return Promise.reject(error);
    }
  );

  return instance;
};

const api = createApiClient();

export const apiService = {
  getHealth: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>("/api/health");
    return response.data;
  },

  getModelInfo: async (): Promise<ModelInfoResponse> => {
    const response = await api.get<ModelInfoResponse>("/api/model");
    return response.data;
  },

  loadModel: async (modelPath: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.post("/api/model/load", { model_path: modelPath });
    return response.data;
  },

  detectImage: async (file: File): Promise<DetectionResponse> => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post<DetectionResponse>("/api/detect/image", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  },

  detectFolder: async (folderPath: string): Promise<FolderDetectionResponse> => {
    const response = await api.post<FolderDetectionResponse>("/api/detect/folder", {
      folder_path: folderPath,
    });
    return response.data;
  },

  getCalibration: async (): Promise<CalibrationResponse> => {
    const response = await api.get<CalibrationResponse>("/api/calibration");
    return response.data;
  },

  mapCoordinates: async (pixelX: number, pixelY: number): Promise<MappingResponse> => {
    const response = await api.post<MappingResponse>("/api/mapping", {
      pixel_x: pixelX,
      pixel_y: pixelY,
    });
    return response.data;
  },

  getStatistics: async (): Promise<ProjectStatisticsResponse> => {
    const response = await api.get<ProjectStatisticsResponse>("/api/statistics");
    return response.data;
  },

  getLogs: async (lines: number = 100): Promise<string[]> => {
    const response = await api.get<string[]>(`/api/logs?lines=${lines}`);
    return response.data;
  },
};
