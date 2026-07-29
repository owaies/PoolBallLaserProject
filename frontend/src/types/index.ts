export interface DetectionItem {
  detection_id: string;
  class_name: string;
  confidence: number;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  center_x: number;
  center_y: number;
  width: number;
  height: number;
  is_accepted?: boolean;
  rejection_reason?: string | null;
  aspect_ratio?: number;
  circularity?: number;
}

export interface DetectionResponse {
  detections: DetectionItem[];
  all_detections: DetectionItem[];
  annotated_image_url: string;
  debug_annotated_image_url?: string | null;
  processing_time: number;
}

export interface FolderDetectionResponse {
  total_images: number;
  total_detections: number;
  csv_path: string;
  json_path: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime: number;
  gpu_available: boolean;
  current_model: string | null;
}

export interface ModelInfoResponse {
  model_name: string;
  classes: Record<number, string>;
  image_size: number;
  confidence_threshold: number;
  device: string;
}

export interface CalibrationResponse {
  camera_matrix: number[][] | null;
  distortion_coefficients: number[] | null;
  is_calibrated: boolean;
}

export interface MappingResponse {
  world_x: number;
  world_y: number;
}

export interface ProjectStatisticsResponse {
  number_of_images: number;
  number_of_detections: number;
  average_confidence: number;
  model_version: string;
  training_date: string | null;
}
