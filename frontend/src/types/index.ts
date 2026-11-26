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
  success: boolean;
  confidence_threshold: number;
  detections: DetectionItem[];
  all_detections: DetectionItem[];
  annotated_image_url: string;
  debug_annotated_image_url?: string | null;
  processing_time: number;
  game_id?: string;
  model_name?: string;
  image_width?: number;
  image_height?: number;
}

export interface GameInfo {
  game_id: string;
  game_name: string;
  display_name: string;
  icon?: string;
  description?: string;
  model_installed: boolean;
  model_version?: string;
  classes: Record<number, string> | string[];
  table?: {
    width_mm: number;
    height_mm: number;
  };
  ball_count: number;
  is_active: boolean;
}

export interface GamesResponse {
  current_game_id: string;
  games: GameInfo[];
  model_info: ModelInfoResponse;
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
