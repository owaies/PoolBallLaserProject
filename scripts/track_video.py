import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

BALL_COLORS = {
    'cue-ball': (255, 255, 255),
    'white-ball': (255, 255, 255),
    'red-ball': (0, 0, 230),
    'yellow-ball': (0, 215, 255),
    'green-ball': (0, 180, 0),
    'brown-ball': (19, 69, 139),
    'blue-ball': (235, 120, 0),
    'pink-ball': (180, 105, 255),
    'black-ball': (30, 30, 30),
}

def track_snooker_video(video_source, model_path, output_path=None, show=False):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(int(video_source) if str(video_source).isdigit() else video_source)
    if not cap.isOpened():
        print(f'Error opening video source: {video_source}')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f'Tracking video {video_source} with model {model_path}...')
    track_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
        annotated_frame = frame.copy()

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            names = results[0].names

            for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                c_name = names.get(cls_id, 'ball')
                color = BALL_COLORS.get(c_name, (0, 255, 0))

                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(center)
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                # Draw trajectory
                pts = track_history[track_id]
                for i in range(1, len(pts)):
                    thickness = int(1.5 + (i / len(pts)) * 2)
                    cv2.line(annotated_frame, pts[i - 1], pts[i], color, thickness)

                # Draw bounding box & ID label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f'#{track_id} {c_name} {int(conf * 100)}%'
                cv2.putText(annotated_frame, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        if writer:
            writer.write(annotated_frame)

        if show:
            cv2.imshow('Snooker Ball Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
        print(f'Annotated tracking video saved to: {output_path}')
    if show:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Video path or webcam index (0)')
    parser.add_argument('--model', type=str, default='models/snooker/best.pt', help='Trained YOLO weights')
    parser.add_argument('--output', type=str, default=None, help='Output annotated video path')
    parser.add_argument('--show', action='store_true', help='Display window')
    args = parser.parse_args()

    track_snooker_video(args.source, args.model, args.output, args.show)
