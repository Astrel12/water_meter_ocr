import cv2
import argparse
from generate_image import read_config_file
from staged_inference import TwoStageInference

config = read_config_file("inference_config.json")

parser = argparse.ArgumentParser(description='Takes images from camera or from video, '
                                             'and launches inference of ocr string by means of two SqueezeDet models'
                                 )
parser.add_argument('--config', metavar='<path_to_config>', type=str,
                    default=None,
                    help='Path to config file')
parser.add_argument('--video', metavar='<path_to_video>', type=str,
                    default=None,
                    help='Path to video folder, rtsp url or integer index for capture string')
parser.add_argument('--demo_output', metavar='<path_to_demo_video>', type=str,
                    default=None,
                    help='Path to video file for demo output')
parser.add_argument('--demo_fps', metavar='<demo_video_fps>', type=float,
                    default=5.0,
                    help='Path to video file for demo output')
args = parser.parse_args()

video_source = cv2.VideoCapture(args.video)
if not video_source.isOpened():
    camera_index = 0
    try:
        camera_index = int(args.video)
    except ValueError as _:
        pass
    video_source = cv2.VideoCapture(camera_index)
    if not video_source.isOpened():
        video_source = None

if video_source is None:
    print("Failed to open video")
    exit(-1)

width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))

if args.demo_output:
    vw = cv2.VideoWriter(args.demo_output, fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=args.demo_fps,
                         frameSize=(width, height))
else:
    vw = None

inference = TwoStageInference(config)

process_next_frame = True

while process_next_frame:
    process_next_frame, image = video_source.read()
    if process_next_frame:
        boxes, ocr_result, scores = inference.predict(image)
        for i, b in enumerate(inference.last_boxes):
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)

        for i, b in enumerate(inference.last_digit_boxes):
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            cv2.putText(image, inference.last_digits[i], (int(b[0]), int(b[3])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)

        for i, b in enumerate(boxes):
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 4)
            cv2.putText(image, ocr_result[i], (int((b[0] * 2 + b[2]) / 3), int((b[1] + b[3]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1, cv2.LINE_AA)
            score_test = str(scores[i])
            cv2.putText(image, score_test, (int(b[0]), int(b[3])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Image", image)
        if vw:
            vw.write(image)

    key = cv2.waitKey(3)
    if key == 32 or key == 27 or key == 13:
        process_next_frame = False

video_source.release()
