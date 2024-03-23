
'''
This script is used to track the position of the finger in the video.

Inputs: 
    video_in: video to be tracked
    fps: framerate of the video
    start_time: time in the first frame of the video

Outputs:
    video_out: video with the tracking
    txt_out: text file with the position of the finger in each frame and the time. [x,y,t]

How to use: 
    1. Run the script
    2. Change the start_time parameter to the time in the first frame.
    3. Change the 
    3. Restart the script
    4. Select the ROI of the finger in the first frame
    5. press space to start the tracking
    6. press q to stop the tracking
'''

import cv2
import numpy as np
import argparse

def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = (seconds - int(seconds)) * 1000
    return "%d:%02d:%02d.%03d" % (hours, minutes, int(seconds), milliseconds)



def main(args):
    fps = args.fps
    video_in = args.video_in
    start_time = args.start_time

    video_out = video_in[:-4] + "_tracked.mov"
    txt_out = video_in[:-4] + ".txt"

    dt = 1/fps
    hours =    float(start_time[0:2])
    minuts =   float(start_time[3:5])
    seconds =  float(start_time[6:])
    start_seconds = (hours*60 +minuts) * 60 + seconds

    video = cv2.VideoCapture(video_in)
    ret, frame = video.read()

    bbox = cv2.selectROI(frame, False)
    tracker = cv2.TrackerMIL_create()
    tracker.init(frame, bbox)

    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    finger_position = []
    counter = 0
    while True:
        counter += 1
        ret, frame = video.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x_coordinate = x + w/2
            y_coordinate = y + h/2
            start_seconds += dt
            finger_position.append([x_coordinate, y_coordinate, start_seconds])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(convert_seconds(start_seconds)), (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    np.savetxt(txt_out, finger_position)
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video input.')

    parser.add_argument('--fps', type=int, required=True, help='Frames per second of the video')
    parser.add_argument('--video_in', type=str, required=True, help='Input video file path')
    parser.add_argument('--start_time', type=str, required=True, help='Start time in format HH:MM:SS.mmm')

    args = parser.parse_args()
    main(args)


