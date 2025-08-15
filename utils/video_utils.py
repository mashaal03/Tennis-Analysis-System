#
# Replace the content of your video_utils.py with this
#
import cv2

def read_video(video_path):
    """
    Reads a video file and returns its frames and FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None
    
    # Get the original frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    # Return both the list of frames and the FPS
    return frames, fps

def save_video(output_video_frames, output_video_path, fps):
    """
    Saves a list of frames as a video file with the specified FPS.
    """
    if not output_video_frames:
        print("Warning: No frames to save.")
        return

    height, width = output_video_frames[0].shape[:2]
    
    if output_video_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Compatible with .mp4
    else:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Best for .avi

    # Use the fps value passed into the function
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()