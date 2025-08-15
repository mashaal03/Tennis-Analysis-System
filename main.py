from utils import (
    read_video, 
    save_video, 
    measure_distance,
    draw_player_stats,
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_center_of_bbox
)
import constants 
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
import pandas as pd
from mini_court import MiniCourt
from copy import deepcopy
import numpy as np

def main():
    # Read video
    input_video_path = "input_videos/input_video6.mp4"
    video_frames, fps = read_video(input_video_path)
    
    # Initialize trackers
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolo8m_best.pt')

    # Get player and ball detections
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path="tracker_stubs/player_detections.pkl"
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path="tracker_stubs/ball_detections.pkl"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect court lines
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Filter to only the 2 players on court
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Initialize MiniCourt with homography capabilities
    mini_court = MiniCourt(video_frames[0], court_keypoints)

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # --- NEW: Convert positions to mini court coordinates using homography ---
    player_mini_court_detections = []
    ball_mini_court_detections = []

    for frame_num, player_bbox in enumerate(player_detections):
        # Transform player positions
        foot_positions = {player_id: get_foot_position(bbox) for player_id, bbox in player_bbox.items()}
        if foot_positions:
            points = np.array(list(foot_positions.values()))
            ids = list(foot_positions.keys())
            transformed_points = mini_court.transform_points(points)
            transformed_dict = {ids[i]: transformed_points[i] for i in range(len(ids))}
            player_mini_court_detections.append(transformed_dict)
        else:
            player_mini_court_detections.append({})
        
        # Transform ball position for the same frame
        if ball_detections[frame_num].get(1):
            ball_pos = get_center_of_bbox(ball_detections[frame_num][1])
            transformed_ball = mini_court.transform_points([ball_pos])
            if transformed_ball.size > 0:
                ball_mini_court_detections.append({1: transformed_ball[0]})
            else:
                 ball_mini_court_detections.append({})
        else:
            ball_mini_court_detections.append({})
            
    # --- End of new conversion logic ---
    
    # Initialize player stats tracking
    player_stats_data = [{
        'frame_num': 0, 'player_1_number_of_shots': 0, 'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0, 'player_1_total_player_speed': 0, 'player_1_last_player_speed': 0,
        'player_2_number_of_shots': 0, 'player_2_total_shot_speed': 0, 'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0, 'player_2_last_player_speed': 0,
    }]

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / fps

        # Get distance covered by the ball in pixels on the mini court
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1]
        )
        # Convert this pixel distance to meters
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Determine which player hit the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(
            player_positions[player_id], ball_mini_court_detections[start_frame][1]
        ))

        # Calculate opponent speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id]
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        # Update stats
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent
        player_stats_data.append(current_player_stats)

    # Prepare stats DataFrame for drawing
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left').ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / (player_stats_data_df['player_1_number_of_shots'] + 1e-6)
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / (player_stats_data_df['player_2_number_of_shots'] + 1e-6)
    player_stats_data_df = player_stats_data_df.fillna(0)

    # --- Draw Output ---
    # Draw Player and Ball Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    # Draw court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    # Draw Mini Court and positions
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))
    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)
    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save the final video
    save_video(output_video_frames, "output_videos/output_video.mp4", fps)

if __name__ == "__main__":
    main()
