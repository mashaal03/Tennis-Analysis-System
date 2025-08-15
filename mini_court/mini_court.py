import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self, frame, original_court_keypoints):
        """
        Initializes the MiniCourt object.
        Computes the homography matrix for perspective transformation.
        """
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        # Set up positions for drawing the mini court background
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

        # Compute and store the homography matrix
        self.homography_matrix = self._compute_homography_matrix(original_court_keypoints)

    def _compute_homography_matrix(self, original_court_keypoints):
        """
        Computes the homography matrix to map points from the original video frame
        to the top-down mini court view.
        """
        # Define the 4 corner points of the doubles court in the original video
        # Order: Top-left, Top-right, Bottom-right, Bottom-left
        source_points = np.array([
            [original_court_keypoints[0], original_court_keypoints[1]],
            [original_court_keypoints[2], original_court_keypoints[3]],
            [original_court_keypoints[6], original_court_keypoints[7]],
            [original_court_keypoints[4], original_court_keypoints[5]]
        ], dtype=np.float32)

        # Define the corresponding 4 corner points on our mini court drawing
        destination_points = np.array([
            [self.drawing_key_points[0], self.drawing_key_points[1]],
            [self.drawing_key_points[2], self.drawing_key_points[3]],
            [self.drawing_key_points[6], self.drawing_key_points[7]],
            [self.drawing_key_points[4], self.drawing_key_points[5]]
        ], dtype=np.float32)

        # Calculate the homography matrix using OpenCV
        matrix, _ = cv2.findHomography(source_points, destination_points)
        return matrix

    def transform_points(self, points):
        """
        Transforms a list of points from the video's perspective to the mini court's
        top-down perspective using the pre-computed homography matrix.
        """
        if len(points) == 0:
            return []
        
        # Reshape points to (N, 1, 2) for perspectiveTransform function
        reshaped_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # Apply the perspective transformation
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.homography_matrix)

        # Reshape back to a simple list of (x, y) coordinates
        return transformed_points.reshape(-1, 2)

    def convert_meters_to_pixels(self, meters):
        """Converts meters to pixels for drawing the mini court layout."""
        return convert_meters_to_pixel_distance(meters,
                                               constants.DOUBLE_LINE_WIDTH,
                                               self.court_drawing_width)
    
    def set_court_drawing_key_points(self):
        """Sets the keypoints for drawing the mini court layout."""
        drawing_key_points = [0]*28
        # point 0 
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 11
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 
        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        """Sets the lines for drawing the mini court."""
        self.lines = [
            (0, 2), (4, 5), (6, 7), (1, 3), (0, 1),
            (8, 9), (10, 11), (2, 3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):
        """Draws the court lines and net on the frame."""
        # Draw lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)
        
        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)
        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        """
        Draws points on the mini court, clipping them to the background rectangle boundaries.
        """
        for frame_num, frame in enumerate(frames):
            if frame_num < len(positions):
                for _, position in positions[frame_num].items():
                    x, y = position[0], position[1]

                    # Define the boundaries of the drawable background rectangle for clipping
                    min_x = self.start_x
                    max_x = self.end_x
                    min_y = self.start_y
                    max_y = self.end_y 

                    # Clip the coordinates to be within the background rectangle
                    x = int(np.clip(x, min_x, max_x))
                    y = int(np.clip(y, min_y, max_y))
                    
                    cv2.circle(frame, (x, y), 5, color, -1)
        return frames
