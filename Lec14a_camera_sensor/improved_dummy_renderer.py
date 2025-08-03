#!/usr/bin/env python3

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

class ImprovedDummyRenderer:
    """Generates more realistic-looking dummy images for OpenVLA"""
    
    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height
        
    def render_scene(self, qpos, target_position, episode_step=0):
        """Render a more realistic-looking scene"""
        # Create base image with realistic background
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add gradient background (simulating lighting)
        for y in range(self.height):
            intensity = int(40 + (y / self.height) * 20)  # Gradient from dark to light
            image[y, :] = [intensity, intensity, intensity]
        
        # Add some "texture" noise
        noise = np.random.randint(-5, 6, (self.height, self.width, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Draw "floor" (bottom third)
        floor_y = int(self.height * 0.7)
        image[floor_y:, :] = [60, 60, 60]  # Dark gray floor
        
        # Add floor texture
        for x in range(0, self.width, 4):
            for y in range(floor_y, self.height, 4):
                if np.random.random() > 0.7:
                    image[y, x] = [80, 80, 80]
        
        # Draw robot/block as a more realistic object
        if len(qpos) >= 3:
            # Convert joint positions to screen coordinates
            center_x = int(self.width * 0.5 + qpos[0] * 30)
            center_y = int(self.height * 0.6 - qpos[1] * 30)
            center_x = np.clip(center_x, 10, self.width - 10)
            center_y = np.clip(center_y, 10, self.height - 10)
            
            # Draw "robot" as a 3D-looking box
            box_size = 15
            self._draw_3d_box(image, center_x, center_y, box_size, [200, 100, 50])  # Orange robot
            
            # Add shadow
            shadow_y = center_y + box_size + 5
            if shadow_y < self.height:
                cv2.ellipse(image, (center_x, shadow_y), (box_size//2, 3), 0, 0, 360, [20, 20, 20], -1)
        
        # Draw target as a realistic object
        target_x = int(self.width * 0.5 + target_position[0] * 30)
        target_y = int(self.height * 0.6 - target_position[1] * 30)
        target_x = np.clip(target_x, 10, self.width - 10)
        target_y = np.clip(target_y, 10, self.height - 10)
        
        # Draw target as a green cylinder
        self._draw_cylinder(image, target_x, target_y, 8, [50, 200, 50])
        
        # Add some "environment" details
        self._add_environment_details(image)
        
        # Add subtle lighting effects
        self._add_lighting_effects(image)
        
        return image
    
    def _draw_3d_box(self, image, x, y, size, color):
        """Draw a 3D-looking box"""
        # Main face
        cv2.rectangle(image, (x-size//2, y-size//2), (x+size//2, y+size//2), color, -1)
        
        # Top face (lighter)
        top_color = [min(255, c + 30) for c in color]
        cv2.rectangle(image, (x-size//2, y-size//2), (x+size//2, y-size//4), top_color, -1)
        
        # Right face (darker)
        right_color = [max(0, c - 30) for c in color]
        cv2.rectangle(image, (x+size//2, y-size//2), (x+size//4, y+size//2), right_color, -1)
        
        # Add edges
        cv2.rectangle(image, (x-size//2, y-size//2), (x+size//2, y+size//2), [0, 0, 0], 1)
    
    def _draw_cylinder(self, image, x, y, radius, color):
        """Draw a cylinder-like object"""
        # Main circle
        cv2.circle(image, (x, y), radius, color, -1)
        
        # Top circle (lighter)
        top_color = [min(255, c + 40) for c in color]
        cv2.circle(image, (x, y-2), radius-1, top_color, -1)
        
        # Bottom circle (darker)
        bottom_color = [max(0, c - 40) for c in color]
        cv2.circle(image, (x, y+2), radius-1, bottom_color, -1)
        
        # Outline
        cv2.circle(image, (x, y), radius, [0, 0, 0], 1)
    
    def _add_environment_details(self, image):
        """Add some environmental details"""
        # Add some "walls" in background
        wall_color = [30, 30, 30]
        cv2.line(image, (0, 0), (self.width, 0), wall_color, 2)
        cv2.line(image, (0, 0), (0, self.height), wall_color, 2)
        
        # Add some "grid lines" on floor
        for x in range(0, self.width, 20):
            cv2.line(image, (x, int(self.height * 0.7)), (x, self.height), [40, 40, 40], 1)
    
    def _add_lighting_effects(self, image):
        """Add subtle lighting effects"""
        # Add a subtle vignette effect
        center_x, center_y = self.width // 2, self.height // 2
        for y in range(self.height):
            for x in range(self.width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                darken = int((distance / max_distance) * 10)
                image[y, x] = np.clip(image[y, x] - darken, 0, 255)

# Example usage
if __name__ == "__main__":
    renderer = ImprovedDummyRenderer(128, 128)
    
    # Test with some sample data
    qpos = np.array([0.1, 0.2, 0.0])
    target = np.array([0.5, 0.3, 0.5])
    
    image = renderer.render_scene(qpos, target)
    
    # Save test image
    cv2.imwrite("test_improved_dummy.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print("Saved test_improved_dummy.png") 