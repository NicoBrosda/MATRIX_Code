import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# === Global Variables ===
window_name = "Select corners"
window_size = (800, 800)
points = []
cache_point = None
zoom_start = None
zoom_region = None
zoom_mode = False
pixel_size_mm = 25.4 / 9600  # Default pixel size in mm
dpi = 9600  # Default DPI
scale_factor = 0.1  # Default scale factor for visualization

# Initialize variables that will be used in multiple functions
original_hole_points = []
warped_second_bottom_edge = []
current_results_path = None

# === Results Path Setup ===
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/WedgePos/')  # Base results directory

class SelectionDialog:
    def __init__(self, default_results_path):
        self.root = tk.Tk()
        self.root.title("MATRIX Code - Selection")
        self.root.geometry("500x300")
        
        self.choice = None
        self.image_path = None
        self.results_path = None
        self.default_results_path = default_results_path
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Workflow selection
        ttk.Label(main_frame, text="Select Workflow:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.workflow_var = tk.StringVar(value="new")
        
        ttk.Radiobutton(main_frame, text="Process new image", 
                       variable=self.workflow_var, value="new").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Load existing data", 
                       variable=self.workflow_var, value="load").grid(row=2, column=0, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Reload and replot only", 
                       variable=self.workflow_var, value="replot").grid(row=3, column=0, sticky=tk.W)
        
        # Results path selection (only for new images)
        self.results_frame = ttk.LabelFrame(main_frame, text="Results Path (for new images)")
        self.results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.results_path_var = tk.StringVar(value=str(self.default_results_path))
        ttk.Entry(self.results_frame, textvariable=self.results_path_var, width=50).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(self.results_frame, text="Browse", command=self.browse_results_path).grid(row=0, column=1, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Continue", command=self.on_continue).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).grid(row=0, column=1, padx=5)
        
        # Update results frame visibility based on workflow
        self.workflow_var.trace_add('write', self.update_results_frame)
        self.update_results_frame()
        
    def update_results_frame(self, *args):
        """Show/hide results path selection based on workflow choice"""
        if self.workflow_var.get() == "new":
            self.results_frame.grid()
        else:
            self.results_frame.grid_remove()
    
    def browse_results_path(self):
        path = filedialog.askdirectory(initialdir=str(self.default_results_path))
        if path:
            self.results_path_var.set(path)
    
    def on_continue(self):
        if self.workflow_var.get() == "new":
            self.image_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg")]
            )
            if not self.image_path:
                messagebox.showerror("Error", "Please select an image")
                return
            self.results_path = Path(self.results_path_var.get())
        else:
            # For both 'load' and 'replot' options, select a results directory
            self.image_path = filedialog.askdirectory(
                title="Select Results Directory",
                initialdir=str(self.default_results_path)
            )
            if not self.image_path:
                messagebox.showerror("Error", "Please select a results directory")
                return
            self.results_path = Path(self.image_path)  # Use the selected directory as results path
        
        self.choice = self.workflow_var.get()
        self.root.quit()
    
    def on_cancel(self):
        self.root.quit()
    
    def show(self):
        self.root.mainloop()
        self.root.destroy()
        return {
            'choice': self.choice,
            'image_path': Path(self.image_path) if self.image_path else None,
            'results_path': self.results_path
        }

def get_user_selection():
    """Get user selection using tkinter dialog"""
    default_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/WedgePos/')
    dialog = SelectionDialog(default_path)
    return dialog.show()

def select_file(title="Select file", initial_dir=None, file_types=None):
    """Open a file dialog to select a file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir,
        filetypes=file_types
    )
    return Path(file_path) if file_path else None

def select_directory(title="Select directory", initial_dir=None):
    """Open a directory dialog to select a folder"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    dir_path = filedialog.askdirectory(
        title=title,
        initialdir=initial_dir
    )
    return Path(dir_path) if dir_path else None

def get_user_choice():
    """Get user choice for workflow"""
    while True:
        print("\nChoose workflow:")
        print("1. Process new image")
        print("2. Load existing data")
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Please enter 1 or 2.")

def get_results_path():
    """Get results path from user"""
    default_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/WedgePos/')
    print(f"\nCurrent default results path: {default_path}")
    print("Do you want to use a different path? (y/n)")
    if input().strip().lower() == 'y':
        new_path = select_directory("Select results directory", str(default_path))
        if new_path:
            return new_path
    return default_path

def load_existing_data(results_dir):
    """Load existing data from a results directory"""
    # Try to load corner data
    corner_data = None
    try:
        with open(results_dir / "corner_data.json", 'r') as f:
            corner_data = json.load(f)
    except FileNotFoundError:
        print("corner_data.json not found in selected directory")
        return None
    
    # Try to load measurements data
    measurements_data = None
    try:
        with open(results_dir / "measurements_data.json", 'r') as f:
            measurements_data = json.load(f)
    except FileNotFoundError:
        print("measurements_data.json not found in selected directory")
        return None
    
    # Load the original image
    image_path = None
    if 'image_path' in corner_data:
        image_path = Path(corner_data['image_path'])
        if not image_path.exists():
            print(f"Original image not found at: {image_path}")
            print("Please select the original image")
            image_path = select_file("Select original image", file_types=[("Image files", "*.bmp *.png *.jpg *.jpeg")])
            if not image_path:
                return None
    else:
        print("Image path not found in data. Please select the original image")
        image_path = select_file("Select original image", file_types=[("Image files", "*.bmp *.png *.jpg *.jpeg")])
        if not image_path:
            return None
    
    # Load global variables from the data
    global pixel_size_mm, dpi
    pixel_size_mm = corner_data.get('pixel_size_mm', 25.4 / 9600)
    dpi = corner_data.get('dpi', 9600)
    
    return {
        'corner_data': corner_data,
        'measurements_data': measurements_data,
        'image_path': image_path
    }

def process_new_image():
    """Process a new image"""
    print("\nSelect the image to process")
    image_path = select_file("Select image", file_types=[("Image files", "*.bmp *.png *.jpg *.jpeg")])
    if not image_path:
        print("No image selected. Exiting.")
        return None
    
    return {
        'image_path': image_path,
        'is_new': True
    }

def save_points_data(points_data, filename="points_data.json"):
    """Save point selections and measurements to a JSON file"""
    try:
        # Ensure the directory exists
        current_results_path.mkdir(parents=True, exist_ok=True)
        
        # Create a copy of the data to modify
        data_to_save = points_data.copy()
        
        # Convert Path objects to strings
        if 'image_path' in data_to_save:
            data_to_save['image_path'] = str(data_to_save['image_path'])
        
        # Convert numpy arrays and tuples to lists for JSON serialization
        # Ensure all coordinates are integers and in original image space
        if 'corners' in data_to_save:
            data_to_save['corners'] = [(int(x), int(y)) for x, y in data_to_save['corners']]
            print(f"Corner coordinates saved: {data_to_save['corners']}")
        
        # Add global variables to the data
        data_to_save['pixel_size_mm'] = pixel_size_mm
        data_to_save['dpi'] = dpi
        data_to_save['scale_factor'] = scale_factor
        
        # Save the data
        with open(current_results_path / filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
            
        print(f"Saved data to {current_results_path / filename}")
        return True
    except Exception as e:
        print(f"Error saving points data: {e}")
        return False

def load_points_data(filename="points_data.json"):
    """Load point selections and measurements from a JSON file"""
    try:
        file_path = current_results_path / filename
        if not file_path.exists():
            print(f"File not found: {filename}")
            return None
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded data from {file_path}")
        print(f"Corner coordinates loaded: {data.get('corners', [])}")
        
        # Convert string paths back to Path objects
        if 'image_path' in data:
            data['image_path'] = Path(data['image_path'])
            
        # Ensure coordinates are tuples
        if 'corners' in data:
            data['corners'] = [(int(x), int(y)) for x, y in data['corners']]
        if 'hole_points' in data:
            data['hole_points'] = [(int(x), int(y)) for x, y in data['hole_points']]
        if 'edge_points' in data:
            data['edge_points'] = [(int(x), int(y)) for x, y in data['edge_points']]
            
        # Validate corner coordinates
        if 'corners' in data:
            corners = data['corners']
            if len(corners) != 4:
                print("Error: Invalid number of corners")
                return None
            
            # Basic sanity check for corner coordinates
            img_path = data.get('image_path')
            if img_path and img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    for x, y in corners:
                        if not (0 <= x < w and 0 <= y < h):
                            print(f"Warning: Corner coordinate ({x}, {y}) outside image bounds ({w}, {h})")
            
        # Update global variables if they exist in the data
        global pixel_size_mm, dpi, scale_factor
        if 'pixel_size_mm' in data:
            pixel_size_mm = data['pixel_size_mm']
        if 'dpi' in data:
            dpi = data['dpi']
        if 'scale_factor' in data:
            scale_factor = data['scale_factor']
            
        return data
    except Exception as e:
        print(f"Error loading points data: {e}")
        return None

def calculate_edge_lengths(corners, is_display_space=False):
    """Calculate lengths of all edges between corners
    Args:
        corners: List of corner coordinates
        is_display_space: Whether the coordinates are in display space (scaled) or original space
    """
    lengths = []
    for i in range(len(corners)):
        x1, y1 = corners[i]
        x2, y2 = corners[(i+1)%len(corners)]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        # If in display space, convert length to original space for mm calculation
        if is_display_space:
            length = length / scale_factor
        lengths.append(length)
    return lengths

def save_plot(fig, filename, dpi=150):
    """Save a matplotlib figure with consistent settings"""
    # Save with basic parameters first
    temp_path = current_results_path / f"temp_{filename}"
    fig.savefig(temp_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # If it's a JPEG, recompress it with OpenCV
    if filename.lower().endswith(('.jpg', '.jpeg')):
        img = cv2.imread(str(temp_path))
        if img is not None:
            cv2.imwrite(str(current_results_path / filename), img, 
                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            temp_path.unlink()  # Delete temporary file
        else:
            # If reading fails, just rename the temp file
            temp_path.rename(current_results_path / filename)
    else:
        # For non-JPEG files, just rename the temp file
        temp_path.rename(current_results_path / filename)

def save_cv2_image(image, filename, max_size=2000):
    """Save an OpenCV image with size limiting and compression
    
    Args:
        image: OpenCV image to save
        filename: Output filename
        max_size: Maximum dimension (width or height) of the output image
    """
    # Calculate scaling factor if image is too large
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    # Save with JPEG compression
    cv2.imwrite(str(current_results_path / filename), image, 
                [cv2.IMWRITE_JPEG_QUALITY, 90])  # 90% JPEG quality

def order_corners(corners):
    """Order corners as: top-left (C1), top-right (C2), bottom-left (C3), bottom-right (C4)"""
    # Sort corners based on y-coordinate (top to bottom)
    sorted_by_y = sorted(corners, key=lambda p: p[1])
    
    # Get top two and bottom two points
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    
    # Sort top points by x coordinate
    top_left, top_right = sorted(top_points, key=lambda p: p[0])
    
    # Sort bottom points by x coordinate
    bottom_left, bottom_right = sorted(bottom_points, key=lambda p: p[0])
    
    # Return corners in the correct order: top-left, top-right, bottom-left, bottom-right
    ordered = [top_left, top_right, bottom_left, bottom_right]
    print(f"Ordered corners: {ordered}")
    return ordered

def draw_edges_with_lengths(img, corners, color=(0, 255, 0), thickness=2, text_color=None, font_scale=0.8):
    """Draw edges between corners and add length labels.
    Note: This function assumes coordinates are in display space and converts measurements accordingly.
    """
    if text_color is None:
        text_color = color
    
    h, w = img.shape[:2]
    
    # Define edges in the original order
    edges = [
        (0, 1),  # Top edge
        (1, 3),  # Right edge
        (3, 2),  # Bottom edge
        (2, 0)   # Left edge
    ]
    
    # Calculate appropriate thickness based on image size
    base_thickness = max(1, min(4, int(max(w, h) * 0.002)))  # Ensure thickness is between 1 and 4
    thickness = min(base_thickness, thickness)  # Use the smaller of the two
    
    # Draw edges and labels
    for i, (start_idx, end_idx) in enumerate(edges):
        pt1 = corners[start_idx]
        pt2 = corners[end_idx]
        
        # Draw edge
        cv2.line(img, tuple(map(int, pt1)), tuple(map(int, pt2)), color, thickness)
        
        # Calculate length in original space
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length_pixels = np.sqrt(dx*dx + dy*dy)
        # Convert display space length to original space for mm calculation
        length_mm = (length_pixels / scale_factor) * pixel_size_mm
        
        # Calculate label position
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        
        # Add offset to avoid overlapping with the line
        if abs(dx) > abs(dy):  # Horizontal-ish line
            offset_y = -20 if mid_y > h/2 else 20
            offset_x = 0
        else:  # Vertical-ish line
            offset_x = -20 if mid_x > w/2 else 20
            offset_y = 0
        
        # Ensure label stays within image bounds
        label = f"{length_mm:.1f}mm"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        
        # Adjust position to keep label inside image
        label_x = mid_x + offset_x
        label_y = mid_y + offset_y
        
        # Keep label inside image bounds
        if label_x + label_size[0] > w:
            label_x = w - label_size[0] - 5
        if label_x < 0:
            label_x = 5
        if label_y + label_size[1] > h:
            label_y = h - 5
        if label_y - label_size[1] < 0:
            label_y = label_size[1] + 5
        
        # Draw length label
        cv2.putText(img, label, (int(label_x), int(label_y)),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
    
    return img

def visualize_corners(image, display_corners, image_path, timestamp=None, title="Corner Selection"):
    """Visualize the selected corners on the original image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    # Draw edges and lengths using the display corners for visualization
    img = draw_edges_with_lengths(img, display_corners)
    
    # Draw corners and labels on top
    for i, (x, y) in enumerate(display_corners, 1):
        x = max(0, min(int(x), w-1))
        y = max(0, min(int(y), h-1))
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(img, f"C{i}", (x+15, y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    save_plot(plt.gcf(), "01_corner_selection.jpg")
    
    # Convert display coordinates back to original space for saving
    original_corners = [(int(x / scale_factor), int(y / scale_factor)) for x, y in display_corners]
    print(f"[COORDS] Saving original corner coordinates: {original_corners}")
    
    # Save corner data with original coordinates
    corner_data = {
        "corners": original_corners,
        "timestamp": timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "image_path": str(image_path),
        "image_dimensions": {"width": int(w / scale_factor), "height": int(h / scale_factor)},
        "scale_factor": scale_factor
    }
    save_points_data(corner_data, "corner_data.json")

def calculate_perpendicular_distance(point, edge_points, is_display_space=False):
    """Calculate perpendicular distance from a point to a line defined by two points
    Args:
        point: The point to calculate distance from
        edge_points: List of two points defining the line
        is_display_space: Whether the coordinates are in display space (scaled) or original space
    """
    x0, y0 = point
    x1, y1 = edge_points[0]
    x2, y2 = edge_points[1]
    
    # Calculate line equation parameters (ax + by + c = 0)
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    
    # Calculate perpendicular distance
    numerator = abs(a*x0 + b*y0 + c)
    denominator = np.sqrt(a*a + b*b)
    
    distance = numerator / denominator if denominator != 0 else 0
    
    # If in display space, convert distance to original space
    if is_display_space:
        distance = distance / scale_factor
        
    return distance

def visualize_final_results(output, measurements_data, title="Final Results with Measurements", 
                          cross_size=None, line_thickness=None, font_scale=None, 
                          text_thickness=None, text_offset=None, border_colors=None,
                          timestamp=None):
    """Visualize the final results with all measurements.
    Note: This function assumes input coordinates are in display space and converts measurements accordingly.
    """
    img = output.copy()
    warped_h, warped_w = img.shape[:2]
    
    # If visualization constants are not provided, calculate them
    if cross_size is None:
        base_size = max(warped_w, warped_h)
        cross_size = max(1, int(base_size * 0.002))  # Ensure at least 1 pixel
        line_thickness = max(1, int(base_size * 0.001))  # Ensure at least 1 pixel
        font_scale = base_size * 0.0005
        text_thickness = max(1, int(base_size * 0.0005))  # Ensure at least 1 pixel
        text_offset = int(base_size * 0.01)
    
    if border_colors is None:
        border_colors = {
            'edges': (0, 255, 0),
            'second_bottom': (255, 0, 0)
        }
    
    # Draw border edges and lengths
    corners = [(0, 0), (warped_w-1, 0), (0, warped_h-1), (warped_w-1, warped_h-1)]
    img = draw_edges_with_lengths(img, corners, color=border_colors['edges'], 
                                thickness=line_thickness, font_scale=font_scale)
    
    # Draw second bottom edge points and line
    edge_points = measurements_data['edge_points']
    for (x, y) in edge_points:
        cv2.circle(img, (x, y), cross_size, border_colors['second_bottom'], -1)
    if len(edge_points) >= 2:
        cv2.line(img, edge_points[0], edge_points[1], 
                border_colors['second_bottom'], line_thickness*2)
        
        # Add length label for second bottom edge (converting to original space for measurement)
        dx = edge_points[1][0] - edge_points[0][0]
        dy = edge_points[1][1] - edge_points[0][1]
        length_pixels = np.sqrt(dx*dx + dy*dy)
        length_mm = (length_pixels / scale_factor) * pixel_size_mm
        mid_x = (edge_points[0][0] + edge_points[1][0]) // 2
        mid_y = (edge_points[0][1] + edge_points[1][1]) // 2
        cv2.putText(img, f"SBE: {length_mm:.1f}mm", 
                   (mid_x + text_offset, mid_y + text_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, border_colors['second_bottom'], 
                   text_thickness)
    
    # Draw hole points and measurements
    hole_points = measurements_data['hole_points']
    for idx, (x, y) in enumerate(hole_points, start=1):
        # Calculate point color based on index
        hue = int(180 * (idx - 1) / len(hole_points))
        point_color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        point_color = tuple(map(int, point_color))
        
        # Draw point
        cv2.line(img, (x-cross_size, y), (x+cross_size, y), point_color, line_thickness)
        cv2.line(img, (x, y-cross_size), (x, y+cross_size), point_color, line_thickness)
        
        # Calculate and draw distances (converting to original space for measurements)
        distances = {
            'LE': ((x / scale_factor) * pixel_size_mm, (0, y), (x, y)),
            'RE': (((warped_w - x) / scale_factor) * pixel_size_mm, (x, y), (warped_w-1, y)),
            'TE': ((y / scale_factor) * pixel_size_mm, (x, 0), (x, y)),
            'BE': (((warped_h - y) / scale_factor) * pixel_size_mm, (x, y), (x, warped_h-1))
        }
        
        for edge_name, (dist, start, end) in distances.items():
            cv2.line(img, start, end, point_color, line_thickness)
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            cv2.putText(img, f"P{idx}-{edge_name}: {dist:.1f}mm",
                       (mid_x + text_offset, mid_y + text_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, point_color, text_thickness)
        
        # Calculate and draw SBE distance
        if len(edge_points) >= 2:
            dist_second_bottom = calculate_perpendicular_distance((x, y), edge_points, is_display_space=True) * pixel_size_mm
            x1, y1 = edge_points[0]
            x2, y2 = edge_points[1]
            
            # Calculate intersection point
            if x2 - x1 == 0:  # Vertical line
                x_intersect = x1
                y_intersect = y
            else:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                m_perp = -1/m if m != 0 else float('inf')
                if m_perp == float('inf'):
                    x_intersect = x
                    y_intersect = m * x + b
                else:
                    b_perp = y - m_perp * x
                    x_intersect = (b_perp - b) / (m - m_perp)
                    y_intersect = m * x_intersect + b
            
            cv2.line(img, (x, y), (int(x_intersect), int(y_intersect)), point_color, line_thickness)
            mid_x = (x + int(x_intersect)) // 2
            mid_y = (y + int(y_intersect)) // 2
            cv2.putText(img, f"P{idx}-SBE: {dist_second_bottom:.1f}mm",
                       (mid_x + text_offset, mid_y + text_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, point_color, text_thickness)
    
    plt.figure(figsize=(16, 16))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=16)
    plt.axis('off')
    save_plot(plt.gcf(), "03_final_measurements.jpg")
    
    # Save measurements data
    measurements_data = {
        "hole_points": hole_points,
        "edge_points": edge_points,
        "image_dimensions": {"width": warped_w, "height": warped_h},
        "pixel_size_mm": pixel_size_mm,
        "corners": corners if 'corners' in locals() else measurements_data.get('corners', []),
        "timestamp": timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "is_original_space": False
    }
    
    # Ensure all coordinates are integers
    measurements_data["corners"] = [(int(x), int(y)) for x, y in measurements_data["corners"]]
    measurements_data["hole_points"] = [(int(x), int(y)) for x, y in measurements_data["hole_points"]]
    measurements_data["edge_points"] = [(int(x), int(y)) for x, y in measurements_data["edge_points"]]
    
    # save_points_data(measurements_data, "measurements_data.json")

# === Zoom functionality ===
def mouse_callback(event, x, y, flags, param):
    global zoom_start, zoom_region, zoom_mode, cache_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if zoom_mode and zoom_region:
            x_min, y_min, x_max, y_max = zoom_region
            # Map the click coordinates back to the original image space
            global_x = x_min + int((x / window_size[0]) * (x_max - x_min))
            global_y = y_min + int((y / window_size[1]) * (y_max - y_min))
            cache_point = (global_x, global_y)
            print(f"Point selected at position: ({global_x}, {global_y})")
        else:
            cache_point = (x, y)
            print(f"Point selected at position: ({x}, {y})")
        print("Press 'Enter' to confirm point.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        zoom_start = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        if zoom_start is not None:
            zoom_end = (x, y)
            x1, y1 = zoom_start
            x2, y2 = zoom_end
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            if x_max - x_min > 10 and y_max - y_min > 10:
                zoom_region = (x_min, y_min, x_max, y_max)
                zoom_mode = True
            zoom_start = None

# === Modified collect_clicks to use Delete/Backspace key for point removal and zooming ===
def collect_clicks(window_name, image, num_points=4, confirm_each_point=True):
    global points, cache_point, zoom_start, zoom_region, zoom_mode
    points = []  # Reset points list
    cache_point = None
    zoom_start = None
    zoom_region = None
    zoom_mode = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *window_size)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        img_to_show = image.copy()

        if zoom_mode and zoom_region:
            x_min, y_min, x_max, y_max = zoom_region
            zoomed = image[y_min:y_max, x_min:x_max]
            zoomed_resized = cv2.resize(zoomed, window_size, interpolation=cv2.INTER_NEAREST)

            # Show existing points (mapped into zoom view)
            for (px, py) in points:
                if x_min <= px < x_max and y_min <= py < y_max:
                    zx = int((px - x_min) / (x_max - x_min) * window_size[0])
                    zy = int((py - y_min) / (y_max - y_min) * window_size[1])
                    cv2.circle(zoomed_resized, (zx, zy), 5, (0, 0, 255), -1)

            # Show cached point if exists
            if cache_point:
                if x_min <= cache_point[0] < x_max and y_min <= cache_point[1] < y_max:
                    zx = int((cache_point[0] - x_min) / (x_max - x_min) * window_size[0])
                    zy = int((cache_point[1] - y_min) / (y_max - y_min) * window_size[1])
                    cv2.circle(zoomed_resized, (zx, zy), 5, (0, 255, 0), -1)

            cv2.imshow(window_name, zoomed_resized)
        else:
            # Show existing points
            for (px, py) in points:
                cv2.circle(img_to_show, (px, py), 5, (0, 0, 255), -1)

            # Show cached point if exists
            if cache_point:
                cv2.circle(img_to_show, cache_point, 5, (0, 255, 0), -1)

            cv2.imshow(window_name, img_to_show)

        key = cv2.waitKey(1)

        # Confirm the cached point by pressing Enter
        if key == 13 and cache_point:  # 'Enter' key to confirm point
            points.append(cache_point)
            print(f"Point confirmed: {cache_point}")
            cache_point = None  # Reset the cached point

            # If we have enough points and it's not hole selection, break
            if len(points) >= num_points and num_points != 999:
                break

        # Point removal by pressing the 'Delete' or 'Backspace' key
        if key == 40 or key == 127:  # 40 (Delete) or 127 (Backspace)
            if points:
                removed_point = points.pop()  # Remove the last point
                print(f"Point removed: {removed_point}")
            else:
                print("No points to remove.")
            cache_point = None  # Reset the cached point after removal

        # Press 'Tab' to reset zoom
        if key == 9:  # 'Tab' key
            zoom_mode = False
            zoom_region = None

        # ESC key handling
        if key == 27:  # ESC key
            # For hole selection (num_points=999), return collected points
            if num_points == 999 and points:
                print(f"Finished hole selection with {len(points)} points")
                break
            # For other selections, only exit if we have the required number of points
            elif len(points) >= num_points:
                break
            # Otherwise, clear the current selection
            else:
                points = []
                cache_point = None
                print("Selection cleared. Please start again or press ESC with required points to finish.")

    cv2.destroyWindow(window_name)
    return points

def regenerate_visualizations(data, current_results_path):
    """Regenerate all visualizations from loaded data.
    Note: Expects data to be in original coordinate space and handles scaling for visualization.
    """
    try:
        print("\n[COORDS] === Starting Visualization Regeneration ===")
        
        # Load the image
        full_image = cv2.imread(str(data['image_path']))
        if full_image is None:
            print(f"Error: Could not load image at {data['image_path']}")
            return False
            
        # Get data (all in original space)
        corner_data = data['corner_data']
        measurements_data = data['measurements_data']
        
        # Create timestamp for this regeneration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the scale factor and pixel size
        scale_factor = corner_data.get('scale_factor', 0.1)
        pixel_size_mm = corner_data.get('pixel_size_mm', 25.4 / 9600)  # Original pixel size
        
        print(f"[COORDS] Original image dimensions: {full_image.shape[1]}x{full_image.shape[0]}")
        
        # Create display version of original image
        display_image = cv2.resize(full_image, (0, 0), fx=scale_factor, fy=scale_factor)
        print(f"[COORDS] Display image dimensions: {display_image.shape[1]}x{display_image.shape[0]}")
        
        # Get original coordinates and ensure they're ordered correctly
        original_corners = corner_data['corners']
        original_corners = order_corners(original_corners)
        print(f"[COORDS] Original corner coordinates: {original_corners}")
        
        # Convert to display space for visualization
        display_corners = [(int(x * scale_factor), int(y * scale_factor)) for x, y in original_corners]
        print(f"[COORDS] Display corner coordinates: {display_corners}")
        
        # Regenerate corner selection visualization
        print("\n[COORDS] === Regenerating Corner Visualization ===")
        visualize_corners(display_image, display_corners, data['image_path'], timestamp)
        
        # Compute perspective transform using original coordinates
        print("\n[COORDS] === Computing Perspective Transform ===")
        width = int(np.linalg.norm(np.array(original_corners[1]) - np.array(original_corners[0])))
        height = int(np.linalg.norm(np.array(original_corners[2]) - np.array(original_corners[0])))
        print(f"[COORDS] Target warped dimensions (original space): {width}x{height}")
        
        # Calculate transform in original space
        src_pts = np.array(original_corners, dtype="float32")
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply transform in original space
        warped = cv2.warpPerspective(full_image, M, (width, height))
        warped_h, warped_w = warped.shape[:2]
        print(f"[COORDS] Warped dimensions (original space): {warped_w}x{warped_h}")
        
        # Create display version of warped image
        display_warped = cv2.resize(warped, (0, 0), fx=scale_factor, fy=scale_factor)
        display_warped_h, display_warped_w = display_warped.shape[:2]
        print(f"[COORDS] Display warped dimensions: {display_warped_w}x{display_warped_h}")
        
        # Get edge and hole points from measurements data (these are in original space)
        original_edge_points = measurements_data['edge_points']
        original_hole_points = measurements_data['hole_points']
        
        print(f"[COORDS] Original measurement points:")
        print(f"[COORDS] - Edge points: {original_edge_points}")
        print(f"[COORDS] - Hole points: {original_hole_points}")
        
        # Convert points to display space for visualization
        display_edge_points = [(int(x * scale_factor), int(y * scale_factor)) for x, y in original_edge_points]
        display_hole_points = [(int(x * scale_factor), int(y * scale_factor)) for x, y in original_hole_points]
        
        print(f"[COORDS] Display measurement points:")
        print(f"[COORDS] - Edge points: {display_edge_points}")
        print(f"[COORDS] - Hole points: {display_hole_points}")
        
        # Create side-by-side visualization
        plt.figure(figsize=(20, 10))
        
        # Original image with corners
        plt.subplot(121)
        img_with_corners = display_image.copy()
        img_with_corners = draw_edges_with_lengths(img_with_corners, display_corners, 
                                                 color=(0, 255, 0), thickness=4)
        
        for i, (x, y) in enumerate(display_corners, 1):
            cv2.circle(img_with_corners, (int(x), int(y)), 20, (0, 0, 255), -1)
            cv2.putText(img_with_corners, f"C{i}", (int(x+30), int(y+30)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        
        plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        plt.title("Original Image with Corners", fontsize=16)
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(display_warped, cv2.COLOR_BGR2RGB))
        plt.title("Perspective Corrected", fontsize=16)
        plt.axis('off')
        
        plt.tight_layout()
        save_plot(plt.gcf(), "02_perspective_correction.jpg")
        
        # Create visualization data structure with display coordinates
        viz_data = {
            "hole_points": display_hole_points,
            "edge_points": display_edge_points,
            "corners": display_corners,
            "image_path": str(data['image_path']),
            "warped_dimensions": {"width": display_warped_w, "height": display_warped_h},
            "timestamp": timestamp,
            "scale_factor": scale_factor,
            "pixel_size_mm": pixel_size_mm,
            "is_original_space": False
        }
        
        # Generate final visualization using display coordinates
        visualize_final_results(display_warped, viz_data)
        
        print(f"\n[COORDS] Results have been saved to: {current_results_path}")
        return True
        
    except Exception as e:
        print(f"Error during visualization regeneration: {e}")
        import traceback
        traceback.print_exc()
        return False

# === Main Function to Start Point Selection and Zooming ===
if __name__ == "__main__":
    try:
        # Get user selection
        selection = get_user_selection()
        if not selection['choice']:
            print("No selection made. Exiting.")
            exit()
        
        # Set up results path
        if selection['choice'] == 'new':
            print("\n=== Starting New Image Processing ===")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_results_path = selection['results_path'] / timestamp
            current_results_path.mkdir(parents=True, exist_ok=True)
            
            # Process new image
            image_path = selection['image_path']
            if not image_path.exists():
                print(f"Error: Image file not found at {image_path}")
                exit()
                
            # Load original image (full resolution)
            full_image = cv2.imread(str(image_path))
            if full_image is None:
                print(f"Error: Could not load image at {image_path}")
                exit()
                
            # Initialize global variables
            dpi = 9600
            scale_factor = 0.1  # Only used for display
            pixel_size_mm = 25.4 / dpi  # Original pixel size
            
            full_h, full_w = full_image.shape[:2]
            print(f"\n[COORDS] Original image dimensions: {full_w}x{full_h}")
            
            # === PHASE 1: Input - Get coordinates in original space ===
            
            # Create display image for selection
            display_image = cv2.resize(full_image, (0, 0), fx=scale_factor, fy=scale_factor)
            display_h, display_w = display_image.shape[:2]
            print(f"[COORDS] Display dimensions: {display_w}x{display_h}")
            
            # Select corners in display space and convert to original space
            print("\n[COORDS] === Corner Selection ===")
            display_corners = collect_clicks("Select corners", display_image, num_points=4)
            if len(display_corners) != 4:
                print("Error: Need exactly 4 corner points")
                exit()
            
            # Convert to original space immediately
            original_corners = [(int(x / scale_factor), int(y / scale_factor)) for x, y in display_corners]
            print(f"[COORDS] Corner coordinates in original space: {original_corners}")
            
            # Order corners in original space
            original_corners = order_corners(original_corners)
            print(f"[COORDS] Ordered corners in original space: {original_corners}")
            
            # Save corner data in original space
            corner_data = {
                "corners": original_corners,
                "timestamp": timestamp,
                "image_path": str(image_path),
                "image_dimensions": {"width": full_w, "height": full_h},
                "scale_factor": scale_factor
            }
            save_points_data(corner_data, "corner_data.json")
            
            # === PHASE 2: Processing - All calculations in original space ===
            
            # Compute perspective transform in original space
            print("\n[COORDS] === Perspective Transform ===")
            width = int(np.linalg.norm(np.array(original_corners[1]) - np.array(original_corners[0])))
            height = int(np.linalg.norm(np.array(original_corners[2]) - np.array(original_corners[0])))
            print(f"[COORDS] Target warped dimensions (original space): {width}x{height}")
            
            # Transform in original space
            src_pts = np.array(original_corners, dtype="float32")
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [0, height - 1],
                [width - 1, height - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(full_image, M, (width, height))
            warped_h, warped_w = warped.shape[:2]
            print(f"[COORDS] Warped dimensions (original space): {warped_w}x{warped_h}")
            
            # Create display version of warped image
            display_warped = cv2.resize(warped, (0, 0), fx=scale_factor, fy=scale_factor)
            
            # Select edge points in display space and convert to original space
            print("\n[COORDS] === Edge Point Selection ===")
            display_edge_points = collect_clicks("Select second bottom edge", display_warped, num_points=2)
            if len(display_edge_points) != 2:
                print("Error: Need exactly 2 points for second bottom edge")
                exit()
            
            # Convert to original space immediately
            original_edge_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in display_edge_points]
            print(f"[COORDS] Edge points in original space: {original_edge_points}")
            
            # Select hole points in display space and convert to original space
            print("\n[COORDS] === Hole Point Selection ===")
            display_hole_points = collect_clicks("Select holes", display_warped, num_points=999)
            
            # Convert to original space immediately
            original_hole_points = []
            if display_hole_points:
                original_hole_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in display_hole_points]
                print(f"[COORDS] Hole points in original space: {original_hole_points}")
            
            # Save all measurements in original space
            measurements_data = {
                "hole_points": original_hole_points,
                "edge_points": original_edge_points,
                "corners": original_corners,
                "image_path": str(image_path),
                "warped_dimensions": {"width": warped_w, "height": warped_h},
                "timestamp": timestamp,
                "scale_factor": scale_factor,
                "pixel_size_mm": pixel_size_mm
            }
            
            # Save before any visualization
            save_points_data(measurements_data, "measurements_data.json")
            print(f"[COORDS] Saved measurement data in original space")
            
            # === PHASE 3: Visualization - Scale coordinates for display ===
            
            # Scale coordinates for visualization
            display_corners_ordered = [(int(x * scale_factor), int(y * scale_factor)) for x, y in original_corners]
            
            # Visualize corner selection
            visualize_corners(display_image, display_corners_ordered, image_path, timestamp)
            
            # Create side-by-side visualization
            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            img_with_corners = display_image.copy()
            img_with_corners = draw_edges_with_lengths(img_with_corners, display_corners_ordered, 
                                                     color=(0, 255, 0), thickness=4)
            
            for i, (x, y) in enumerate(display_corners_ordered, 1):
                cv2.circle(img_with_corners, (int(x), int(y)), 20, (0, 0, 255), -1)
                cv2.putText(img_with_corners, f"C{i}", (int(x+30), int(y+30)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
            plt.title("Original Image with Corners", fontsize=16)
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(cv2.cvtColor(display_warped, cv2.COLOR_BGR2RGB))
            plt.title("Perspective Corrected", fontsize=16)
            plt.axis('off')
            
            plt.tight_layout()
            save_plot(plt.gcf(), "02_perspective_correction.jpg")
            
            # Create visualization data structure with display coordinates
            viz_data = {
                "hole_points": display_hole_points,
                "edge_points": display_edge_points,
                "corners": display_corners_ordered,
                "image_path": str(image_path),
                "warped_dimensions": {"width": int(warped_w * scale_factor), 
                                    "height": int(warped_h * scale_factor)},
                "timestamp": timestamp,
                "scale_factor": scale_factor,
                "pixel_size_mm": pixel_size_mm,
                "is_original_space": False
            }
            
            # Generate final visualization
            visualize_final_results(display_warped, viz_data)
            
            print(f"\nResults have been saved to: {current_results_path}")
            
        elif selection['choice'] == 'replot':
            print("\n=== Starting Replot (Visualization Only) ===")
            current_results_path = selection['results_path']
            
            # Load existing data
            data = load_existing_data(current_results_path)
            if not data:
                print("Error: Could not load existing data")
                exit()
            
            print("\n[COORDS] === Loaded Data ===")
            print(f"[COORDS] Corner data:")
            print(f"[COORDS] - Corners: {data['corner_data']['corners']}")
            print(f"[COORDS] - Scale factor: {data['corner_data'].get('scale_factor', 0.1)}")
            print(f"\n[COORDS] Measurement data:")
            print(f"[COORDS] - Edge points: {data['measurements_data']['edge_points']}")
            print(f"[COORDS] - Hole points: {data['measurements_data']['hole_points']}")
            print(f"[COORDS] - Warped dimensions: {data['measurements_data'].get('warped_dimensions', 'Not found')}")
            
            # Regenerate visualizations without changes
            print("\n[COORDS] === Regenerating Visualizations ===")
            if not regenerate_visualizations(data, current_results_path):
                print("Failed to regenerate visualizations")
                exit()
            
        else:  # 'load' option
            print("\n=== Starting Load (New Measurements) ===")
            current_results_path = selection['results_path']
            
            # Load existing corner data
            data = load_existing_data(current_results_path)
            if not data:
                print("Error: Could not load existing data")
                exit()
            
            print("\n[COORDS] === Loading Corner Data ===")
            corner_data = data['corner_data']
            original_corners = corner_data['corners']
            scale_factor = corner_data.get('scale_factor', 0.1)
            pixel_size_mm = corner_data.get('pixel_size_mm', 25.4 / 9600)
            
            print(f"[COORDS] Original corners: {original_corners}")
            
            # Load and process original image
            full_image = cv2.imread(str(corner_data['image_path']))
            if full_image is None:
                print(f"Error: Could not load image at {corner_data['image_path']}")
                exit()
            
            # Compute perspective transform using original corners
            print("\n[COORDS] === Computing Perspective Transform ===")
            width = int(np.linalg.norm(np.array(original_corners[1]) - np.array(original_corners[0])))
            height = int(np.linalg.norm(np.array(original_corners[2]) - np.array(original_corners[0])))
            print(f"[COORDS] Target warped dimensions (original space): {width}x{height}")
            
            # Transform in original space
            src_pts = np.array(original_corners, dtype="float32")
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [0, height - 1],
                [width - 1, height - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(full_image, M, (width, height))
            warped_h, warped_w = warped.shape[:2]
            print(f"[COORDS] Warped dimensions (original space): {warped_w}x{warped_h}")
            
            # Create display version for selection
            display_warped = cv2.resize(warped, (0, 0), fx=scale_factor, fy=scale_factor)
            
            # Select new edge points in display space
            print("\n[COORDS] === Edge Point Selection ===")
            display_edge_points = collect_clicks("Select second bottom edge", display_warped, num_points=2)
            if len(display_edge_points) != 2:
                print("Error: Need exactly 2 points for second bottom edge")
                exit()
            
            # Convert to original space immediately
            original_edge_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in display_edge_points]
            print(f"[COORDS] Edge points in original space: {original_edge_points}")
            
            # Select new hole points in display space
            print("\n[COORDS] === Hole Point Selection ===")
            display_hole_points = collect_clicks("Select holes", display_warped, num_points=999)
            
            # Convert to original space immediately
            original_hole_points = []
            if display_hole_points:
                original_hole_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in display_hole_points]
                print(f"[COORDS] Hole points in original space: {original_hole_points}")
            
            # Create new measurements data with original coordinates
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            measurements_data = {
                "hole_points": original_hole_points,
                "edge_points": original_edge_points,
                "corners": original_corners,  # Keep original corners
                "image_path": str(corner_data['image_path']),
                "warped_dimensions": {"width": warped_w, "height": warped_h},
                "timestamp": timestamp,
                "scale_factor": scale_factor,
                "pixel_size_mm": pixel_size_mm
            }
            
            # Save new measurements
            print("\n[COORDS] === Saving New Measurements ===")
            save_points_data(measurements_data, "measurements_data.json")
            print(f"[COORDS] Saved new measurement data")
            
            # Create visualization data with display coordinates
            viz_data = {
                "hole_points": display_hole_points,
                "edge_points": display_edge_points,
                "corners": [(int(x * scale_factor), int(y * scale_factor)) for x, y in original_corners],
                "image_path": str(corner_data['image_path']),
                "warped_dimensions": {"width": int(warped_w * scale_factor), 
                                    "height": int(warped_h * scale_factor)},
                "timestamp": timestamp,
                "scale_factor": scale_factor,
                "pixel_size_mm": pixel_size_mm,
                "is_original_space": False
            }
            
            # Generate visualizations
            print("\n[COORDS] === Generating Visualizations ===")
            display_image = cv2.resize(full_image, (0, 0), fx=scale_factor, fy=scale_factor)
            display_corners = [(int(x * scale_factor), int(y * scale_factor)) for x, y in original_corners]
            
            # Visualize corner selection
            visualize_corners(display_image, display_corners, corner_data['image_path'], timestamp)
            
            # Create side-by-side visualization
            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            img_with_corners = display_image.copy()
            img_with_corners = draw_edges_with_lengths(img_with_corners, display_corners, 
                                                     color=(0, 255, 0), thickness=4)
            
            for i, (x, y) in enumerate(display_corners, 1):
                cv2.circle(img_with_corners, (int(x), int(y)), 20, (0, 0, 255), -1)
                cv2.putText(img_with_corners, f"C{i}", (int(x+30), int(y+30)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
            plt.title("Original Image with Corners", fontsize=16)
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(cv2.cvtColor(display_warped, cv2.COLOR_BGR2RGB))
            plt.title("Perspective Corrected", fontsize=16)
            plt.axis('off')
            
            plt.tight_layout()
            save_plot(plt.gcf(), "02_perspective_correction.jpg")
            
            # Generate final visualization
            visualize_final_results(display_warped, viz_data)
            
            print(f"\n[COORDS] Results have been saved to: {current_results_path}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
