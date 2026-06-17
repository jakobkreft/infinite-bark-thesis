import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np
import cv2

class SegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Tool")
        
        # Data
        self.image_folder = ""
        self.mask_folder = ""
        self.image_list = []
        self.current_index = 0
        self.points = [] # Stores (x, y) tuples in ORIGINAL image coordinates
        self.current_image = None # PIL Image (original)
        self.tk_image = None # ImageTk (resized)
        self.original_image_cv = None # OpenCV image
        self.scale_factor = 1.0
        
        # UI Setup
        self.setup_ui()
        
    def setup_ui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Button(control_frame, text="Select Image Folder", command=self.select_image_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Select Mask Folder", command=self.select_mask_folder).pack(side=tk.LEFT, padx=5)
        
        self.lbl_status = tk.Label(control_frame, text="No folder selected")
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        
        # Canvas
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_resize)
        
        # Bottom Control Panel
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        tk.Button(bottom_frame, text="<< Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Next >>", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        self.lbl_index = tk.Label(bottom_frame, text="0 / 0")
        self.lbl_index.pack(side=tk.LEFT, padx=10)
        
        tk.Button(bottom_frame, text="Redo Points", command=self.reset_points).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Save Mask", command=self.save_mask, bg="lightblue").pack(side=tk.RIGHT, padx=5)

    def select_image_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.image_folder = folder
            self.load_images()
            
    def select_mask_folder(self):
        folder = filedialog.askdirectory(title="Select Mask Folder")
        if folder:
            self.mask_folder = folder
            
    def load_images(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.image_list = [f for f in os.listdir(self.image_folder) if f.lower().endswith(valid_extensions)]
        self.image_list.sort()
        self.current_index = 0
        if self.image_list:
            self.load_current_image()
        else:
            self.lbl_status.config(text="No images found in folder")
            
    def load_current_image(self):
        if not self.image_list:
            return
            
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        self.original_image_cv = cv2.imread(image_path)
        
        # Load for display
        self.current_image = Image.open(image_path)
        self.display_image()
        self.update_status()
        self.reset_points()

    def display_image(self):
        if self.current_image is None:
            return
            
        # Calculate scale to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Initial load might happen before canvas is sized
            canvas_width = 800
            canvas_height = 600
            
        img_w, img_h = self.current_image.size
        scale_w = canvas_width / img_w
        scale_h = canvas_height / img_h
        self.scale_factor = min(scale_w, scale_h, 1.0) # Don't upscale, only downscale? Or fit? User said "fit".
        self.scale_factor = min(scale_w, scale_h) # Fit completely
        
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)
        
        resized_image = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        
        self.canvas.delete("all")
        # Center the image
        x_offset = (canvas_width - new_w) // 2
        y_offset = (canvas_height - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image)
        self.draw_overlays(x_offset, y_offset)

    def on_resize(self, event):
        if self.current_image:
            self.display_image()

    def on_canvas_click(self, event):
        if len(self.points) >= 4:
            return
            
        # Convert canvas coords to image coords
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_w, img_h = self.current_image.size
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)
        x_offset = (canvas_width - new_w) // 2
        y_offset = (canvas_height - new_h) // 2
        
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        img_x = (cx - x_offset) / self.scale_factor
        img_y = (cy - y_offset) / self.scale_factor
        
        self.points.append((img_x, img_y))
        self.display_image() # Redraw with new points

    def draw_overlays(self, x_off, y_off):
        # Draw points
        r = 3
        for i, (px, py) in enumerate(self.points):
            cx = px * self.scale_factor + x_off
            cy = py * self.scale_factor + y_off
            color = "red" if i < 2 else "orange"
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color, outline=color)
            
        # Draw infinite lines
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        def draw_extended_line(p1, p2, color):
            # Calculate line equation y = mx + c or x = c
            x1, y1 = p1
            x2, y2 = p2
            
            # Map to canvas coords for drawing
            cx1 = x1 * self.scale_factor + x_off
            cy1 = y1 * self.scale_factor + y_off
            cx2 = x2 * self.scale_factor + x_off
            cy2 = y2 * self.scale_factor + y_off
            
            if abs(cx2 - cx1) < 1e-5: # Vertical
                self.canvas.create_line(cx1, 0, cx1, h, fill=color, width=2)
            else:
                m = (cy2 - cy1) / (cx2 - cx1)
                c = cy1 - m * cx1
                # Intersections with x=0 and x=w
                y_at_0 = c
                y_at_w = m * w + c
                self.canvas.create_line(0, y_at_0, w, y_at_w, fill=color, width=2)

        if len(self.points) >= 2:
            draw_extended_line(self.points[0], self.points[1], "blue")
            
        if len(self.points) >= 4:
            draw_extended_line(self.points[2], self.points[3], "blue")

    def update_status(self):
        self.lbl_index.config(text=f"{self.current_index + 1} / {len(self.image_list)}")
        self.lbl_status.config(text=f"Processing: {self.image_list[self.current_index]}")

    def reset_points(self):
        self.points = []
        self.display_image()

    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def save_mask(self):
        if not self.mask_folder:
            messagebox.showwarning("Warning", "Please select a mask save folder first.")
            return
        
        if len(self.points) != 4:
            messagebox.showwarning("Warning", "Please select 4 points (2 top, 2 bottom) first.")
            return
            
        if self.original_image_cv is None:
            return

        h, w = self.original_image_cv.shape[:2]
        
        # Create meshgrid
        Y, X = np.indices((h, w))
        
        def get_line_sign(p1, p2, X, Y):
            x1, y1 = p1
            x2, y2 = p2
            # Cross product (P - A) x (B - A)
            # (x - x1)(y2 - y1) - (y - y1)(x2 - x1)
            return (X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)

        # Line 1: Top
        sign1 = get_line_sign(self.points[0], self.points[1], X, Y)
        
        # Line 2: Bottom
        sign2 = get_line_sign(self.points[2], self.points[3], X, Y)
        
        # Determine which side is "in between"
        # We check a point on Line 2 against Line 1
        mid2 = ((self.points[2][0] + self.points[3][0])/2, (self.points[2][1] + self.points[3][1])/2)
        ref_sign1 = get_line_sign(self.points[0], self.points[1], mid2[0], mid2[1])
        
        # Check a point on Line 1 against Line 2
        mid1 = ((self.points[0][0] + self.points[1][0])/2, (self.points[0][1] + self.points[1][1])/2)
        ref_sign2 = get_line_sign(self.points[2], self.points[3], mid1[0], mid1[1])
        
        # Logic: 
        # Pixel is valid if it has same sign w.r.t Line 1 as Line 2 has
        # AND same sign w.r.t Line 2 as Line 1 has.
        
        mask1 = (sign1 * ref_sign1) >= 0
        mask2 = (sign2 * ref_sign2) >= 0
        
        final_mask = np.logical_and(mask1, mask2).astype(np.uint8) * 255
        
        # Save
        filename = self.image_list[self.current_index]
        name, ext = os.path.splitext(filename)
        save_path = os.path.join(self.mask_folder, name + ".png")
        
        cv2.imwrite(save_path, final_mask)
        print(f"Saved mask to {save_path}")
        
        self.next_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationTool(root)
    root.mainloop()
