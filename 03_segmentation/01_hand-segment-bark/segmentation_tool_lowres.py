import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np

# Constants
CLASSES = {
    0: {'name': 'Background', 'color': (0, 0, 0)},
    1: {'name': 'slepice / odrezane veje', 'color': (0, 255, 255)},   # Cyan
    2: {'name': 'mehanske poškodbe / odluščeno', 'color': (255, 128, 0)}, # Orange
}
NUM_CLASSES = len(CLASSES)
DEFAULT_BRUSH_SIZE = 1 # In grid cells
DEFAULT_OPACITY = 0.2
TARGET_IMAGE_WIDTH = 1000
GRID_WIDTH = 20

class SegmentationAppLowRes:
    def __init__(self, root):
        self.root = root
        self.root.title("Low-Res Image Segmentation Tool")
        self.root.geometry("1200x800")

        # State Variables
        self.image_dir = ""
        self.mask_dir = ""
        self.output_image_dir = ""
        
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        
        self.processed_image = None # PIL Image (RGB) - 1000px wide, cropped
        self.mask_image = None      # PIL Image (L) - 20px wide
        
        self.display_image = None   # PIL Image (RGB) - blended for display
        self.tk_image = None        # ImageTk for Canvas

        self.current_class = 1
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.opacity = DEFAULT_OPACITY
        
        self.scale = 1.0 # Canvas scale relative to processed image
        self.offset_x = 0
        self.offset_y = 0
        
        self.last_grid_x = None
        self.last_grid_y = None
        
        self.cursor_id = None
        self.cell_size_px = 0 # Size of one grid cell in processed image pixels

        self._setup_ui()
        self._bind_events()

    def _setup_ui(self):
        # Top Toolbar
        self.toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(self.toolbar, text="1. Input Images", command=self.select_image_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="2. Output Images", command=self.select_output_image_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="3. Output Masks", command=self.select_mask_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="Save (Ctrl+S)", command=self.save_data).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Frame(self.toolbar, width=20).pack(side=tk.LEFT) # Spacer
        
        tk.Button(self.toolbar, text="< Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_image_name = tk.Label(self.toolbar, text="No Image Loaded")
        self.lbl_image_name.pack(side=tk.LEFT, padx=10)
        tk.Button(self.toolbar, text="Next >", command=self.next_image).pack(side=tk.LEFT, padx=5, pady=5)

        # Side Toolbar (Controls)
        self.controls = tk.Frame(self.root, bd=1, relief=tk.RAISED, width=200)
        self.controls.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.controls, text="Classes", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.class_var = tk.IntVar(value=self.current_class)
        for cls_id, cls_info in CLASSES.items():
            frame = tk.Frame(self.controls)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            color_hex = '#%02x%02x%02x' % cls_info['color']
            lbl_color = tk.Label(frame, bg=color_hex, width=4)
            lbl_color.pack(side=tk.LEFT, padx=5)
            
            rb = tk.Radiobutton(frame, text=f"{cls_id}: {cls_info['name']}", variable=self.class_var, value=cls_id, command=self.change_class)
            rb.pack(side=tk.LEFT)

        tk.Label(self.controls, text="Brush Size (Grid Cells)", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        self.scale_brush = tk.Scale(self.controls, from_=1, to=10, orient=tk.HORIZONTAL, command=self.change_brush_size)
        self.scale_brush.set(self.brush_size)
        self.scale_brush.pack(fill=tk.X, padx=10)

        # Main Canvas
        self.canvas_frame = tk.Frame(self.root, bg="gray")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _bind_events(self):
        self.root.bind("<Control-s>", lambda e: self.save_data())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("[", lambda e: self.adjust_brush_size(-1))
        self.root.bind("]", lambda e: self.adjust_brush_size(1))
        self.root.bind("]", lambda e: self.adjust_brush_size(1))
        
        for cls_id in CLASSES:
            self.root.bind(str(cls_id), lambda e, idx=cls_id: self.set_class(idx))

        self.canvas.bind("<Button-1>", self.start_paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_paint)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Motion>", self.update_cursor)
        self.canvas.bind("<Leave>", self.hide_cursor)
        self.canvas.bind("<Enter>", self.show_cursor)

    def select_image_dir(self):
        path = filedialog.askdirectory(title="Select Input Images Directory")
        if path:
            self.image_dir = path
            self.image_files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            if not self.image_files:
                messagebox.showwarning("No Images", "No image files found in the selected directory.")
                return
            self.current_image_index = 0
            self.load_image()

    def select_output_image_dir(self):
        path = filedialog.askdirectory(title="Select Output Images Directory")
        if path:
            self.output_image_dir = path

    def select_mask_dir(self):
        path = filedialog.askdirectory(title="Select Output Masks Directory")
        if path:
            self.mask_dir = path

    def load_image(self):
        if not self.image_files or self.current_image_index < 0:
            return

        filename = self.image_files[self.current_image_index]
        self.current_image_path = os.path.join(self.image_dir, filename)
        self.lbl_image_name.config(text=f"{filename} ({self.current_image_index + 1}/{len(self.image_files)})")

        try:
            # 1. Load Original
            original = Image.open(self.current_image_path).convert("RGB")
            
            # 2. Resize to 1000px width
            w, h = original.size
            scale_factor = TARGET_IMAGE_WIDTH / w
            new_w = TARGET_IMAGE_WIDTH
            new_h = int(h * scale_factor)
            resized = original.resize((new_w, new_h), Image.LANCZOS)
            
            # 3. Calculate Grid and Crop
            # We want mask width = 20
            self.cell_size_px = new_w / GRID_WIDTH # 1000 / 20 = 50.0
            
            grid_h = int(new_h / self.cell_size_px)
            target_h = int(grid_h * self.cell_size_px)
            
            # Center crop
            top = (new_h - target_h) // 2
            bottom = top + target_h
            
            self.processed_image = resized.crop((0, top, new_w, bottom))
            
            # 4. Initialize Mask (20 x grid_h)
            mask_size = (GRID_WIDTH, grid_h)
            
            # Try to load existing mask
            mask_path = self._get_mask_path(filename)
            if mask_path and os.path.exists(mask_path):
                loaded_mask = Image.open(mask_path).convert("L")
                if loaded_mask.size == mask_size:
                    self.mask_image = loaded_mask
                else:
                    # Size mismatch - create new
                    self.mask_image = Image.new("L", mask_size, 0)
            else:
                self.mask_image = Image.new("L", mask_size, 0)

            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _get_mask_path(self, image_filename):
        if not self.mask_dir:
            return None
        basename = os.path.splitext(image_filename)[0]
        return os.path.join(self.mask_dir, basename + ".png")
        
    def _get_output_image_path(self, image_filename):
        if not self.output_image_dir:
            return None
        # Save as JPG for images usually, or PNG
        basename = os.path.splitext(image_filename)[0]
        return os.path.join(self.output_image_dir, basename + ".jpg")

    def update_display(self):
        if self.processed_image is None:
            return

        # Upscale mask to match processed image size for overlay
        mask_array = np.array(self.mask_image)
        
        # Create colored overlay
        overlay_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
        for cls_id, cls_info in CLASSES.items():
            if cls_id == 0: continue 
            overlay_array[mask_array == cls_id] = cls_info['color']

        overlay_small = Image.fromarray(overlay_array, mode='RGB')
        overlay_full = overlay_small.resize(self.processed_image.size, Image.NEAREST)
        
        # Create Alpha
        mask_bool = mask_array > 0
        mask_alpha_small = Image.fromarray((mask_bool * 255 * self.opacity).astype(np.uint8), mode='L')
        mask_alpha_full = mask_alpha_small.resize(self.processed_image.size, Image.NEAREST)
        
        # Composite
        display = self.processed_image.convert("RGBA")
        overlay_rgba = overlay_full.convert("RGBA")
        overlay_rgba.putalpha(mask_alpha_full)
        
        display.alpha_composite(overlay_rgba)
        self.display_image = display.convert("RGB")
        
        self.draw_image_on_canvas()

    def draw_image_on_canvas(self):
        if self.display_image is None:
            return
            
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = self.display_image.size
        
        if cw == 0 or ch == 0: return

        scale_w = cw / iw
        scale_h = ch / ih
        self.scale = min(scale_w, scale_h)
        
        new_w = int(iw * self.scale)
        new_h = int(ih * self.scale)
        
        resized = self.display_image.resize((new_w, new_h), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        self.offset_x = (cw - new_w) // 2
        self.offset_y = (ch - new_h) // 2
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
        
        # Draw Grid Lines (Optional, but helpful for pixel art)
        # self.draw_grid() 

    def on_resize(self, event):
        if self.processed_image:
            self.draw_image_on_canvas()

    def start_paint(self, event):
        self.paint(event)

    def stop_paint(self, event):
        self.last_grid_x = None
        self.last_grid_y = None

    def paint(self, event):
        if not self.processed_image:
            return

        cx, cy = event.x, event.y
        
        # Canvas -> Processed Image
        px = (cx - self.offset_x) / self.scale
        py = (cy - self.offset_y) / self.scale
        
        # Processed Image -> Grid
        gx = int(px / self.cell_size_px)
        gy = int(py / self.cell_size_px)
        
        # Bounds check
        if gx < 0 or gx >= self.mask_image.width or gy < 0 or gy >= self.mask_image.height:
            return

        # Paint on mask
        draw = ImageDraw.Draw(self.mask_image)
        
        # If we have line drawing logic, we need it in grid coords
        if self.last_grid_x is not None:
             draw.line([self.last_grid_x, self.last_grid_y, gx, gy], fill=self.current_class, width=self.brush_size)
        
        # Point/Circle
        r = self.brush_size // 2
        # For pixel art, maybe just fill the square? 
        # If brush size is 1, just point.
        # If brush size > 1, maybe square or circle? Let's do square for pixel grid.
        
        x1 = gx - r
        y1 = gy - r
        x2 = gx + r + (1 if self.brush_size % 2 != 0 else 0) # Adjust for even/odd
        y2 = gy + r + (1 if self.brush_size % 2 != 0 else 0)
        
        # Actually standard PIL rectangle is [x0, y0, x1, y1] inclusive? No, second point is inclusive.
        # Let's just use rectangle
        draw.rectangle([x1, y1, x2-1, y2-1], fill=self.current_class)
        
        self.last_grid_x = gx
        self.last_grid_y = gy
        
        self.update_display()
        self.update_cursor(event)

    def update_cursor(self, event):
        if not self.processed_image:
            return
            
        # Snap cursor to grid
        cx, cy = event.x, event.y
        px = (cx - self.offset_x) / self.scale
        py = (cy - self.offset_y) / self.scale
        
        gx = int(px / self.cell_size_px)
        gy = int(py / self.cell_size_px)
        
        # Back to canvas coords
        # Top-left of cell
        cell_x = gx * self.cell_size_px * self.scale + self.offset_x
        cell_y = gy * self.cell_size_px * self.scale + self.offset_y
        
        cell_w = self.cell_size_px * self.scale
        
        # Brush size
        r = self.brush_size // 2
        
        # Calculate bounds of the brush on grid
        x1_g = gx - r
        y1_g = gy - r
        x2_g = gx + r + (1 if self.brush_size % 2 != 0 else 0)
        y2_g = gy + r + (1 if self.brush_size % 2 != 0 else 0)
        
        # Convert to canvas
        x1_c = x1_g * self.cell_size_px * self.scale + self.offset_x
        y1_c = y1_g * self.cell_size_px * self.scale + self.offset_y
        x2_c = x2_g * self.cell_size_px * self.scale + self.offset_x
        y2_c = y2_g * self.cell_size_px * self.scale + self.offset_y
        
        if self.cursor_id is None:
            self.cursor_id = self.canvas.create_rectangle(x1_c, y1_c, x2_c, y2_c, outline="white", width=2, tag="cursor")
        else:
            self.canvas.coords(self.cursor_id, x1_c, y1_c, x2_c, y2_c)
            self.canvas.tag_raise(self.cursor_id)

    def hide_cursor(self, event):
        if self.cursor_id:
            self.canvas.delete(self.cursor_id)
            self.cursor_id = None

    def show_cursor(self, event):
        self.update_cursor(event)

    def change_class(self):
        self.current_class = self.class_var.get()

    def set_class(self, idx):
        if idx in CLASSES:
            self.current_class = idx
            self.class_var.set(idx)

    def change_brush_size(self, val):
        self.brush_size = int(val)

    def adjust_brush_size(self, delta):
        new_size = self.brush_size + delta
        new_size = max(1, min(10, new_size))
        self.brush_size = new_size
        self.scale_brush.set(new_size)

    def prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()

    def save_data(self):
        if not self.mask_dir or not self.output_image_dir:
            messagebox.showwarning("Warning", "Please select both Output Images and Output Masks directories.")
            return
            
        if self.mask_image and self.processed_image and self.current_image_path:
            filename = os.path.basename(self.current_image_path)
            
            # Save Mask
            mask_path = self._get_mask_path(filename)
            self.mask_image.save(mask_path)
            
            # Save Processed Image
            img_path = self._get_output_image_path(filename)
            self.processed_image.save(img_path, quality=95)
            
            print(f"Saved mask to {mask_path}")
            print(f"Saved image to {img_path}")
            self.root.title(f"Low-Res Tool - Saved {filename}")

    def show_help(self):
        help_text = """
        Low-Res Segmentation Tool Guide
        
        1. Setup:
           - Select Input Images folder.
           - Select Output Images folder (for resized/cropped images).
           - Select Output Masks folder.
        
        2. Logic:
           - Images are resized to 1000px width.
           - Images are cropped to fit the 20px wide grid.
           - Masks are 20 pixels wide (very low res).
        
        3. Painting:
           - Paint blocks on the grid.
           - Brush size is in grid cells.
        """
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationAppLowRes(root)
    root.mainloop()
