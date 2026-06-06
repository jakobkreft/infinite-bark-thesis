import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np

# Constants
CLASSES = {
    0: {'name': 'Background', 'color': (0, 0, 0)},
    1: {'name': 'lišaji / mah / listje', 'color': (255, 0, 0)},     # Red
    2: {'name': 'olupljena smreka', 'color': (0, 255, 0)},     # Green
    3: {'name': 'gladko lubje', 'color': (0, 0, 255)},     # Blue
    4: {'name': 'hrapavo lubje', 'color': (255, 255, 0)},   # Yellow
    5: {'name': 'zelo hrapavo / luskasto lubje', 'color': (255, 0, 255)},   # Magenta
    6: {'name': 'slepice / odrezane veje', 'color': (0, 255, 255)},   # Cyan
    7: {'name': 'mehanske poškodbe / odluščeno', 'color': (255, 128, 0)}, # Orange
}
NUM_CLASSES = len(CLASSES)
DEFAULT_BRUSH_SIZE = 20
DEFAULT_OPACITY = 0.5

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Tool")
        self.root.geometry("1200x800")

        # State Variables
        self.image_dir = ""
        self.mask_dir = ""
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        
        self.original_image = None  # PIL Image (RGB) - Full Res
        self.mask_image = None      # PIL Image (L) - Full Res
        
        # View (Downscaled) images for display performance
        self.view_image = None      # PIL Image (RGB)
        self.view_mask = None       # PIL Image (L)
        self.view_scale = 1.0       # Scale factor (view / original)
        
        self.display_image = None   # PIL Image (RGB) - blended for display
        self.tk_image = None        # ImageTk for Canvas

        self.current_class = 1
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.opacity = DEFAULT_OPACITY
        
        self.scale = 1.0 # Canvas scale relative to view image
        self.offset_x = 0
        self.offset_y = 0
        
        self.last_x = None
        self.last_y = None
        
        self.cursor_id = None

        self._setup_ui()
        self._bind_events()

    def _setup_ui(self):
        # Top Toolbar
        self.toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(self.toolbar, text="Open Images", command=self.select_image_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="Open Masks", command=self.select_mask_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="Save (Ctrl+S)", command=self.save_mask).pack(side=tk.LEFT, padx=5, pady=5)
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
            # Create a colored block for the class
            frame = tk.Frame(self.controls)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Color indicator
            color_hex = '#%02x%02x%02x' % cls_info['color']
            lbl_color = tk.Label(frame, bg=color_hex, width=4)
            lbl_color.pack(side=tk.LEFT, padx=5)
            
            # Radio button
            rb = tk.Radiobutton(frame, text=f"{cls_id}: {cls_info['name']}", variable=self.class_var, value=cls_id, command=self.change_class)
            rb.pack(side=tk.LEFT)

        tk.Label(self.controls, text="Brush Size", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        self.scale_brush = tk.Scale(self.controls, from_=1, to=2000, orient=tk.HORIZONTAL, command=self.change_brush_size)
        self.scale_brush.set(self.brush_size)
        self.scale_brush.pack(fill=tk.X, padx=10)

        # Main Canvas
        self.canvas_frame = tk.Frame(self.root, bg="gray")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _bind_events(self):
        self.root.bind("<Control-s>", lambda e: self.save_mask())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("[", lambda e: self.adjust_brush_size(-5))
        self.root.bind("]", lambda e: self.adjust_brush_size(5))
        
        for i in range(NUM_CLASSES):
            self.root.bind(str(i), lambda e, idx=i: self.set_class(idx))

        self.canvas.bind("<Button-1>", self.start_paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_paint)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Motion>", self.update_cursor)
        self.canvas.bind("<Leave>", self.hide_cursor)
        self.canvas.bind("<Enter>", self.show_cursor)

    def select_image_dir(self):
        path = filedialog.askdirectory(title="Select Images Directory")
        if path:
            self.image_dir = path
            self.image_files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            if not self.image_files:
                messagebox.showwarning("No Images", "No image files found in the selected directory.")
                return
            self.current_image_index = 0
            self.load_image()

    def select_mask_dir(self):
        path = filedialog.askdirectory(title="Select Masks Directory")
        if path:
            self.mask_dir = path

    def load_image(self):
        if not self.image_files or self.current_image_index < 0:
            return

        filename = self.image_files[self.current_image_index]
        self.current_image_path = os.path.join(self.image_dir, filename)
        self.lbl_image_name.config(text=f"{filename} ({self.current_image_index + 1}/{len(self.image_files)})")

        try:
            self.original_image = Image.open(self.current_image_path).convert("RGB")
            
            # Create downscaled view image
            # Target max dimension for view (e.g., 2000px)
            max_dim = 2000
            w, h = self.original_image.size
            if w > max_dim or h > max_dim:
                self.view_scale = min(max_dim / w, max_dim / h)
                new_w = int(w * self.view_scale)
                new_h = int(h * self.view_scale)
                self.view_image = self.original_image.resize((new_w, new_h), Image.BILINEAR)
            else:
                self.view_scale = 1.0
                self.view_image = self.original_image.copy()

            # Load existing mask if it exists, otherwise create new
            mask_path = self._get_mask_path(filename)
            if mask_path and os.path.exists(mask_path):
                self.mask_image = Image.open(mask_path).convert("L")
                # Ensure mask matches image size
                if self.mask_image.size != self.original_image.size:
                     self.mask_image = self.mask_image.resize(self.original_image.size, Image.NEAREST)
            else:
                self.mask_image = Image.new("L", self.original_image.size, 0)

            # Create downscaled mask for view
            if self.view_scale != 1.0:
                self.view_mask = self.mask_image.resize(self.view_image.size, Image.NEAREST)
            else:
                self.view_mask = self.mask_image.copy()

            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _get_mask_path(self, image_filename):
        if not self.mask_dir:
            return None
        # Assuming mask has same name as image, maybe different extension or same
        basename = os.path.splitext(image_filename)[0]
        # Try png first as it is lossless
        return os.path.join(self.mask_dir, basename + ".png")

    def update_display(self):
        if self.view_image is None:
            return

        # Create colored overlay from mask (using view_mask)
        mask_array = np.array(self.view_mask)
        
        # Create an RGB image for the mask overlay
        overlay_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
        
        for cls_id, cls_info in CLASSES.items():
            if cls_id == 0: continue 
            overlay_array[mask_array == cls_id] = cls_info['color']

        overlay_image = Image.fromarray(overlay_array, mode='RGB')
        
        mask_bool = mask_array > 0
        mask_alpha = Image.fromarray((mask_bool * 255 * self.opacity).astype(np.uint8), mode='L')
        
        # Composite
        display = self.view_image.convert("RGBA")
        overlay_rgba = overlay_image.convert("RGBA")
        overlay_rgba.putalpha(mask_alpha)
        
        display.alpha_composite(overlay_rgba)
        self.display_image = display.convert("RGB")
        
        self.draw_image_on_canvas()

    def draw_image_on_canvas(self):
        if self.display_image is None:
            return
            
        # Calculate scale to fit
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = self.display_image.size
        
        if cw == 0 or ch == 0: return # Not ready yet

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

    def on_resize(self, event):
        if self.view_image:
            self.draw_image_on_canvas()

    def start_paint(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.paint(event)

    def stop_paint(self, event):
        self.last_x = None
        self.last_y = None

    def paint(self, event):
        if not self.view_image:
            return

        cx, cy = event.x, event.y
        
        # Calculate coordinates in view image
        vx = (cx - self.offset_x) / self.scale
        vy = (cy - self.offset_y) / self.scale
        
        # Calculate coordinates in original image
        ox = vx / self.view_scale
        oy = vy / self.view_scale
        
        # Previous coordinates
        if self.last_x is None:
            self.last_x = cx
            self.last_y = cy
            
        lx = (self.last_x - self.offset_x) / self.scale
        ly = (self.last_y - self.offset_y) / self.scale
        
        lox = lx / self.view_scale
        loy = ly / self.view_scale

        # Draw on View Mask (Fast feedback)
        draw_view = ImageDraw.Draw(self.view_mask)
        # Scale brush size for view
        view_brush_size = self.brush_size * self.view_scale
        draw_view.line([lx, ly, vx, vy], fill=self.current_class, width=int(view_brush_size), joint="curve")
        # Also draw circles at ends to make it round
        r_view = view_brush_size / 2
        draw_view.ellipse([vx - r_view, vy - r_view, vx + r_view, vy + r_view], fill=self.current_class)
        
        # Draw on Original Mask (Full Res)
        draw_orig = ImageDraw.Draw(self.mask_image)
        draw_orig.line([lox, loy, ox, oy], fill=self.current_class, width=self.brush_size, joint="curve")
        r_orig = self.brush_size / 2
        draw_orig.ellipse([ox - r_orig, oy - r_orig, ox + r_orig, oy + r_orig], fill=self.current_class)

        self.last_x = cx
        self.last_y = cy
        
        # Update display
        self.update_display()
        
        # Keep cursor visible/updated during paint
        self.update_cursor(event)

    def update_cursor(self, event):
        if not self.view_image:
            return
            
        x, y = event.x, event.y
        # Calculate radius on canvas
        # brush_size is in original image pixels
        # view_scale is view/original
        # scale is canvas/view
        # So radius on canvas = (brush_size * view_scale * scale) / 2
        
        r = (self.brush_size * self.view_scale * self.scale) / 2
        
        if self.cursor_id is None:
            self.cursor_id = self.canvas.create_oval(x-r, y-r, x+r, y+r, outline="white", width=2, tag="cursor")
        else:
            self.canvas.coords(self.cursor_id, x-r, y-r, x+r, y+r)
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
        # We can't easily update cursor position here without mouse event, 
        # but next mouse move will fix it.

    def adjust_brush_size(self, delta):
        new_size = self.brush_size + delta
        new_size = max(1, min(2000, new_size))
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

    def save_mask(self):
        if self.mask_image and self.mask_dir and self.current_image_path:
            filename = os.path.basename(self.current_image_path)
            save_path = self._get_mask_path(filename)
            if save_path:
                self.mask_image.save(save_path)
                print(f"Saved mask to {save_path}")
                # Optional: Visual feedback
                self.root.title(f"Image Segmentation Tool - Saved {filename}")
        elif not self.mask_dir:
            messagebox.showwarning("Warning", "Please select a mask output directory first.")

    def show_help(self):
        help_text = """
        Image Segmentation Tool Guide
        
        1. Setup:
           - Click 'Open Images' to select the folder containing your images.
           - Click 'Open Masks' to select where to save the masks.
        
        2. Painting:
           - Select a Class from the right panel or use keys 1-7.
           - Adjust Brush Size using the slider or keys '[' and ']'.
           - Left-click and drag on the image to paint.
           - The mask is saved as a grayscale image (0=Background, 1=Class 1, etc.).
        
        3. Navigation:
           - Use 'Prev' and 'Next' buttons or Left/Right arrow keys.
           - 'Save' (Ctrl+S) saves the current mask.
           
        4. Shortcuts:
           - 1-7: Select Class
           - [ / ]: Decrease/Increase Brush Size
           - Left/Right: Previous/Next Image
           - Ctrl+S: Save Mask
        """
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
