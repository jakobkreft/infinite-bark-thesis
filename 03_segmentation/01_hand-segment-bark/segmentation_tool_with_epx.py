import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES = {
    0: {'name': 'Background', 'color': (0, 0, 0)},
    1: {'name': 'slepice / odrezane veje', 'color': (0, 255, 255)},      # Cyan
    2: {'name': 'mehanske poškodbe / odluščeno', 'color': (255, 128, 0)},  # Orange
}
NUM_CLASSES = len(CLASSES)

DEFAULT_BRUSH_SIZE = 1            # In grid cells
DEFAULT_OPACITY = 0.2

# NEW PRINCIPLE:
# The mask resolution is derived from the ORIGINAL image.
# 1 pixel in the mask == MASK_CELL_SIZE x MASK_CELL_SIZE pixels in the original.
MASK_CELL_SIZE = 50

# The original (cropped) image can be very large, so we composite the overlay
# on a downscaled copy for responsiveness. The full-res image is kept for saving.
DISPLAY_MAX_WIDTH = 1500


# ---------------------------------------------------------------------------
# EPX / Scale2x upscaling (label-preserving)
# ---------------------------------------------------------------------------
def scale2x(arr):
    """One EPX / Scale2x pass: doubles resolution, smoothing diagonals.

    Works on an integer label array (each value is a class id) so it never
    invents intermediate / invalid labels the way a bilinear resize would.

            B
          D E F
            H
    """
    h, w = arr.shape

    up = np.empty_like(arr);    up[1:] = arr[:-1];     up[0] = arr[0]       # B
    down = np.empty_like(arr);  down[:-1] = arr[1:];   down[-1] = arr[-1]   # H
    left = np.empty_like(arr);  left[:, 1:] = arr[:, :-1];  left[:, 0] = arr[:, 0]    # D
    right = np.empty_like(arr); right[:, :-1] = arr[:, 1:]; right[:, -1] = arr[:, -1]  # F

    E = arr
    e0 = np.where((left == up) & (up != right) & (left != down), left, E)    # top-left
    e1 = np.where((up == right) & (up != left) & (right != down), right, E)  # top-right
    e2 = np.where((left == down) & (down != right) & (left != up), left, E)  # bottom-left
    e3 = np.where((down == right) & (down != left) & (right != up), right, E)  # bottom-right

    out = np.empty((h * 2, w * 2), dtype=arr.dtype)
    out[0::2, 0::2] = e0
    out[0::2, 1::2] = e1
    out[1::2, 0::2] = e2
    out[1::2, 1::2] = e3
    return out


def epx_upscale_to(mask_arr, target_w, target_h):
    """Upscale a low-res label mask to (target_w, target_h) using EPX/Scale2x.

    Scale2x only doubles, and our target scale (MASK_CELL_SIZE = 50) is not a
    power of two, so we repeatedly apply Scale2x until we *exceed* the target,
    then drop down to the exact target with NEAREST (which preserves labels).
    The overshoot-then-downsample keeps the smoothed EPX edges.
    """
    cur = mask_arr.astype(np.uint8)
    while cur.shape[1] < target_w or cur.shape[0] < target_h:
        cur = scale2x(cur)

    img = Image.fromarray(cur, mode='L')
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.NEAREST)
    return img


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class SegmentationAppLowRes:
    def __init__(self, root):
        self.root = root
        self.root.title("Low-Res Image Segmentation Tool (EPX)")
        self.root.geometry("1200x800")

        # Directories
        self.image_dir = ""
        self.mask_dir = ""
        self.output_image_dir = ""
        self.epx_mask_dir = ""

        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None

        self.full_image = None       # PIL RGB - full-res, cropped to fit mask exactly (saved output)
        self.processed_image = None  # PIL RGB - downscaled copy used for on-screen compositing
        self.mask_image = None       # PIL L   - low-res mask (mask_w x mask_h)

        self.display_image = None    # PIL RGB - blended (image + overlay) at display resolution
        self.tk_image = None         # ImageTk for Canvas

        self.current_class = 1
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.opacity = DEFAULT_OPACITY

        self.scale = 1.0             # Canvas scale relative to processed_image
        self.offset_x = 0
        self.offset_y = 0

        self.last_grid_x = None
        self.last_grid_y = None

        self.cursor_id = None
        self.cell_size_px = 0.0      # Size of one grid cell in processed_image (display) pixels

        self._setup_ui()
        self._bind_events()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        # Top Toolbar
        self.toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(self.toolbar, text="1. Input Images", command=self.select_image_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="2. Output Images", command=self.select_output_image_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="3. Output Masks", command=self.select_mask_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="4. EPX Masks", command=self.select_epx_mask_dir).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="Save (Ctrl+S)", command=self.save_data).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.toolbar, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Frame(self.toolbar, width=20).pack(side=tk.LEFT)  # Spacer

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

            rb = tk.Radiobutton(frame, text=f"{cls_id}: {cls_info['name']}", variable=self.class_var,
                                value=cls_id, command=self.change_class)
            rb.pack(side=tk.LEFT)

        tk.Label(self.controls, text="Brush Size (Grid Cells)", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        self.scale_brush = tk.Scale(self.controls, from_=1, to=10, orient=tk.HORIZONTAL, command=self.change_brush_size)
        self.scale_brush.set(self.brush_size)
        self.scale_brush.pack(fill=tk.X, padx=10)

        self.lbl_grid_info = tk.Label(self.controls, text="Mask: -", justify=tk.LEFT, fg="gray20")
        self.lbl_grid_info.pack(pady=(20, 5), padx=10, anchor="w")

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

        for cls_id in CLASSES:
            self.root.bind(str(cls_id), lambda e, idx=cls_id: self.set_class(idx))

        self.canvas.bind("<Button-1>", self.start_paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_paint)
        # Right click erases (paints background)
        self.canvas.bind("<Button-3>", self.start_erase)
        self.canvas.bind("<B3-Motion>", self.erase)
        self.canvas.bind("<ButtonRelease-3>", self.stop_paint)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Motion>", self.update_cursor)
        self.canvas.bind("<Leave>", self.hide_cursor)
        self.canvas.bind("<Enter>", self.show_cursor)

    # ------------------------------------------------------------ directories
    def select_image_dir(self):
        path = filedialog.askdirectory(title="Select Input Images Directory")
        if path:
            self.image_dir = path
            self.image_files = sorted([f for f in os.listdir(path)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
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
        path = filedialog.askdirectory(title="Select Output Masks Directory (low-res)")
        if path:
            self.mask_dir = path

    def select_epx_mask_dir(self):
        path = filedialog.askdirectory(title="Select EPX Masks Directory (full-res, Scale2x)")
        if path:
            self.epx_mask_dir = path

    # ----------------------------------------------------------------- loading
    def load_image(self):
        if not self.image_files or self.current_image_index < 0:
            return

        filename = self.image_files[self.current_image_index]
        self.current_image_path = os.path.join(self.image_dir, filename)
        self.lbl_image_name.config(text=f"{filename} ({self.current_image_index + 1}/{len(self.image_files)})")

        try:
            # 1. Load original at full resolution
            original = Image.open(self.current_image_path).convert("RGB")
            w, h = original.size

            # 2. Derive mask resolution from the original: 1 mask px = MASK_CELL_SIZE original px
            mask_w = max(1, w // MASK_CELL_SIZE)
            mask_h = max(1, h // MASK_CELL_SIZE)

            # 3. Crop the original so it fits the mask grid exactly (center crop)
            crop_w = mask_w * MASK_CELL_SIZE
            crop_h = mask_h * MASK_CELL_SIZE
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            self.full_image = original.crop((left, top, left + crop_w, top + crop_h))

            # 4. Build a downscaled copy for fast on-screen compositing
            if crop_w > DISPLAY_MAX_WIDTH:
                disp_scale = DISPLAY_MAX_WIDTH / crop_w
                disp_w = DISPLAY_MAX_WIDTH
                disp_h = max(1, int(round(crop_h * disp_scale)))
                self.processed_image = self.full_image.resize((disp_w, disp_h), Image.LANCZOS)
            else:
                self.processed_image = self.full_image.copy()

            # Grid cell size measured in the *display* image
            self.cell_size_px = self.processed_image.width / mask_w

            # 5. Initialize / load the low-res mask (mask_w x mask_h)
            mask_size = (mask_w, mask_h)
            mask_path = self._get_mask_path(filename)
            if mask_path and os.path.exists(mask_path):
                loaded_mask = Image.open(mask_path).convert("L")
                self.mask_image = loaded_mask if loaded_mask.size == mask_size else Image.new("L", mask_size, 0)
            else:
                self.mask_image = Image.new("L", mask_size, 0)

            self.lbl_grid_info.config(
                text=f"Original: {w}x{h}\nCropped: {crop_w}x{crop_h}\nMask: {mask_w}x{mask_h}\nCell: {MASK_CELL_SIZE}px"
            )

            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _get_mask_path(self, image_filename):
        if not self.mask_dir:
            return None
        basename = os.path.splitext(image_filename)[0]
        return os.path.join(self.mask_dir, basename + ".png")

    def _get_epx_mask_path(self, image_filename):
        if not self.epx_mask_dir:
            return None
        basename = os.path.splitext(image_filename)[0]
        return os.path.join(self.epx_mask_dir, basename + ".png")

    def _get_output_image_path(self, image_filename):
        if not self.output_image_dir:
            return None
        basename = os.path.splitext(image_filename)[0]
        return os.path.join(self.output_image_dir, basename + ".jpg")

    # --------------------------------------------------------------- rendering
    def update_display(self):
        if self.processed_image is None:
            return

        mask_array = np.array(self.mask_image)

        # Colored overlay (low-res), then nearest-upscale to display size
        overlay_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
        for cls_id, cls_info in CLASSES.items():
            if cls_id == 0:
                continue
            overlay_array[mask_array == cls_id] = cls_info['color']

        overlay_small = Image.fromarray(overlay_array, mode='RGB')
        overlay_full = overlay_small.resize(self.processed_image.size, Image.NEAREST)

        mask_bool = mask_array > 0
        mask_alpha_small = Image.fromarray((mask_bool * 255 * self.opacity).astype(np.uint8), mode='L')
        mask_alpha_full = mask_alpha_small.resize(self.processed_image.size, Image.NEAREST)

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

        if cw == 0 or ch == 0:
            return

        self.scale = min(cw / iw, ch / ih)
        new_w = max(1, int(iw * self.scale))
        new_h = max(1, int(ih * self.scale))

        resized = self.display_image.resize((new_w, new_h), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.offset_x = (cw - new_w) // 2
        self.offset_y = (ch - new_h) // 2

        self.canvas.delete("all")
        self.cursor_id = None
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

    def on_resize(self, event):
        if self.processed_image:
            self.draw_image_on_canvas()

    # ----------------------------------------------------------------- painting
    def start_paint(self, event):
        self._apply_brush(event, self.current_class)

    def start_erase(self, event):
        self._apply_brush(event, 0)

    def paint(self, event):
        self._apply_brush(event, self.current_class)

    def erase(self, event):
        self._apply_brush(event, 0)

    def stop_paint(self, event):
        self.last_grid_x = None
        self.last_grid_y = None

    def _apply_brush(self, event, value):
        if not self.processed_image or self.mask_image is None:
            return

        # Canvas -> display image
        px = (event.x - self.offset_x) / self.scale
        py = (event.y - self.offset_y) / self.scale

        # Display image -> grid
        gx = int(px / self.cell_size_px)
        gy = int(py / self.cell_size_px)

        if gx < 0 or gx >= self.mask_image.width or gy < 0 or gy >= self.mask_image.height:
            return

        draw = ImageDraw.Draw(self.mask_image)

        if self.last_grid_x is not None:
            draw.line([self.last_grid_x, self.last_grid_y, gx, gy], fill=value, width=self.brush_size)

        r = self.brush_size // 2
        odd = 1 if self.brush_size % 2 != 0 else 0
        x1 = gx - r
        y1 = gy - r
        x2 = gx + r + odd
        y2 = gy + r + odd
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=value)

        self.last_grid_x = gx
        self.last_grid_y = gy

        self.update_display()
        self.update_cursor(event)

    # ------------------------------------------------------------------- cursor
    def update_cursor(self, event):
        if not self.processed_image:
            return

        px = (event.x - self.offset_x) / self.scale
        py = (event.y - self.offset_y) / self.scale

        gx = int(px / self.cell_size_px)
        gy = int(py / self.cell_size_px)

        r = self.brush_size // 2
        odd = 1 if self.brush_size % 2 != 0 else 0
        x1_g = gx - r
        y1_g = gy - r
        x2_g = gx + r + odd
        y2_g = gy + r + odd

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

    # ------------------------------------------------------------------ classes
    def change_class(self):
        self.current_class = self.class_var.get()

    def set_class(self, idx):
        if idx in CLASSES:
            self.current_class = idx
            self.class_var.set(idx)

    def change_brush_size(self, val):
        self.brush_size = int(val)

    def adjust_brush_size(self, delta):
        self.brush_size = max(1, min(10, self.brush_size + delta))
        self.scale_brush.set(self.brush_size)

    # --------------------------------------------------------------- navigation
    def prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()

    # -------------------------------------------------------------------- saving
    def save_data(self):
        if not self.mask_dir or not self.output_image_dir or not self.epx_mask_dir:
            messagebox.showwarning(
                "Warning",
                "Please select Output Images, Output Masks, and EPX Masks directories."
            )
            return

        if not (self.mask_image and self.full_image and self.current_image_path):
            return

        filename = os.path.basename(self.current_image_path)

        # 1. Low-res mask
        mask_path = self._get_mask_path(filename)
        self.mask_image.save(mask_path)

        # 2. Cropped full-res image (matches the mask grid exactly)
        img_path = self._get_output_image_path(filename)
        self.full_image.save(img_path, quality=95)

        # 3. EPX / Scale2x mask, upscaled to full image resolution
        mask_arr = np.array(self.mask_image)
        target_w, target_h = self.full_image.size
        epx_mask = epx_upscale_to(mask_arr, target_w, target_h)
        epx_path = self._get_epx_mask_path(filename)
        epx_mask.save(epx_path)

        print(f"Saved low-res mask : {mask_path}")
        print(f"Saved image        : {img_path}")
        print(f"Saved EPX mask     : {epx_path} ({target_w}x{target_h})")
        self.root.title(f"Low-Res Tool (EPX) - Saved {filename}")

    # ---------------------------------------------------------------------- help
    def show_help(self):
        help_text = """
        Low-Res Segmentation Tool (EPX) Guide

        1. Setup:
           - 1. Input Images   : source images.
           - 2. Output Images  : cropped (full-res) images go here.
           - 3. Output Masks   : low-res masks (one px = 50x50 px).
           - 4. EPX Masks      : full-res masks upscaled with Scale2x.

        2. Principle:
           - 1 mask pixel  = 50 x 50 pixels in the ORIGINAL image.
           - Mask size is derived from the original, then the image is
             center-cropped so it fits the grid perfectly (no resize/scaling
             of content, only cropping).

        3. Painting:
           - Left click / drag : paint current class.
           - Right click / drag: erase (paint background).
           - Brush size is in grid cells.
           - Keys 0/1/2 select class, [ and ] change brush size.

        4. Saving (Ctrl+S):
           - Low-res mask (mask_w x mask_h).
           - Cropped full-res image.
           - EPX/Scale2x mask at full image resolution (label-preserving,
             smoothed edges instead of blocky nearest-neighbour).
        """
        messagebox.showinfo("Help", help_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationAppLowRes(root)
    root.mainloop()
