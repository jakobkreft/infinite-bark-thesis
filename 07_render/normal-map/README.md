# Normal map

Convert a (bark) texture into a tangent-space **normal map** so it can be used as
a material in Blender. The image is read as a grayscale heightmap (brighter =
higher) and gradients are taken with a Sobel filter. Output uses the OpenGL
(+Y up) convention, which Blender's Normal Map node expects.

`normalmap.py`

Run: `python normalmap.py input.png output.png [--strength 2.0] [--blur 1.5] [--flip-green]`
