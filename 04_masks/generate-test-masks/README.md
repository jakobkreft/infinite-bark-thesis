# Test masks

Generate a small, hand-designed suite of 3-class PNG masks (uint8 values 0/1/2)
for sanity-testing the diffusion model's mask conditioning: tileability, sharp
boundaries, thin and circular structures, and multi-scale/overlap stress cases.

`generate_test_masks.py` — writes the mask suite (and color-visualized versions).

Run: `python generate_test_masks.py`
