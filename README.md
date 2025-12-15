# **Gliese-CUA-Tool-Call-8B-Localization**

> A Gradio-based demonstration for the prithivMLmods/Gliese-CUA-Tool-Call-8B model, specialized in GUI element localization. Users upload UI screenshots, provide task instructions (e.g., "Click on the search bar"), and receive predicted click coordinates in `Click(x, y)` format, visualized as crosshairs and labels on the image. Features model download to local directory for offline use, smart image resizing, and coordinate scaling to original resolution.

## Features

- **Element Localization**: Natural language tasks predict precise pixel coordinates for UI components (e.g., buttons, inputs).
- **Action Visualization**: Overlays red crosshairs with yellow labels on the output image using PIL for clear action points.
- **Smart Resizing**: Automatically resizes inputs based on model processor params (min/max pixels, patch/merge sizes) for optimal inference.
- **Coordinate Scaling**: Adjusts resized coordinates back to original image dimensions for accurate absolute positioning.
- **Efficient Inference**: Uses bfloat16/float32 precision on CUDA; generates up to 128 new tokens with deterministic output.
- **Local Model Storage**: Downloads model via Hugging Face Hub snapshot to `./model/` for faster reloads and offline capability.
- **Custom Theme**: OrangeRedTheme with gradients for an intuitive interface.
- **Queueing Support**: Handles up to 50 concurrent inferences.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16; falls back to CPU).
- Stable internet for initial model download (subsequent runs use local cache).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Gliese-CUA-Tool-Call-8B-Localization.git
   cd Gliese-CUA-Tool-Call-8B-Localization
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   gradio==6.1.0
   transformers==4.57.1
   huggingface-hub
   numpy
   torch
   torchvision
   accelerate
   qwen-vl-utils
   requests
   pillow
   spaces
   ```

3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces). The first run downloads the model (~8B params) to `./model/Gliese-CUA-Tool-Call-8B`.

## Usage

1. **Upload Image**: Provide a UI screenshot (e.g., PNG of a web page or app; height up to 500px).

2. **Enter Task**: Describe the target (e.g., "Locate the search bar" or "Find the submit button").

3. **Call CUA Agent**: Click the button to run inference.

4. **View Results**:
   - Text: Raw model response with parsed `Click(x, y)`.
   - Image: Annotated screenshot with crosshair visualization.

### Example Workflow
- Upload a browser screenshot.
- Task: "Click on the search bar."
- Output: `Click(250, 150)` and image with red crosshair on the bar.

## Troubleshooting

- **Model Download Fails**: Check internet; resume with `resume_download=True`. Verify `allow_patterns="Localization-8B/**"`.
- **Loading Errors**: Ensure transformers 4.57.1; check CUDA with `torch.cuda.is_available()`. Use `torch.float32` if bfloat16 OOM.
- **No Coordinates Parsed**: Task must be localization-focused; raw output in console. Increase max_new_tokens if needed.
- **Resizing Issues**: `smart_resize` enforces min/max pixels; fallback to original if errors.
- **Visualization Problems**: PIL font fallback used; ensure RGB images.
- **Queue Full**: Increase `max_size` in `demo.queue()`.
- **Spaces Deployment**: Install `spaces`; set `show_error=True` for debugging.

## Contributing

Contributions encouraged! Fork the repo, create a feature branch (e.g., for multi-target support), and submit PRs with tests. Focus areas:
- Extension to tool-calling beyond localization.
- Batch image processing.
- Custom prompt templates.

Repository: [https://github.com/PRITHIVSAKTHIUR/Gliese-CUA-Tool-Call-8B-Localization.git](https://github.com/PRITHIVSAKTHIUR/Gliese-CUA-Tool-Call-8B-Localization.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
