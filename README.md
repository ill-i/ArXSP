# ArXSP
ArXSP (Archival Spectrum Processor) is a PyQt5-based desktop application for reducing archival FITS spectra. The application supports automated calibration workflows and manual image adjustments via an intuitive GUI.

## Requirements

- **Python** 3.8 or higher
- **Dependencies**: listed in `Pro.AP22784884/req.txt` (e.g., PyQt5, Astropy, NumPy, Matplotlib)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ill-i/ArXSP.git
   cd ArXSP
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r Pro.AP22784884/req.txt
   ```

## Directory Structure

```plaintext
├── Pro.AP22784884
│   ├── ArxSR.py                       # Core reduction algorithms
│   ├── calibration_polynomial.csv     # Calibration data
│   ├── crop_styles.css                # Stylesheet for cropping mode
│   ├── CvEditor.py                    # FITS header editor window
│   ├── dark.css                       # Dark theme stylesheet
│   ├── GUIcomp.py                     # Custom PyQt5 components
│   ├── HeaderEditWindow.py            # Header editing window
│   ├── light.css                      # Light theme stylesheet
│   ├── modelalign
│   │   └── model_align.py             # Spectrum alignment module
│   ├── modelchar
│   │   └── model_char.py              # Spectral characteristics module
│   ├── modelcropper
│   │   ├── crop_styles.css            # Module-specific stylesheet
│   │   └── model_cropper.py           # Frame cropping logic
│   ├── modelusage
│   │   └── model_usage.py             # Utility functions module
│   ├── mp_viewer_app.py               # Application entry point
│   ├── ProAP22784884.desktop          # Linux desktop shortcut
│   ├── Pro.AP22784884.py              # Main executable script
│   ├── ProAP22784884.spec             # RPM spec file
│   ├── qrc_resources.py               # Qt resource loader
│   ├── req.txt                        # List of Python dependencies
│   ├── resources/                     # Icons and images
│   └── resources.qrc                  # Qt resource collection
└── README.md                          # This file
``` 

## Usage

1. **Run the application** from the virtual environment:
   ```bash
   python Pro.AP22784884/Pro.AP22784884.py
   ```

## Acknowledgments

ArXSP is registered in the Republic of Kazakhstan under Certificate No. 60032 (Kazpatent), June 19, 2025.
