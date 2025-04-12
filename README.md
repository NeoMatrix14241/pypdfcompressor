# PyPDFCompressor

Ang malupit na compressor ng PDF with lossless quality orrayt

A powerful Python utility for compressing PDF files while preserving DPI (dots per inch) and OCR (Optical Character Recognition) layers. This tool offers both single-file and batch processing capabilities with multi-threading support.

## Features

- 🚀 Multi-threaded batch processing
- 🎯 DPI preservation
- 📝 OCR layer preservation
- 📊 Detailed analysis mode
- ⚡ Fast mode for quick compression
- 📈 Progress tracking
- 📋 Comprehensive logging
- 🌲 Recursive directory processing
- 🎨 Adjustable compression quality

## Requirements

### Dependencies
1. Python Modules
```bash
pip install PyMuPDF tqdm psutil
```
2. **[GhostScript](https://ghostscript.com/releases/gsdnld.html)**

### System Requirements
- Python 3.x
- Ghostscript (`gswin64c.exe` must be in system PATH)
- Sufficient system memory for processing large PDF files
- Multi-core processor (recommended)

## Installation

1. Clone this repository or download the script
2. Install required Python packages:
```bash
pip install PyMuPDF tqdm psutil
```
3. Download and install ghostscript https://ghostscript.com/releases/gsdnld.html
4. Add ghostscript to environment path

## Usage

Just run the **start_compressor.bat**

### Basic Command Structure
```bash
python pdf_compressor.py <input> <output> [--quality QUALITY] [--fast]
```

### Arguments

- `input`: Path to input PDF file or directory containing PDF files
- `output`: Path to output directory for compressed files
- `--quality`: Compression quality (0-100, default: 50)
  - Higher values = better quality but larger file size
  - Lower values = smaller file size but lower quality
- `--fast`: Enable fast mode (skips detailed page analysis)

### Examples when using your own command/batch file

1. Compress a single PDF file:
```bash
python pdf_compressor.py document.pdf output_directory
```

2. Compress all PDFs in a directory:
```bash
python pdf_compressor.py input_directory output_directory
```

3. Compress with high quality (100):
```bash
python pdf_compressor.py document.pdf output_directory --quality 100
```

4. Use fast mode for quick compression:
```bash
python pdf_compressor.py input_directory output_directory --fast
```

## Feature Details

### Multi-threading
- Automatically detects and utilizes available CPU cores
- Processes multiple files simultaneously for faster batch compression

### Analysis Modes

#### Detailed Mode (Default)
- Comprehensive page-by-page analysis
- Detailed image information extraction
- DPI detection for each image
- Provides extensive compression statistics

#### Fast Mode
- Quick file analysis
- Skips detailed page inspection
- Faster processing time
- Suitable for batch processing

## Output Example

```
=== PDF Compression Task Started ===
Mode: Detailed analysis
Available CPU threads: 8
Input path: /path/to/input
Output path: /path/to/output
Quality setting: 50

Processing file: example.pdf
- Initial size: 10.5MB
- Final size: 3.2MB
- Compression ratio: 69.52%
- Processing time: 2.34s
```

## Technical Details

### Preservation Features
- Maintains original DPI settings
- Preserves OCR layers
- Retains PDF annotations
- Maintains PDF structure and marked content
- Preserves color depth based on quality settings

### Quality Settings Impact
- 0-49: 8-bit color depth
- 50-84: 16-bit color depth
- 85-100: 24-bit color depth

## Performance Tips

1. Use `--fast` mode for large batch operations
2. Adjust `--quality` based on your needs:
   - Use 30-50 for archival
   - Use 60-75 for general use
   - Use 80-100 for high-quality requirements
3. Ensure sufficient disk space in output directory
4. Close other memory-intensive applications during processing
