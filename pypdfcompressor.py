#!/usr/bin/env python3
import os
import argparse
import subprocess
from typing import Optional, Dict
import fitz  # PyMuPDF
from datetime import datetime, timezone
import time
from tqdm import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.user = os.getlogin()

    def get_max_threads(self):
        """Get the maximum number of threads available on the system"""
        return psutil.cpu_count(logical=True)

    def log_with_timestamp(
        self, message: str, level: str = "info", thread_name: str = None
    ):
        """Log message with current UTC timestamp using timezone-aware datetime"""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        thread_info = f"[{thread_name}] " if thread_name else ""
        if level.lower() == "error":
            logger.error(f"{timestamp} - {thread_info}{message}")
        else:
            logger.info(f"{timestamp} - {thread_info}{message}")

    def get_quick_pdf_info(
        self, pdf_path: str, thread_name: str = None
    ) -> Optional[int]:
        """Quick PDF analysis - just gets basic info without page-by-page analysis"""
        try:
            doc = fitz.open(pdf_path)
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

            self.log_with_timestamp(
                f"\nQuick Analysis of: {os.path.basename(pdf_path)}",
                thread_name=thread_name,
            )
            self.log_with_timestamp(
                f"- Total Pages: {len(doc)}", thread_name=thread_name
            )
            self.log_with_timestamp(
                f"- File Size: {file_size_mb:.2f} MB", thread_name=thread_name
            )

            # Quick check of first page only for DPI
            if len(doc) > 0:
                first_page = doc[0]
                image_list = first_page.get_images()
                max_dpi = 0
                for img in image_list:
                    xref = img[0]
                    image = doc.extract_image(xref)
                    if image and "dpi" in image:
                        max_dpi = max(max_dpi, max(image.get("dpi", (0, 0))))

                if max_dpi > 0:
                    self.log_with_timestamp(
                        f"- Sample DPI (first page): {max_dpi}", thread_name=thread_name
                    )

            doc.close()
            return max_dpi if max_dpi > 0 else None

        except Exception as e:
            self.log_with_timestamp(
                f"Error in quick analysis of {pdf_path}: {str(e)}", "error", thread_name
            )
            return None

    def analyze_pdf_page(
        self, page: fitz.Page, page_num: int, thread_name: str = None
    ) -> Dict:
        """Analyze a single PDF page for detailed information"""
        result = {
            "page_number": page_num,
            "images": [],
            "max_dpi": 0,
            "total_images": 0,
            "page_size": f"{page.rect.width:.2f}x{page.rect.height:.2f} points",
        }

        image_list = page.get_images()
        result["total_images"] = len(image_list)

        for img_idx, img in enumerate(image_list, 1):
            xref = img[0]
            try:
                image = page.parent.extract_image(xref)
                if image:
                    img_info = {
                        "index": img_idx,
                        "width": image.get("width", 0),
                        "height": image.get("height", 0),
                        "dpi": max(image.get("dpi", (0, 0))),
                        "colorspace": image.get("colorspace", "Unknown"),
                        "size_kb": len(image.get("image", b"")) / 1024,
                    }
                    result["images"].append(img_info)
                    result["max_dpi"] = max(result["max_dpi"], img_info["dpi"])
            except Exception as e:
                self.log_with_timestamp(
                    f"Error analyzing image {img_idx} on page {page_num}: {str(e)}",
                    "error",
                    thread_name,
                )

        return result

    def get_pdf_dpi(self, pdf_path: str, thread_name: str = None) -> Optional[int]:
        """Extract detailed DPI and page information from a PDF file"""
        try:
            self.log_with_timestamp(f"\n{'='*50}", thread_name=thread_name)
            self.log_with_timestamp(
                f"Starting detailed analysis of: {os.path.basename(pdf_path)}",
                thread_name=thread_name,
            )
            self.log_with_timestamp(f"{'='*50}", thread_name=thread_name)

            doc = fitz.open(pdf_path)
            max_dpi = 0
            total_images = 0
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

            self.log_with_timestamp(f"Document Properties:", thread_name=thread_name)
            self.log_with_timestamp(
                f"- Total Pages: {len(doc)}", thread_name=thread_name
            )
            self.log_with_timestamp(
                f"- File Size: {file_size_mb:.2f} MB", thread_name=thread_name
            )
            self.log_with_timestamp(
                f"- PDF Version: {doc.metadata.get('format', 'Unknown')}",
                thread_name=thread_name,
            )
            self.log_with_timestamp(
                f"- Title: {doc.metadata.get('title', 'Untitled')}",
                thread_name=thread_name,
            )
            self.log_with_timestamp(
                f"\nStarting page-by-page analysis:", thread_name=thread_name
            )

            for page_num, page in enumerate(doc, 1):
                self.log_with_timestamp(
                    f"\nAnalyzing Page {page_num}/{len(doc)}:", thread_name=thread_name
                )
                page_info = self.analyze_pdf_page(page, page_num, thread_name)

                # Log page details
                self.log_with_timestamp(
                    f"  Page Size: {page_info['page_size']}", thread_name=thread_name
                )
                self.log_with_timestamp(
                    f"  Images Found: {page_info['total_images']}",
                    thread_name=thread_name,
                )

                if page_info["images"]:
                    self.log_with_timestamp("  Image Details:", thread_name=thread_name)
                    for img in page_info["images"]:
                        self.log_with_timestamp(
                            f"    - Image {img['index']}: "
                            f"{img['width']}x{img['height']} pixels, "
                            f"DPI: {img['dpi']}, "
                            f"ColorSpace: {img['colorspace']}, "
                            f"Size: {img['size_kb']:.2f}KB",
                            thread_name=thread_name,
                        )

                max_dpi = max(max_dpi, page_info["max_dpi"])
                total_images += page_info["total_images"]

            doc.close()

            self.log_with_timestamp(f"\nAnalysis Summary:", thread_name=thread_name)
            self.log_with_timestamp(
                f"- Maximum DPI found: {max_dpi}", thread_name=thread_name
            )
            self.log_with_timestamp(
                f"- Total images: {total_images}", thread_name=thread_name
            )
            self.log_with_timestamp(f"{'='*50}\n", thread_name=thread_name)

            return max_dpi if max_dpi > 0 else None

        except Exception as e:
            self.log_with_timestamp(
                f"Error analyzing PDF {pdf_path}: {str(e)}", "error", thread_name
            )
            return None

    def compress_single_pdf(self, args):
        """Wrapper function for compressing a single PDF (used with ThreadPoolExecutor)"""
        input_path, output_path, quality = args
        try:
            return self.compress_pdf(input_path, output_path, quality)
        except Exception as e:
            self.log_with_timestamp(f"Error processing {input_path}: {str(e)}", "error")
            return False

    def compress_pdf(self, input_path: str, output_path: str, quality: int = 50, fast_mode: bool = True) -> bool:
        """Compress a single PDF file while preserving DPI and OCR layers."""
        try:
            # Convert to absolute paths
            input_path = os.path.abspath(input_path)
            output_path = os.path.abspath(output_path)
            
            if not os.path.exists(input_path):
                self.log_with_timestamp(f"Input file not found: {input_path}", "error")
                return False
                
            start_time = time.time()
            thread_name = f"Thread-{threading.current_thread().ident}"
            
            self.log_with_timestamp(f"Processing file: {input_path}", thread_name=thread_name)
            
            # Use quick analysis in fast mode, detailed analysis otherwise
            if fast_mode:
                original_dpi = self.get_quick_pdf_info(input_path, thread_name)
            else:
                original_dpi = self.get_pdf_dpi(input_path, thread_name)
            
            # Calculate compression settings
            compression_level = max(0, min(9, int((100 - quality) / 11)))
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Prepare Ghostscript command with quoted paths
            gs_call = [
                'gswin64c.exe',
                '-sDEVICE=pdfwrite',
                '-dCompatibilityLevel=1.5',
                '-dNOPAUSE',
                '-dBATCH',
                '-dSAFER',
                '-dPrinted=false',  # Add this line
                '-dPreserveAnnots=true',
                '-dAutoRotatePages=/None',
                '-dPreservePDFObjects=true',
                '-dPreserveMarkedContent=true',
                '-dPreserveStructure=true',
                f'-dCompressFonts=true',
                f'-dCompressPages=true',
                f'-dCompressLevel={compression_level}',
                f'-dJPEGQ={quality}',
                '-dColorImageDownsampleType=/Bicubic',
                '-dGrayImageDownsampleType=/Bicubic',
                '-dMonoImageDownsampleType=/Bicubic',
                f'-dColorImageDepth={8 if quality < 50 else 16 if quality < 85 else 24}',
            ]

            if original_dpi:
                gs_call.extend([
                    f'-dColorImageResolution={original_dpi}',
                    f'-dGrayImageResolution={original_dpi}',
                    f'-dMonoImageResolution={original_dpi}'
                ])

            # Quote the file paths
            gs_call.extend([
                f'-sOutputFile="{output_path}"',
                f'"{input_path}"'
            ])

            # Execute compression
            self.log_with_timestamp(f"Starting compression with command: {' '.join(gs_call)}", thread_name=thread_name)
            process = subprocess.run(
                ' '.join(gs_call),  # Join command as string
                shell=True,  # Use shell to handle quoted paths
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            if process.returncode == 0 and os.path.exists(output_path):
                initial_size = os.path.getsize(input_path) / (1024 * 1024)  # Convert to MB
                final_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
                compression_ratio = (1 - final_size/initial_size) * 100
                elapsed_time = time.time() - start_time
                
                self.log_with_timestamp(
                    f"\nCompression Results for {os.path.basename(input_path)}:", 
                    thread_name=thread_name
                )
                self.log_with_timestamp(
                    f"- Initial size: {initial_size:.2f}MB", 
                    thread_name=thread_name
                )
                self.log_with_timestamp(
                    f"- Final size: {final_size:.2f}MB", 
                    thread_name=thread_name
                )
                self.log_with_timestamp(
                    f"- Compression ratio: {compression_ratio:.2f}%", 
                    thread_name=thread_name
                )
                self.log_with_timestamp(
                    f"- Processing time: {elapsed_time:.2f}s", 
                    thread_name=thread_name
                )
                return True
            else:
                self.log_with_timestamp(
                    f"Failed to compress {input_path}\nError: {process.stderr}", 
                    "error", 
                    thread_name=thread_name
                )
                return False
                
        except Exception as e:
            self.log_with_timestamp(
                f"Error compressing {input_path}: {str(e)}", 
                "error", 
                thread_name=thread_name
            )
            return False

    def process_directory(self, input_folder: str, output_folder: str, quality: int, fast_mode: bool = True) -> None:
        """Process all PDFs in a directory recursively using multithreading"""
        try:
            # Convert to absolute paths
            input_folder = os.path.abspath(input_folder)
            output_folder = os.path.abspath(output_folder)
            
            # Get list of PDF files
            pdf_files = []
            for root, _, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        input_path = os.path.abspath(os.path.join(root, file))
                        rel_path = os.path.relpath(os.path.dirname(input_path), input_folder)
                        output_subdir = os.path.join(output_folder, rel_path)
                        os.makedirs(output_subdir, exist_ok=True)
                        output_path = os.path.abspath(os.path.join(output_subdir, f"{os.path.basename(file)}"))
                        pdf_files.append((input_path, output_path, quality))

            if not pdf_files:
                self.log_with_timestamp("No PDF files found in the input directory")
                return

            # Get maximum number of threads
            max_threads = min(len(pdf_files), self.get_max_threads())
            self.log_with_timestamp(f"Processing {len(pdf_files)} files using {max_threads} threads")

            # Process files using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = []
                for args in pdf_files:
                    input_path, output_path, quality = args
                    self.log_with_timestamp(f"Queueing file: {input_path}")
                    futures.append(
                        executor.submit(
                            self.compress_pdf, 
                            input_path, 
                            output_path, 
                            quality,
                            fast_mode
                        )
                    )
                
                # Process results with progress bar
                successful = 0
                failed = 0
                with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file") as pbar:
                    for future in as_completed(futures):
                        try:
                            if future.result():
                                successful += 1
                            else:
                                failed += 1
                        except Exception as e:
                            self.log_with_timestamp(f"Error in thread: {str(e)}", "error")
                            failed += 1
                        pbar.update(1)

            # Log final statistics
            self.log_with_timestamp(
                f"\nCompression Statistics:"
                f"\n- Total files processed: {len(pdf_files)}"
                f"\n- Successfully compressed: {successful}"
                f"\n- Failed: {failed}"
                f"\n- Success rate: {(successful/len(pdf_files)*100):.2f}%"
            )

        except Exception as e:
            self.log_with_timestamp(f"Error processing directory: {str(e)}", "error")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Recursively compress PDF files while preserving DPI and OCR layers"
    )

    parser.add_argument(
        "input", type=str, help="Input PDF file or directory containing PDF files"
    )

    parser.add_argument(
        "output", type=str, help="Output directory for compressed PDF files"
    )

    parser.add_argument(
        "--quality",
        type=int,
        choices=range(0, 101),
        default=50,
        metavar="0-100",
        help="Compression quality (0-100, higher = better quality but larger file size, default: 50)",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode (skip detailed page analysis)",
    )

    args = parser.parse_args()
    processor = PDFProcessor()

    # Log start of processing
    processor.log_with_timestamp("=== PDF Compression Task Started ===")
    processor.log_with_timestamp(
        f"Mode: {'Fast' if args.fast else 'Detailed analysis'}"
    )
    processor.log_with_timestamp(
        f"Available CPU threads: {processor.get_max_threads()}"
    )
    processor.log_with_timestamp(f"Input path: {args.input}")
    processor.log_with_timestamp(f"Output path: {args.output}")
    processor.log_with_timestamp(f"Quality setting: {args.quality}")

    try:
        if os.path.isfile(args.input):
            processor.log_with_timestamp("Processing single file mode")
            os.makedirs(args.output, exist_ok=True)
            output_path = os.path.join(
                args.output, f"{os.path.basename(args.input)}"
            )
            processor.compress_pdf(args.input, output_path, args.quality, args.fast)
        elif os.path.isdir(args.input):
            processor.log_with_timestamp("Processing directory mode")
            processor.process_directory(
                args.input, args.output, args.quality, args.fast
            )
        else:
            processor.log_with_timestamp(
                f"Error: {args.input} is not a valid file or directory", "error"
            )
            return

        processor.log_with_timestamp("=== PDF Compression Task Completed ===")

    except Exception as e:
        processor.log_with_timestamp(f"Task failed: {str(e)}", "error")
        processor.log_with_timestamp("=== PDF Compression Task Failed ===")


if __name__ == "__main__":
    main()
