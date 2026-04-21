#!/usr/bin/python3
# make sure you're logged in with `huggingface-cli login`
import argparse
import re
import subprocess
import glob
import os
import shutil
import sys
import platform
import torch
from diffusers import StableDiffusionPipeline


def newest_output() -> str:
    """Get the most recent output image. Raises ValueError if none exist."""
    flist = glob.glob("output/*.png")
    if not flist:
        raise ValueError("No images found in output/ directory")
    f = max(flist, key=os.path.getctime)
    return f


def newest_keeper() -> str:
    """Get the most recent keeper image. Raises ValueError if none exist."""
    flist = glob.glob("keepers/*.png")
    if not flist:
        raise ValueError("No images found in keepers/ directory")
    f = max(flist, key=os.path.getctime)
    return f


def ensure_directories():
    """Ensure required directories exist."""
    for directory in ["output", "keepers"]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def display_image(filepath: str):
    """Safely display an image using platform-appropriate viewer."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    try:
        if platform.system() == "Windows":
            subprocess.run(["explorer.exe", os.path.abspath(filepath)], check=True)
        elif platform.system() == "Darwin":
            subprocess.run(["open", os.path.abspath(filepath)], check=True)
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", os.path.abspath(filepath)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to display image: {e}")


def sanitize_filename(text: str) -> str:
    """Sanitize text to create a safe filename."""
    if not text or not isinstance(text, str):
        raise ValueError("Prompt must be a non-empty string")

    # Remove invalid filename characters and limit length
    sanitized = re.sub(r"[^a-zA-Z0-9 ]", "", text).split()
    if not sanitized:
        raise ValueError("Prompt contains no valid characters after sanitization")

    filename = "-".join(sanitized)
    # Limit filename length to prevent filesystem issues
    if len(filename) > 200:
        filename = filename[:200]

    return filename


def validate_copy_destination(destination: str):
    """Validate and normalize the copy destination path."""
    # Prevent path traversal attacks
    normalized = os.path.normpath(destination)

    # Reject absolute paths (must be relative)
    if os.path.isabs(normalized):
        raise ValueError("Destination path must be relative, not absolute")

    # Reject attempts to traverse parent directories
    if normalized.startswith(".."):
        raise ValueError("Destination path cannot traverse parent directories")

    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion text-to-image pipeline"
    )
    parser.add_argument(
        "-p", "--prompt", type=str, help="The prompt from which to generate an image"
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display the most recently-generated image",
    )
    parser.add_argument(
        "-k",
        "--keep",
        action="store_true",
        help="Move the most recently-generated image to keepers",
    )
    parser.add_argument(
        "-r",
        "--retrieve",
        action="store_true",
        help="Display the most recently-generated keeper",
    )
    parser.add_argument(
        "-c",
        "--copy",
        type=str,
        help="Copy the most recently-generated keeper to the specified location",
    )
    args = parser.parse_args()

    # Show help if no arguments provided
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    # Ensure required directories exist
    try:
        ensure_directories()
    except OSError as e:
        print(f"Error creating required directories: {e}", file=sys.stderr)
        sys.exit(1)

    # Process display action
    if args.display:
        try:
            f = newest_output()
            display_image(f)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"Error displaying image: {e}", file=sys.stderr)
            sys.exit(1)

    # Process keep action
    if args.keep:
        try:
            f = newest_output()
            if not os.path.exists("keepers"):
                os.makedirs("keepers", exist_ok=True)
            shutil.move(f, "keepers/")
        except (ValueError, FileNotFoundError, shutil.Error) as e:
            print(f"Error keeping image: {e}", file=sys.stderr)
            sys.exit(1)

    # Process retrieve action
    if args.retrieve:
        try:
            f = newest_keeper()
            display_image(f)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"Error retrieving image: {e}", file=sys.stderr)
            sys.exit(1)

    # Process copy action
    if args.copy:
        try:
            dest = validate_copy_destination(args.copy)
            f = newest_keeper()
            if os.path.exists(dest) and os.path.isdir(dest):
                shutil.copy(f, dest)
            else:
                shutil.copy(f, dest)
        except (ValueError, FileNotFoundError, shutil.Error) as e:
            print(f"Error copying image: {e}", file=sys.stderr)
            sys.exit(1)

    # Process prompt action
    if args.prompt:
        try:
            prompt = args.prompt
            if len(prompt) > 1000:
                raise ValueError("Prompt is too long (max 1000 characters)")

            # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_type=torch.float16, revision="fp16")
            pipe = StableDiffusionPipeline.from_pretrained(
                "./stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
            )
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()

            filename = os.path.join("output", sanitize_filename(prompt) + ".png")
            image = pipe(prompt, height=504, width=896).images[
                0
            ]  # maximum safe 16:9 size
            image.save(filename)
            print(f"Image saved to {filename}")
        except (ValueError, RuntimeError) as e:
            print(f"Error generating image: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
