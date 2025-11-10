"""
Download Pretrained SSVP-SLT Models
====================================

Download pretrained checkpoints from Facebook Research SSVP-SLT.

Usage:
    # Download Base model
    python download_pretrained.py --model base

    # Download Large model
    python download_pretrained.py --model large

    # Download all models
    python download_pretrained.py --model all
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import urllib.request
from tqdm import tqdm


# Model URLs and checksums
PRETRAINED_MODELS = {
    "base": {
        "url": "https://dl.fbaipublicfiles.com/ssvp_slt/how2sign/ssvp_base.pt",
        "md5": "placeholder_md5_hash_base",  # TODO: Update with actual hash
        "size_mb": 340,
        "description": "SSVP-SLT Base model (86M params) - MAE pretrained",
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/ssvp_slt/how2sign/ssvp_large.pt",
        "md5": "placeholder_md5_hash_large",
        "size_mb": 1200,
        "description": "SSVP-SLT Large model (307M params) - MAE pretrained",
    },
    "base_clip": {
        "url": "https://dl.fbaipublicfiles.com/ssvp_slt/how2sign/ssvp_base_clip.pt",
        "md5": "placeholder_md5_hash_base_clip",
        "size_mb": 360,
        "description": "SSVP-SLT Base model - MAE+CLIP pretrained",
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for download."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(
    url: str, output_path: Path, expected_md5: Optional[str] = None
) -> bool:
    """
    Download file from URL with progress bar.

    Args:
        url: Download URL
        output_path: Where to save file
        expected_md5: Expected MD5 checksum (optional)

    Returns:
        True if download successful, False otherwise
    """
    print(f"\n📥 Downloading from: {url}")
    print(f"📁 Saving to: {output_path}")

    try:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=output_path.name
        ) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )

        print("✅ Download complete!")

        # Verify checksum if provided
        if (
            expected_md5
            and expected_md5 != "placeholder_md5_hash_base"
            and expected_md5 != "placeholder_md5_hash_large"
            and expected_md5 != "placeholder_md5_hash_base_clip"
        ):
            print("\n🔍 Verifying checksum...")
            actual_md5 = compute_md5(output_path)

            if actual_md5 == expected_md5:
                print(f"✅ Checksum verified: {actual_md5}")
            else:
                print(f"❌ Checksum mismatch!")
                print(f"   Expected: {expected_md5}")
                print(f"   Got:      {actual_md5}")
                return False
        else:
            print("⚠️  Skipping checksum verification (hash not available)")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def download_model(model_name: str, output_dir: Path, force: bool = False) -> bool:
    """
    Download specific pretrained model.

    Args:
        model_name: Name of model ('base', 'large', 'base_clip')
        output_dir: Directory to save checkpoint
        force: Force re-download if file exists

    Returns:
        True if successful, False otherwise
    """
    if model_name not in PRETRAINED_MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"   Available models: {list(PRETRAINED_MODELS.keys())}")
        return False

    model_info = PRETRAINED_MODELS[model_name]
    output_path = output_dir / f"ssvp_{model_name}.pt"

    print("\n" + "=" * 80)
    print(f"📦 Model: SSVP-{model_name.upper()}")
    print("=" * 80)
    print(f"📝 Description: {model_info['description']}")
    print(f"💾 Size: ~{model_info['size_mb']} MB")
    print(f"📍 Output: {output_path}")

    # Check if already exists
    if output_path.exists() and not force:
        print(f"\n⚠️  File already exists: {output_path}")
        response = input("   Re-download? [y/N]: ").strip().lower()
        if response != "y":
            print("   Skipping download.")
            return True
        else:
            print("   Removing existing file...")
            output_path.unlink()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    success = download_file(
        url=model_info["url"], output_path=output_path, expected_md5=model_info["md5"]
    )

    if success:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Successfully downloaded: {output_path.name}")
        print(f"   Size: {file_size_mb:.1f} MB")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained SSVP-SLT models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Base model
  python download_pretrained.py --model base
  
  # Download Large model
  python download_pretrained.py --model large
  
  # Download all models
  python download_pretrained.py --model all
  
  # Force re-download
  python download_pretrained.py --model base --force
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(PRETRAINED_MODELS.keys()) + ["all"],
        help="Model to download: base, large, base_clip, or all",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints (default: models/checkpoints)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Force re-download if file exists"
    )

    args = parser.parse_args()

    # Convert output to Path
    output_dir = Path(args.output)

    print("=" * 80)
    print("🚀 SSVP-SLT Pretrained Model Downloader")
    print("=" * 80)
    print(f"\n📂 Output directory: {output_dir.absolute()}")

    # Download model(s)
    if args.model == "all":
        models_to_download = list(PRETRAINED_MODELS.keys())
        print(f"\n📦 Downloading {len(models_to_download)} models...")
    else:
        models_to_download = [args.model]

    success_count = 0
    failed_models = []

    for model_name in models_to_download:
        success = download_model(model_name, output_dir, force=args.force)
        if success:
            success_count += 1
        else:
            failed_models.append(model_name)

    # Summary
    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {success_count}/{len(models_to_download)}")

    if failed_models:
        print(f"❌ Failed: {', '.join(failed_models)}")
        print("\n⚠️  Some downloads failed. Please check errors above.")
        sys.exit(1)
    else:
        print("\n🎉 All downloads completed successfully!")
        print(f"\n📍 Models saved to: {output_dir.absolute()}")
        print("\n🚀 Next steps:")
        print("   1. Prepare How2Sign dataset:")
        print("      python prepare_how2sign_for_ssvp.py")
        print("")
        print("   2. Fine-tune model:")
        print("      python finetune_how2sign.py --config configs/finetune_base.yaml")
        sys.exit(0)


if __name__ == "__main__":
    main()
