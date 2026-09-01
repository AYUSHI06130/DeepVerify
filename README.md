# DeepVerify Prototype

Run:

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000

The first image/video run downloads the Hugging Face model:
`shivani1511/deepfake-image-detector`

Optional:
`$env:FACTCHECK_API_KEY="YOUR_KEY"`

This is a prototype. The score is an evidence aggregation score, not a forensic guarantee.

Next upgrades: face-level analysis, temporal video detection, lip-sync analysis, OCR/claim extraction,
real reverse-image search, maintained source reputation data, database storage, and an evidence timeline.