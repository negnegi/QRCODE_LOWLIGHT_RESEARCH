"""
Generate 150 standard, high-quality QR codes (EVENT2026-0001 ... EVENT2026-0150).
The generated codes are in data/qr_codes/. Codes EVENT2026-0101 ... 0150 were the
ones used for the captured datasets (see README).
- Codes are made at STANDARD quality (black on white, full contrast).
- Low light is applied later, when capturing, NOT here.
- Each code has unique, known content = the ground truth.
- A metadata CSV records the correct content for every code.
"""
import qrcode
import csv
import os

OUT_DIR = "qr_codes_150"
os.makedirs(OUT_DIR, exist_ok=True)

# How many codes and how they map to the 3 lighting conditions.
# 150 total: 50 low, 50 medium, 50 high (assigned in order).
TOTAL = 150
PER_CONDITION = 50
CONDITIONS = ["low", "medium", "high"]

rows = []
for i in range(1, TOTAL + 1):
    # Unique, known content for each code (this is the ground truth).
    content = f"EVENT2026-{i:04d}"

    # Which lighting condition this code is planned for.
    condition = CONDITIONS[(i - 1) // PER_CONDITION]

    qr = qrcode.QRCode(
        version=None,                                    # auto-size to fit content
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # H = most robust (good for low light)
        box_size=12,                                     # pixel size of each module
        border=4,                                        # white quiet zone (keep >= 4)
    )
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    filename = f"SCR_{condition}_{i:04d}.png"
    img.save(os.path.join(OUT_DIR, filename))

    rows.append({
        "index": i,
        "filename": filename,
        "ground_truth": content,          # the correct decoded string
        "medium": "screen",
        "planned_condition": condition,   # low / medium / high
        "error_correction": "H",
    })

# Write the metadata / ground-truth CSV.
with open("qr_metadata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["index", "filename", "ground_truth", "medium",
                    "planned_condition", "error_correction"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {TOTAL} QR codes in ./{OUT_DIR}/")
print("Metadata written to qr_metadata.csv")
