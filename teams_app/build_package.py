"""
Build the Microsoft Teams app package (zip) for sideloading.

    python teams_app/build_package.py                          # https://localhost:53000
    python teams_app/build_package.py --url https://my-tunnel.example.net

Output: teams_app/build/vsl-teams-app.zip  (manifest.json + color.png + outline.png)
Upload it in Teams: Apps -> Manage your apps -> Upload an app -> Upload a custom app.

The app has a personal tab and a meeting tab (side panel during the meeting, plus the meeting
chat/details tab). Add it to a meeting with "+ Apps" in the meeting window.
"""

import argparse
import json
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

APP_DIR = Path(__file__).resolve().parent
BUILD = APP_DIR / "build"
APP_ID = "c722f7e1-97d5-4000-9760-8ec0f8ab3df8"   # keep stable so re-uploads update the same app
VERSION = "2.1.0"
BRAND = (91, 95, 199)                               # Teams purple


def font(size):
    for f in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def centered(draw, box, text, fnt, fill):
    l, t, r, b = draw.textbbox((0, 0), text, font=fnt)
    x = box[0] + (box[2] - box[0] - (r - l)) / 2 - l
    y = box[1] + (box[3] - box[1] - (b - t)) / 2 - t
    draw.text((x, y), text, font=fnt, fill=fill)


def make_icons():
    # Color icon: 192x192, full background
    color = Image.new("RGBA", (192, 192), BRAND + (255,))
    d = ImageDraw.Draw(color)
    centered(d, (0, 20, 192, 130), "VSL", font(64), "white")
    centered(d, (0, 120, 192, 170), "A B C", font(26), (220, 222, 255))
    color.save(BUILD / "color.png")

    # Outline icon: 32x32, white on transparent only
    outline = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    d = ImageDraw.Draw(outline)
    d.rounded_rectangle((1, 1, 30, 30), radius=6, outline="white", width=2)
    centered(d, (0, 0, 32, 32), "V", font(18), "white")
    outline.save(BUILD / "outline.png")


def manifest(base):
    host = urlparse(base).hostname
    return {
        "$schema": "https://developer.microsoft.com/en-us/json-schemas/teams/v1.17/MicrosoftTeams.schema.json",
        "manifestVersion": "1.17",
        "version": VERSION,
        "id": APP_ID,
        "developer": {
            "name": "VSL Research Demo",
            "websiteUrl": f"{base}/",
            "privacyUrl": f"{base}/privacy",
            "termsOfUseUrl": f"{base}/terms",
        },
        "name": {"short": "VSL Recognition", "full": "Vietnamese Sign Language Recognition"},
        "description": {
            "short": "Recognise Vietnamese sign-language letters from your webcam.",
            "full": ("Local research demo. Shows your webcam, finds 21 hand landmarks with MediaPipe "
                     "and classifies static Vietnamese sign-language letters with a small landmark "
                     "model trained on your own recordings, all on this computer."),
        },
        "icons": {"color": "color.png", "outline": "outline.png"},
        "accentColor": "#5B5FC7",
        "staticTabs": [
            {
                "entityId": "vsl-recognition",
                "name": "Nhận dạng VSL",
                "contentUrl": f"{base}/?inTeams=1",
                "websiteUrl": f"{base}/",
                "scopes": ["personal"],
            },
            {"entityId": "about", "scopes": ["personal"]},
        ],
        "configurableTabs": [
            {
                "configurationUrl": f"{base}/config?inTeams=1",
                "canUpdateConfiguration": True,
                "scopes": ["groupChat"],
                "context": ["meetingSidePanel", "meetingChatTab", "meetingDetailsTab"],
            },
        ],
        "permissions": ["identity"],
        "devicePermissions": ["media"],
        "validDomains": [host],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://localhost:53000",
                    help="HTTPS base URL where server.py is reachable")
    base = ap.parse_args().url.rstrip("/")
    if not base.startswith("https://"):
        raise SystemExit("Teams requires an https:// URL.")

    BUILD.mkdir(exist_ok=True)
    make_icons()
    (BUILD / "manifest.json").write_text(json.dumps(manifest(base), ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    out = BUILD / "vsl-teams-app.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("manifest.json", "color.png", "outline.png"):
            z.write(BUILD / name, name)
    print(f"Built {out}  (tab URL: {base}/?inTeams=1)")


if __name__ == "__main__":
    main()
