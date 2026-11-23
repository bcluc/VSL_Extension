"""
Create a self-signed HTTPS certificate for https://localhost (Teams tabs require HTTPS).

    python teams_app/make_cert.py

Writes teams_app/certs/localhost.crt + localhost.key. Uses the `openssl` that
ships with Git for Windows if it is not on PATH. Afterwards YOU must trust the
certificate once (the script prints the command) so Teams' web view accepts it.
"""

import shutil
import subprocess
import sys
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parent / "certs"
CANDIDATES = [
    r"C:\Program Files\Git\usr\bin\openssl.exe",
    r"C:\Program Files\Git\mingw64\bin\openssl.exe",
    r"C:\Program Files\Git\ucrt64\bin\openssl.exe",
]


def find_openssl():
    exe = shutil.which("openssl")
    if exe:
        return exe
    for c in CANDIDATES:
        if Path(c).exists():
            return c
    sys.exit("openssl not found. Install Git for Windows or OpenSSL.")


def main():
    CERT_DIR.mkdir(exist_ok=True)
    crt, key = CERT_DIR / "localhost.crt", CERT_DIR / "localhost.key"
    if crt.exists() and key.exists() and "--force" not in sys.argv:
        print(f"Certificate already exists: {crt}  (use --force to recreate)")
    else:
        subprocess.run([
            find_openssl(), "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes",
            "-days", "825", "-keyout", str(key), "-out", str(crt),
            "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-addext", "keyUsage=digitalSignature,keyEncipherment",
            "-addext", "extendedKeyUsage=serverAuth",
        ], check=True)
        print(f"Created {crt} and {key}")

    print("\nTrust it once for your Windows user (run in PowerShell, then restart Teams):")
    print(f'    certutil -user -addstore Root "{crt}"')
    print("Remove it later with:")
    print('    certutil -user -delstore Root localhost')


if __name__ == "__main__":
    main()
