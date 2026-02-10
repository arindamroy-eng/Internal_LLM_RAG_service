#!/usr/bin/env bash
# ──────────────────────────────────────────────
# Setup ROCm 7.x on Ubuntu for MI350X GPUs
# Run as root or with sudo
# ──────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════"
echo "  ROCm Setup for AMD Instinct MI350X"
echo "═══════════════════════════════════════════"

# ── 1. Check OS ──
if ! grep -q "Ubuntu" /etc/os-release; then
    echo "WARNING: This script targets Ubuntu 22.04/24.04. You may need to adapt."
fi

# ── 2. Install prerequisites ──
echo "[1/5] Installing prerequisites..."
apt-get update
apt-get install -y \
    wget \
    gnupg2 \
    curl \
    software-properties-common \
    linux-headers-$(uname -r)

# ── 3. Add AMD ROCm repository ──
echo "[2/5] Adding AMD ROCm repository..."
# Import AMD GPG key
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

# Add repo (adjust version as needed)
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest jammy main" \
    > /etc/apt/sources.list.d/rocm.list

# Pin priority
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    > /etc/apt/preferences.d/rocm-pin-600

apt-get update

# ── 4. Install AMDGPU driver + ROCm ──
echo "[3/5] Installing AMDGPU driver and ROCm..."
apt-get install -y amdgpu-dkms
apt-get install -y rocm-dev rocm-libs rocm-utils

# ── 5. Configure user permissions ──
echo "[4/5] Configuring GPU access permissions..."
usermod -aG video $SUDO_USER 2>/dev/null || true
usermod -aG render $SUDO_USER 2>/dev/null || true

# ── 6. Verify installation ──
echo "[5/5] Verifying installation..."
echo ""

if command -v rocm-smi &>/dev/null; then
    echo "ROCm installation successful!"
    echo ""
    rocm-smi
    echo ""
    echo "GPU count: $(rocm-smi --showid | grep -c GPU)"
else
    echo "WARNING: rocm-smi not found. You may need to reboot first."
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Setup complete."
echo "  Please reboot before starting vLLM."
echo "  After reboot, run: ./scripts/start_vllm.sh"
echo "═══════════════════════════════════════════"
