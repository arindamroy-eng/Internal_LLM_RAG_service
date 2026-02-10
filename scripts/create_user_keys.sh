#!/usr/bin/env bash
# ──────────────────────────────────────────────
# Create users and API keys for the LLM platform
#
# Usage:
#   ./create_user_keys.sh alice              # Single user
#   ./create_user_keys.sh --bulk users.txt   # Bulk from file (one username per line)
# ──────────────────────────────────────────────
set -euo pipefail

# Load env vars if .env exists
if [ -f .env ]; then
    set -a; source .env; set +a
fi

API_BASE="${APP_HOST:-http://localhost}:${APP_PORT:-8080}/api/v1"
ADMIN_KEY="${ADMIN_API_KEY:-sk-admin-change-me}"

create_user() {
    local username=$1
    echo -n "Creating user '${username}'... "

    response=$(curl -sf -X POST "${API_BASE}/admin/users" \
        -H "Authorization: Bearer ${ADMIN_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"username\": \"${username}\"}" 2>&1) || {
        echo "FAILED"
        echo "  Error: ${response}"
        return 1
    }

    api_key=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('api_key',''))" 2>/dev/null)

    if [ -n "$api_key" ]; then
        echo "OK"
        echo "  Username: ${username}"
        echo "  API Key:  ${api_key}"
        echo ""
        # Append to keys file
        echo "${username},${api_key}" >> user_keys.csv
    else
        echo "FAILED (unexpected response)"
        echo "  Response: ${response}"
    fi
}

# ── Main ──
if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  $0 <username>            Create a single user"
    echo "  $0 --bulk <file.txt>     Create users from file (one per line)"
    exit 1
fi

# Initialize output file
if [ ! -f user_keys.csv ]; then
    echo "username,api_key" > user_keys.csv
fi

if [ "$1" = "--bulk" ]; then
    if [ $# -lt 2 ] || [ ! -f "$2" ]; then
        echo "Error: Please provide a valid file with one username per line."
        exit 1
    fi

    echo "═══════════════════════════════════════════"
    echo "  Bulk User Creation"
    echo "═══════════════════════════════════════════"
    echo ""

    total=0
    success=0
    while IFS= read -r username; do
        # Skip empty lines and comments
        username=$(echo "$username" | tr -d '[:space:]')
        [[ -z "$username" || "$username" == \#* ]] && continue

        total=$((total + 1))
        if create_user "$username"; then
            success=$((success + 1))
        fi
    done < "$2"

    echo "═══════════════════════════════════════════"
    echo "  Done: ${success}/${total} users created"
    echo "  Keys saved to: user_keys.csv"
    echo "═══════════════════════════════════════════"
else
    create_user "$1"
    echo "Key saved to: user_keys.csv"
fi
