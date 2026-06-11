#!/usr/bin/env bash
# Map a custom domain to the Cloud Run dashboard (memorable URL).
#
# The default *.run.app URL hash (e.g. sfyons26pq) cannot be changed.
# Use a subdomain you control, e.g. twin-ledger.marketcrunch.ai
#
# Prerequisites:
#   1. You own the domain (marketcrunch.ai or other).
#   2. Domain verified in GCP: https://console.cloud.google.com/run/domains
#      (Search Console or add TXT record — one-time per root domain).
#   3. DNS access at your registrar (Cloudflare, Google Domains, etc.)
#
# Usage:
#   export GCP_PROJECT=turing-course-437219-c0
#   export GCP_REGION=us-central1
#   export CUSTOM_DOMAIN=twin-ledger.marketcrunch.ai
#   ./deploy/setup_custom_domain.sh
#
# After create, add the DNS records printed by gcloud to your DNS provider.
# HTTPS cert is provisioned automatically (can take 15–60 min).

set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE_NAME:-trading-agents}"
DOMAIN="${CUSTOM_DOMAIN:?Set CUSTOM_DOMAIN e.g. twin-ledger.marketcrunch.ai}"

echo "==> Cloud Run custom domain mapping"
echo "    Project:  ${PROJECT}"
echo "    Region:   ${REGION}"
echo "    Service:  ${SERVICE}"
echo "    Domain:   ${DOMAIN}"
echo "    Dashboard: https://${DOMAIN}/dashboard"
echo ""

SERVICE_URL="$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" --project="${PROJECT}" --format='value(status.url)')"
echo "    Current URL: ${SERVICE_URL}"
echo ""

set +e
gcloud beta run domain-mappings describe --domain="${DOMAIN}" \
  --region="${REGION}" --project="${PROJECT}" &>/dev/null
exists=$?
set -e

if [[ ${exists} -eq 0 ]]; then
  echo "==> Domain mapping already exists for ${DOMAIN}"
  gcloud beta run domain-mappings describe --domain="${DOMAIN}" \
    --region="${REGION}" --project="${PROJECT}"
else
  echo "==> Creating domain mapping..."
  gcloud beta run domain-mappings create \
    --service="${SERVICE}" \
    --domain="${DOMAIN}" \
    --region="${REGION}" \
    --project="${PROJECT}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Next: add DNS records at your registrar                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Run this to see required records:"
echo "  gcloud beta run domain-mappings describe --domain=${DOMAIN} --region=${REGION} --project=${PROJECT}"
echo ""
echo "Typical records:"
echo "  • CNAME  ${DOMAIN}  →  ghs.googlehosted.com"
echo "  • (or A/AAAA records if gcloud shows them for root domain)"
echo ""
echo "Verify domain ownership (if not done):"
echo "  https://console.cloud.google.com/run/domains?project=${PROJECT}"
echo ""
echo "Memorable dashboard URL (after DNS + cert):"
echo "  https://${DOMAIN}/dashboard"
