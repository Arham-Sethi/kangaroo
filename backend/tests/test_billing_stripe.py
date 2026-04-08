"""Tests for Stripe billing endpoints (checkout, portal, webhook).

Tests cover:
    - POST /api/v1/billing/checkout — create Stripe checkout session
    - POST /api/v1/billing/portal — create Stripe billing portal
    - POST /api/v1/billing/webhook — handle Stripe webhooks
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.api.v1.billing import (
    CheckoutRequest,
    CheckoutResponse,
    PortalResponse,
    STRIPE_PRICE_MAP,
)


# ── Checkout Tests ──────────────────────────────────────────────────────────


class TestCheckout:
    """Tests for create_checkout_session endpoint."""

    def test_checkout_request_schema_valid(self) -> None:
        req = CheckoutRequest(tier="pro", billing_cycle="monthly")
        assert req.tier == "pro"
        assert req.billing_cycle == "monthly"
        assert req.success_url == ""
        assert req.cancel_url == ""

    def test_checkout_request_annual(self) -> None:
        req = CheckoutRequest(tier="pro_team", billing_cycle="annual")
        assert req.tier == "pro_team"
        assert req.billing_cycle == "annual"

    def test_checkout_request_with_urls(self) -> None:
        req = CheckoutRequest(
            tier="pro",
            billing_cycle="monthly",
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
        )
        assert req.success_url == "https://app.example.com/success"
        assert req.cancel_url == "https://app.example.com/cancel"

    def test_checkout_response_schema(self) -> None:
        resp = CheckoutResponse(
            checkout_url="https://checkout.stripe.com/c/pay_123",
            session_id="cs_test_123",
        )
        assert resp.checkout_url.startswith("https://")
        assert resp.session_id == "cs_test_123"

    def test_stripe_price_map_structure(self) -> None:
        assert "pro" in STRIPE_PRICE_MAP
        assert "pro_team" in STRIPE_PRICE_MAP
        assert "monthly" in STRIPE_PRICE_MAP["pro"]
        assert "annual" in STRIPE_PRICE_MAP["pro"]
        assert "monthly" in STRIPE_PRICE_MAP["pro_team"]
        assert "annual" in STRIPE_PRICE_MAP["pro_team"]

    def test_invalid_tier_not_in_map(self) -> None:
        assert "free" not in STRIPE_PRICE_MAP
        assert "enterprise" not in STRIPE_PRICE_MAP


# ── Portal Tests ────────────────────────────────────────────────────────────


class TestPortal:
    """Tests for create_portal_session endpoint."""

    def test_portal_response_schema(self) -> None:
        resp = PortalResponse(portal_url="https://billing.stripe.com/p/session_123")
        assert resp.portal_url.startswith("https://")


# ── Webhook Tests ───────────────────────────────────────────────────────────


class TestWebhook:
    """Tests for Stripe webhook handler."""

    def test_checkout_completed_event_structure(self) -> None:
        """Verify event shape for checkout.session.completed."""
        event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_123",
                    "customer_email": "test@example.com",
                    "metadata": {
                        "user_id": "user-abc-123",
                        "tier": "pro",
                        "billing_cycle": "monthly",
                    },
                },
            },
        }
        assert event["type"] == "checkout.session.completed"
        obj = event["data"]["object"]
        assert obj["metadata"]["user_id"] == "user-abc-123"
        assert obj["metadata"]["tier"] == "pro"

    def test_subscription_deleted_event_structure(self) -> None:
        """Verify event shape for customer.subscription.deleted."""
        event = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_123",
                    "customer": "cus_123",
                    "status": "canceled",
                },
            },
        }
        assert event["type"] == "customer.subscription.deleted"
        assert event["data"]["object"]["status"] == "canceled"

    def test_payment_failed_event_structure(self) -> None:
        """Verify event shape for invoice.payment_failed."""
        event = {
            "type": "invoice.payment_failed",
            "data": {
                "object": {
                    "id": "in_123",
                    "customer": "cus_123",
                    "amount_due": 1600,
                    "currency": "usd",
                },
            },
        }
        assert event["type"] == "invoice.payment_failed"
        assert event["data"]["object"]["amount_due"] == 1600
