"""The subscription key (Dodo Payments public licence endpoints), with a fake Dodo."""

import datetime as dt

import pytest

from streamsafe.app import license as lic
from streamsafe.app import plan, release

KEY = "2b1f8e2d-c41e-4e8f-b2d3-d9fd61c38f43"


class FakeDodo:
    def __init__(self):
        self.valid, self.down, self.calls = True, False, []
        self.activate_code = 201
        self.product = "pdt_pro"

    def __call__(self, url, body):
        self.calls.append((url, dict(body)))
        if self.down:
            return 0, {}
        if url.endswith("/licenses/activate"):
            if self.activate_code != 201:
                return self.activate_code, {"message": "no"}
            return 201, {"id": "lki_1", "product": {"product_id": self.product}, "customer": {"email": "a@b.c"}}
        if url.endswith("/licenses/validate"):
            return 200, {"valid": self.valid}
        if url.endswith("/licenses/deactivate"):
            return 200, {}
        return 404, {}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "PRODUCT_PLANS", {"pdt_basic": "basic", "pdt_pro": "pro"})
    monkeypatch.setattr(release, "DODO_MODE", "test")
    monkeypatch.delenv("STREAMSAFE_PLAN", raising=False)
    return {"STREAMSAFE_DATA_DIR": str(tmp_path)}


def test_activate_then_check_and_the_plan_follows_the_product(env):
    dodo = FakeDodo()
    st = lic.activate(f"  {KEY}\n", env, dodo)
    assert st.ok and st.plan == "pro" and plan.allows("custom_brand", env)
    url, body = dodo.calls[0]
    assert url == "https://test.dodopayments.com/licenses/activate" and body["license_key"] == KEY and body["name"]
    assert lic.check(env, dodo).ok
    assert dodo.calls[-1][1] == {"license_key": KEY, "license_key_instance_id": "lki_1"}


def test_a_lapsed_subscription_stops_at_once_and_nothing_is_kept(env):
    dodo = FakeDodo()
    lic.activate(KEY, env, dodo)
    dodo.valid = False  # cancelled / not paid: Dodo switches the key off
    st = lic.check(env, dodo)
    assert not st.ok and st.reason == "invalid" and lic.load(env) is None
    assert not plan.allows("custom_brand", env)
    dodo.down = True
    assert lic.check(env, dodo).reason == "no_license"  # no grace for a key Dodo said no to


def test_no_internet_is_forgiven_for_the_grace_days_only(env):
    dodo = FakeDodo()
    t0 = dt.datetime(2026, 11, 17, tzinfo=dt.UTC)
    lic.activate(KEY, env, dodo, now=lambda: t0)
    dodo.down = True
    st = lic.check(env, dodo, now=lambda: t0 + dt.timedelta(days=2))
    assert st.ok and st.grace
    st = lic.check(env, dodo, now=lambda: t0 + dt.timedelta(days=release.LICENSE_GRACE_DAYS + 1))
    assert not st.ok and st.reason == "offline_too_long"


@pytest.mark.parametrize("code,reason", [(404, "not_found"), (422, "limit"), (403, "inactive"), (400, "invalid")])
def test_activation_refusals_say_why(env, code, reason):
    dodo = FakeDodo()
    dodo.activate_code = code
    st = lic.activate(KEY, env, dodo)
    assert not st.ok and st.reason == reason and lic.load(env) is None


def test_a_key_of_another_product_is_refused(env):
    dodo = FakeDodo()
    dodo.product = "pdt_something_else"
    assert lic.activate(KEY, env, dodo).reason == "unknown_product"


def test_deactivate_frees_this_pc(env):
    dodo = FakeDodo()
    lic.activate(KEY, env, dodo)
    assert lic.deactivate(env, dodo) and lic.load(env) is None
    assert dodo.calls[-1][0].endswith("/licenses/deactivate")


def test_controller_refuses_to_start_without_a_valid_subscription(env, monkeypatch):
    from streamsafe.app import controller
    from tests.appfix import FakeProtector

    monkeypatch.setattr(release, "LICENSE_REQUIRED", True)
    c = controller.Controller(env, FakeProtector(), log=lambda line: None)
    assert "why_license_no_license" in c.blockers("k")
    dodo = FakeDodo()
    assert c.activate_license(KEY, dodo).ok and not any(b.startswith("why_license") for b in c.blockers("k"))
    dodo.valid = False
    assert not c.check_license(dodo).ok and "why_license_invalid" in c.blockers("k")
