from pathlib import Path
import jwt
from datetime import datetime, timedelta, timezone


PRIVATE_KEY_PATH = Path(__file__).with_name("private_key.pem")
PYJWT_INSTALL_MESSAGE = (
    "PyJWT with crypto support is required to generate licenses. "
    "Install it with: pip install \"PyJWT[crypto]\""
)


def load_private_key(private_key_path: Path = PRIVATE_KEY_PATH) -> str:
    with private_key_path.open("r", encoding="utf-8") as f:
        return f.read()

def generate_license_jwt(
    license_id: str,
    customer: str,
    key_id: str,
    audience: str = "luno-bot",
    issuer: str = "QuantumMind Software",
    days_valid: int = 365,
    license_type: str = "pro",
    features: list[str] | None = None,
    seats: int = 1,
    machine_id: str | None = None,
) -> str:
    if not hasattr(jwt, "encode"):
        raise RuntimeError(PYJWT_INSTALL_MESSAGE)

    key_id = str(key_id).strip()
    if not key_id:
        raise ValueError("Luno API key_id is required.")

    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=days_valid)

    payload = {
        "iss": issuer,
        "aud": audience,
        "sub": license_id,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "valid_until": exp.strftime("%Y-%m-%d"),
        "customer": customer,
        "key_id": key_id,
        "license_type": license_type,
        "features": features or [],
        "seats": seats,
    }

    if machine_id:
        payload["machine_id"] = machine_id

    token = jwt.encode(payload, load_private_key(), algorithm="RS256")
    return token

if __name__ == "__main__":
    customer = input("Customer name: ").strip()
    if not customer:
        raise ValueError("Customer name is required.")

    key_id = input("Luno API Key ID: ").strip()
    if not key_id:
        raise ValueError("Luno API Key ID is required.")

    days_valid_text = input("Days valid: ").strip()
    if not days_valid_text:
        raise ValueError("Days valid is required.")

    try:
        days_valid = int(days_valid_text)
    except ValueError as exc:
        raise ValueError("Days valid must be a whole number.") from exc

    # if days_valid <= 0:
    #     raise ValueError("Days valid must be greater than 0.")

    token = generate_license_jwt(
        license_id="LIC-000001",
        customer=customer,
        key_id=key_id,
        audience="luno-bot",
        days_valid=days_valid,
        license_type="pro",
        features=["sync", "reports"],
        seats=1,
        machine_id="ABC123XYZ",
    )

    print("JWT License Key:")
    print(token)

    with open("license.key", "w", encoding="utf-8") as f:
        f.write(token)
