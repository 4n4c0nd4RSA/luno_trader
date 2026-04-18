import jwt
from datetime import datetime, timedelta, timezone

PRIVATE_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCSnbjcwVhgyKdx
6szp3UCfNXTYzTgOWEbf5UJ1PdoIT3sDyeG0lDmU2gP7d7Uyo1VG7OXduLNr26Uu
APOl05zKSOy9MjifHBvJsZh/hKtm1BNqy3mTRieyL3uuL6QjdWZY7YYKxUf6m3eA
bDeUELNIB8S+zlUpIFv2WPpL8p9zyMOz+ylJ2Epyh2YFwNUFmgfKY/Rg6orENwx2
5dwtvI3owvaaRyZqF0/zEiYZHouV7gtgRxG2qAd54KPvaAd9vitEjsZaulFXQUc2
6wZX4NLW3RYNMDwYwBY8OOhh8Bp5Ak29Rk0Cy5xyFRpuEGgwf2yWHfbQRrRpZX72
CFVtRaKbAgMBAAECggEAGyAdHJKU5EsbV6MmSDpMA2ijdpz5OwYwDqpe0kwgM5to
2fmWzY8C3Dw9sl+iYX0AgP/BQ3UxlMntVIaOjWaKQo6dZh+NhDpyFa3K1gTYpUyl
TwPYOVwoafoI+uALkRfeWKORQhrfx0jOIi9jVPyF/tSZHOgVr2cyPHCX1kdMIuKB
RbAoA+Hs7mW6xWhth6Dk0EEdAl/MeJmXuNudCTQNXcIYg/QOB/erp4PrmRdcCiu+
kEo8JKIFwdQbc8+AVsIoYuysvz37k8Ub/8gsnHZ9JKDoZvC0v48lofB9MipR6sN3
QhRX1Cc9LzH3wW+P8WID+HE+KUan0xiZpQM9mUN9gQKBgQDGTWHDzJk+n9X9ZUej
aLgb0KB68ZTmZ/YoCp/08M0pMS0vASS8nRXyx35vQTXtiywDTDSykKIlDdH56CmF
bwPwkT5mI9dxFQuqqSbG9yZixU147YG4UXH55gbwTAP5S8MaNcOjWGTU6lfpm/Ww
Fg7K0yBL/hN8xxhjZ/SMxWwqWwKBgQC9RnoHN5LDxvtzKNc2jVrvOVZrBnfkKBof
C5vbsV+pl9IdspXMYPlu7DQcQDWzGhyn5/hZz0O3pkURRkouO1nHvHnUEL1yHX/0
6miWA4gXMya+K2BROjpCkj04PEZINrKV6AHPQhQn0b0AcP9DhCfYtc/GbS0icP4Q
hVa4tFNcwQKBgHwcCDkcPzkDnlFmZuyt6LR34UixFCkUHeq3o58QascCIS5O/+gQ
RKFbHBWTcaYHOXei5URw4xpfyPAoznvVnFie/re8bOU4b8HS9hMGsf6VT8SEmXB7
gOMPhX41hTMsqKIpzhTYiNr9BCKQWrdnRsDIXGlTTm0Eyo3EjTjhgq4LAoGAI3ij
wdp+XMZbtVdADe8rzY1XTrNloJKLYqoQSXnSTbwwGnSvch/yXwFROsIlzizkler5
NguLy00TwIsFt+hTiQUfZ8jDWDGDG2katJJw3LgvWJBUeqSI6pTxeCqDmWD20vUp
8aeWk2fRHdYPYJ3RweFA0RUA0mWOl5YFjJPu04ECgYEAxAvfjso2QDTLm78cu6tD
L3/35LWYyoDAK8bt3TryZkI37HYb7mEhuXuIf5pwDg1TTuUXB1+fnVcUW/EuKBQ+
jqryv4teecvUa+jGVfMvg8zvqquks7r21sS4CIzIft9g8rxmdrk2M7K/4n7Ys4i9
FRSngIFRN3w/wWBJy1vGqtg=
-----END PRIVATE KEY-----
"""

def generate_license_jwt(
    license_id: str,
    customer: str,
    audience: str = "luno-bot",
    issuer: str = "QuantumMind Software",
    days_valid: int = 365,
    license_type: str = "pro",
    features: list[str] | None = None,
    seats: int = 1,
    machine_id: str | None = None,
) -> str:
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
        "license_type": license_type,
        "features": features or [],
        "seats": seats,
    }

    if machine_id:
        payload["machine_id"] = machine_id

    token = jwt.encode(payload, PRIVATE_KEY_PEM, algorithm="RS256")
    return token

if __name__ == "__main__":
    token = generate_license_jwt(
        license_id="LIC-000001",
        customer="ACME Pty Ltd",
        audience="luno-bot",
        days_valid=365,
        license_type="pro",
        features=["sync", "reports"],
        seats=1,
        machine_id="ABC123XYZ",
    )

    print("JWT License Key:")
    print(token)

    with open("license.key", "w", encoding="utf-8") as f:
        f.write(token)