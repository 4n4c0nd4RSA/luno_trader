import os
from typing import Any, Dict, Optional, Tuple

import jwt
try:
    from jwt.exceptions import ExpiredSignatureError, InvalidSignatureError, InvalidTokenError
except Exception:
    class ExpiredSignatureError(Exception):
        pass

    class InvalidSignatureError(Exception):
        pass

    class InvalidTokenError(Exception):
        pass


PYJWT_INSTALL_MESSAGE = (
    "PyJWT with crypto support is required for license validation. "
    "Install it with: pip install \"PyJWT[crypto]\""
)


def load_public_key(public_key_path: str = "public_key.pem") -> str:
    """
    Load the RSA public key from a PEM file.
    """
    if not os.path.exists(public_key_path):
        raise FileNotFoundError(f"Public key file not found: {public_key_path}")

    with open(public_key_path, "r", encoding="utf-8") as f:
        return f.read()


def check_license_jwt(
    token: str,
    public_key_path: str = "public_key.pem",
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    algorithms: Optional[list[str]] = None,
    expected_key_id: Optional[str] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Check whether a JWT license is valid by verifying:
    - RSA signature using the public key (.pub/.pem)
    - expiration timeout via the exp claim

    Returns:
        (is_valid, payload, message)

    Examples:
        valid, payload, message = check_license_jwt(token)

        valid, payload, message = check_license_jwt(
            token,
            public_key_path="public_key.pem",
            issuer="MyCompany",
            audience="my-offline-app",
        )
    """
    try:
        if not hasattr(jwt, "decode"):
            return False, None, PYJWT_INSTALL_MESSAGE

        public_key = load_public_key(public_key_path)

        decode_kwargs: Dict[str, Any] = {
            "key": public_key,
            "algorithms": algorithms or ["RS256"],
            "options": {
                "require": ["exp"],
            },
        }

        if issuer is not None:
            decode_kwargs["issuer"] = issuer

        if audience is not None:
            decode_kwargs["audience"] = audience
        else:
            decode_kwargs["options"]["verify_aud"] = False

        payload = jwt.decode(token, **decode_kwargs)

        key_id = str(payload.get("key_id", "")).strip()
        if not key_id:
            return False, payload, "License token is missing required key_id claim."

        expected_key_id = str(expected_key_id or "").strip()
        if expected_key_id and key_id != expected_key_id:
            return (
                False,
                payload,
                "License key_id does not match the configured Luno API key id.",
            )

        return True, payload, "License is valid."

    except FileNotFoundError as e:
        return False, None, str(e)
    except ExpiredSignatureError:
        return False, None, "License has expired."
    except InvalidSignatureError:
        return False, None, "License signature is invalid."
    except InvalidTokenError as e:
        return False, None, f"License token is invalid: {e}"
    except Exception as e:
        return False, None, f"Unexpected error while validating license: {e}"


def check_license_file(
    license_file_path: str = "license.key",
    public_key_path: str = "public_key.pem",
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    algorithms: Optional[list[str]] = None,
    expected_key_id: Optional[str] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Load a JWT license from a file and validate it.
    """
    try:
        if not os.path.exists(license_file_path):
            return False, None, f"License file not found: {license_file_path}"

        with open(license_file_path, "r", encoding="utf-8") as f:
            token = f.read().strip()

        if not token:
            return False, None, "License file is empty."

        return check_license_jwt(
            token=token,
            public_key_path=public_key_path,
            issuer=issuer,
            audience=audience,
            algorithms=algorithms,
            expected_key_id=expected_key_id,
        )

    except Exception as e:
        return False, None, f"Unexpected error while reading license file: {e}"


if __name__ == "__main__":
    valid, payload, message = check_license_file(
        license_file_path="license.key",
        public_key_path="public_key.pem",
        issuer="QuantumMind Software",          # set to None if you do not want to enforce issuer
        audience="luno-bot",                    # set to None if you do not want to enforce audience
    )

    print(message)
    if valid:
        print("Payload:")
        print(payload)
