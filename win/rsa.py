from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Export private key (KEEP THIS SECRET)
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)

# Export public key (SAFE TO SHIP)
public_key = private_key.public_key()
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)

# Save files
with open("private_key.pem", "wb") as f:
    f.write(private_pem)

with open("public_key.pem", "wb") as f:
    f.write(public_pem)

print("Keys generated:")
print("- private_key.pem (KEEP SECRET)")
print("- public_key.pem (EMBED IN APP)")