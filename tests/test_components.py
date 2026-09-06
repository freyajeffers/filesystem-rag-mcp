import hashlib
import secrets
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.fulltext import FullTextStore
from filesystem_rag_mcp.indexing import Chunk, FileMeta, chunk_file
from filesystem_rag_mcp.oauth import (
    MCPFileRAGAuthProvider,
    _b64url,
    _verify_pkce_s256,
)
from filesystem_rag_mcp.security import PathSecurityError, safe_resolve


def test_security_safe_resolve(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    doc = root / "doc.txt"
    doc.write_text("hello")

    resolved = safe_resolve(root, "doc.txt")
    assert resolved == doc.resolve()

    with pytest.raises(PathSecurityError):
        safe_resolve(root, "../outside.txt")


def test_indexing_chunking(tmp_path: Path):
    doc = tmp_path / "test.txt"
    doc.write_text("Hello world! " * 50)
    meta = FileMeta(
        abs_path=doc,
        rel_path="test.txt",
        size=len(doc.read_bytes()),
        mtime_ns=doc.stat().st_mtime_ns,
        sha256="dummy_sha256",
    )
    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / "data")
    chunks = chunk_file(meta, settings)
    assert len(chunks) > 0
    assert chunks[0].rel_path == "test.txt"


def test_fulltext_and_search(tmp_path: Path):
    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / "data")
    ft = FullTextStore(settings)
    chunks = [
        Chunk(
            chunk_id="c1",
            rel_path="doc1.txt",
            file_path=str(tmp_path / "doc1.txt"),
            start=0,
            end=12,
            text="Python async programming guide",
        ),
        Chunk(
            chunk_id="c2",
            rel_path="doc2.txt",
            file_path=str(tmp_path / "doc2.txt"),
            start=0,
            end=12,
            text="Database management with Postgres",
        ),
    ]
    ft.upsert(chunks)
    assert ft.count() == 2

    hits = ft.search("python", top_k=5)
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"


def test_pkce_verification():
    verifier = secrets.token_urlsafe(32)
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    assert _verify_pkce_s256(verifier, challenge)
    assert not _verify_pkce_s256("wrong", challenge)


@pytest.mark.asyncio
async def test_oauth_flow(tmp_path: Path):
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        oauth_issuer="http://127.0.0.1:8000",
        oauth_allow_dynamic_registration=True,
    )
    provider = MCPFileRAGAuthProvider(settings)

    # 1. DCR (Dynamic Client Registration)
    client_info = OAuthClientInformationFull(
        client_id="test-client-id",
        client_secret="test-client-secret",
        client_name="Test App",
        redirect_uris=[AnyUrl("http://localhost:8080/callback")],
    )
    await provider.register_client(client_info)
    retrieved = await provider.get_client("test-client-id")
    assert retrieved is not None
    assert retrieved.client_id == "test-client-id"

    # 2. Authorize
    auth_params = AuthorizationParams(
        redirect_uri=AnyUrl("http://localhost:8080/callback"),
        redirect_uri_provided_explicitly=True,
        state="xyz123",
        scopes=["fs.rag.read"],
        code_challenge="dummy_challenge",
    )
    redirect_url = await provider.authorize(retrieved, auth_params)
    assert "code=" in redirect_url
    assert "state=xyz123" in redirect_url

    # Extract code
    qs = parse_qs(urlparse(redirect_url).query)
    code = qs["code"][0]

    # 3. Load & Exchange Auth Code
    auth_code_obj = await provider.load_authorization_code(retrieved, code)
    assert auth_code_obj is not None
    assert auth_code_obj.client_id == "test-client-id"

    tokens = await provider.exchange_authorization_code(retrieved, auth_code_obj)
    assert tokens.access_token is not None
    assert tokens.refresh_token is not None

    # 4. Verify Access Token JWT
    access_obj = await provider.load_access_token(tokens.access_token)
    assert access_obj is not None
    assert access_obj.client_id == "test-client-id"
    assert "fs.rag.read" in access_obj.scopes
