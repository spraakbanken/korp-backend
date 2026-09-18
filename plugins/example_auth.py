"""Example authorization plugin using the generic protection helpers.

Enable by setting:

PLUGINS = ["plugins.example_auth"]

Example plugin config (YAML or PLUGINS_CONFIG):

plugins.example_auth:
  protected_corpora: ["CORPUS1", "CORPUS2"]
  protection_details:
    CORPUS1:
      license: "restricted"
  required_header: "X-Authorized-Corpora"

The header specified in `required_header` should contain a comma-separated list of corpora the caller may access (for
demo purposes only).
"""

from __future__ import annotations

from korp import auth, plugin
from korp.dependencies import AuthContext
from plugins import protection_cwb

router = plugin.Plugin("example_auth", __name__)


class ExampleAuth(auth.Authorizer):
    """Example Authorizer showing a non-CWB protection source.

    Protection metadata is loaded from plugin config, while corpus listing still comes from CWB. Authorization is based
    on a request header containing allowed corpora.
    """

    @classmethod
    def openapi_security(cls) -> tuple[dict[str, dict[str, str]], list[dict[str, list[str]]]]:
        """Document the configured corpus-access header consumed by this plugin.

        Returns:
            OpenAPI security schemes and operation requirements.
        """
        scheme_name = "corpusAccessHeader"
        return {
            scheme_name: {
                "type": "apiKey",
                "in": "header",
                "name": router.config("required_header", "X-Authorized-Corpora"),
            }
        }, [{scheme_name: []}]

    async def _fetch_protection_info(  # noqa: PLR6301
        self,
        corpora: list[str],
        auth_ctx: AuthContext,  # noqa: ARG002
    ) -> dict[str, auth.ProtectionInfo]:
        """Build protection metadata from plugin configuration.

        Returns:
            Protection metadata keyed by corpus.
        """
        configured_protected = {corpus.upper() for corpus in router.config("protected_corpora", [])}
        raw_details = router.config("protection_details", {}) or {}
        details_by_corpus = {
            corpus.upper(): details
            for corpus, details in raw_details.items()
            if isinstance(corpus, str) and isinstance(details, dict)
        }

        result: dict[str, auth.ProtectionInfo] = {}
        for corpus in corpora:
            corpus_upper = corpus.upper()
            result[corpus] = auth.ProtectionInfo(
                protected=corpus_upper in configured_protected,
                details=details_by_corpus.get(corpus_upper, {}),
            )
        return result

    async def get_protected_corpora(self, auth_ctx: AuthContext) -> list[str]:
        """Return all protected corpora.

        Returns:
            Uppercased corpus ids marked as protected.
        """
        corpora = protection_cwb.list_corpora(self.cwb)
        protection_info = await self._get_protection_info(corpora, auth_ctx)
        return [corpus.upper() for corpus in corpora if protection_info[corpus].protected]

    async def check_authorization(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> tuple[bool, list[str], str | None]:
        """Authorize requested corpora based on a request header.

        Returns:
            Authorization decision and list of unauthorized corpora.
        """
        protection_info = await self._get_protection_info(corpora, auth_ctx)
        protected_requested = [corpus for corpus in corpora if protection_info[corpus].protected]
        if not protected_requested:
            return True, [], None

        required_header = router.config("required_header", "X-Authorized-Corpora")
        allowed = {
            corpus.strip().upper()
            for corpus in auth_ctx.request.headers.get(required_header, "").split(",")
            if corpus.strip()
        }

        unauthorized = [corpus.upper() for corpus in protected_requested if corpus.upper() not in allowed]
        if unauthorized:
            return False, unauthorized, f"Missing access in {required_header} for: {', '.join(unauthorized)}"
        return True, [], None


AUTHORIZER_CLASS = ExampleAuth
