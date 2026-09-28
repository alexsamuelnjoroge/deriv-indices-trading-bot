"""
PreProposalPool — keeps hot DIGITOVER and DIGITUNDER proposals ready.

Fetches proposals before the signal fires, so buy() is the only roundtrip
needed (~1s vs ~4s for proposal+buy). Refreshes every 30s to prevent expiry.
"""
import asyncio
from loguru import logger


class PreProposalPool:
    """One pre-fetched proposal per contract type (DIGITOVER / DIGITUNDER)."""

    def __init__(
        self,
        client,
        symbol: str,
        barrier: str,
        duration: int,
        duration_unit: str,
        stake: float,
    ):
        self.client        = client
        self.symbol        = symbol
        self.barrier       = barrier
        self.duration      = duration
        self.duration_unit = duration_unit
        self.stake         = stake

        self._ids: dict[str, str | None] = {"DIGITOVER": None, "DIGITUNDER": None}
        self._lock          = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Pre-fetch both proposals, then launch background refresh loop."""
        await self._refresh_all()
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_all(self) -> None:
        await asyncio.gather(
            self._refresh_one("DIGITOVER"),
            self._refresh_one("DIGITUNDER"),
        )

    async def _refresh_one(self, contract_type: str) -> None:
        try:
            pid, _ = await self.client.propose_contract(
                symbol=self.symbol,
                contract_type=contract_type,
                duration=self.duration,
                duration_unit=self.duration_unit,
                stake=self.stake,
                barrier=self.barrier,
            )
            async with self._lock:
                self._ids[contract_type] = pid
            logger.debug(f"[{self.symbol}] Pre-proposal ready: {contract_type} → {pid}")
        except Exception as exc:
            logger.warning(f"[{self.symbol}] Pre-proposal failed ({contract_type}): {exc}")
            async with self._lock:
                self._ids[contract_type] = None

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            await self._refresh_all()

    async def consume(self, contract_type: str) -> str | None:
        """Take the cached proposal ID and immediately queue a replacement."""
        async with self._lock:
            pid = self._ids.get(contract_type)
            self._ids[contract_type] = None
        if pid:
            asyncio.create_task(self._refresh_one(contract_type))
        return pid

    def stop(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
