"""Dry-run-only Freqtrade adapter for the registered 4h lead-lag candidate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from math import isfinite, log
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from vnqlib_research.perpetual_event import (
    ALT_TARGET_PAIRS,
    BTC_REFERENCE_PAIR,
    BtcImpulseAltLag4hSpecV1,
    SignalIntentV1,
    build_signal_frames_v1,
    build_signal_intents_v1,
)


logger = logging.getLogger(__name__)

_TAG_PATTERN = re.compile(
    r"^BIAL4V1\|t=(\d{8}T\d{6}Z)\|d=([LS])\|"
    r"g=([+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\|"
    r"s=((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
_FOUR_HOURS = pd.Timedelta(hours=4)
_TIMEOUT = timedelta(hours=42 * 4)
_ENTRY_VALIDITY = timedelta(seconds=60)
_RISK_SCHEMA = "lead-lag-risk-state-v1"
_RISK_REDUCTION_DRAWDOWN = 0.15
_RISK_HALT_DRAWDOWN = 0.18
_NORMAL_RISK = 0.0075
_REDUCED_RISK = 0.00375
_AUDIT_FIELDS = (
    "event",
    "recorded_at",
    "pair",
    "signal_time",
    "order_time",
    "fill_price",
    "latency_ms",
    "slippage_bps",
    "fee",
    "funding",
    "rejection_reason",
    "exit_reason",
    "side",
    "entry_tag",
    "initial_gap_return",
    "stop_distance",
    "rank",
)


@dataclass(frozen=True, slots=True)
class _EntryMetadata:
    signal_at: pd.Timestamp
    direction: str
    initial_gap_return: float
    stop_distance: float


@dataclass(frozen=True, slots=True)
class _RiskState:
    peak_equity: float
    halted: bool


class BtcImpulseAltLag4hV1(IStrategy):
    """Execution adapter; all entry selection comes from the research core."""

    INTERFACE_VERSION = 3
    timeframe = "4h"
    can_short = True
    startup_candle_count = 100
    process_only_new_candles = True
    position_adjustment_enable = False

    minimal_roi = {"0": 1000.0}
    stoploss = -0.06
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    _spec = BtcImpulseAltLag4hSpecV1()

    def bot_start(self, **kwargs: Any) -> None:
        del kwargs
        if self.config.get("dry_run") is not True:
            raise RuntimeError("BtcImpulseAltLag4hV1 is dry-run only")

    def informative_pairs(self) -> list[tuple[str, str, str]]:
        return [(BTC_REFERENCE_PAIR, self.timeframe, "futures")]

    @staticmethod
    def _to_core_frame(dataframe: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(dataframe.columns):
            raise ValueError("Freqtrade frame is missing date/OHLCV columns")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        mask = dates <= cutoff
        result = dataframe.loc[mask, ["open", "high", "low", "close", "volume"]].copy()
        result.index = pd.DatetimeIndex(dates.loc[mask])
        return result

    def _closed_panel(
        self, dataframe: pd.DataFrame, current_pair: str
    ) -> dict[str, pd.DataFrame]:
        if current_pair not in ALT_TARGET_PAIRS:
            raise ValueError("adapter can only analyze a fixed target pair")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        if dates.empty:
            raise ValueError("Freqtrade frame cannot be empty")
        cutoff = pd.Timestamp(dates.max())
        panel: dict[str, pd.DataFrame] = {}
        for pair in (BTC_REFERENCE_PAIR, *ALT_TARGET_PAIRS):
            raw = (
                dataframe
                if pair == current_pair
                else self.dp.get_pair_dataframe(
                    pair, self.timeframe, candle_type="futures"
                )
            )
            panel[pair] = self._to_core_frame(raw, cutoff)
        return panel

    @staticmethod
    def _entry_tag(intent: Any) -> str:
        direction = "L" if intent.side == "LONG" else "S"
        stamp = intent.signal_at.strftime("%Y%m%dT%H%M%SZ")
        gap = format(float(intent.initial_gap_return), ".17g")
        if not gap.startswith(("+", "-")):
            gap = f"+{gap}"
        stop = format(float(intent.stop_distance), ".17g")
        return f"BIAL4V1|t={stamp}|d={direction}|g={gap}|s={stop}"

    @staticmethod
    def _parse_entry_tag(value: str | None) -> _EntryMetadata | None:
        if not isinstance(value, str) or (match := _TAG_PATTERN.fullmatch(value)) is None:
            return None
        try:
            signal_at = pd.Timestamp(
                datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc
                )
            )
            gap = float(match.group(3))
            stop = float(match.group(4))
        except (TypeError, ValueError, OverflowError):
            return None
        direction = match.group(2)
        if (
            not isfinite(gap)
            or not isfinite(stop)
            or not 0.03 <= stop <= 0.06
            or (direction == "L" and gap <= 0.0)
            or (direction == "S" and gap >= 0.0)
        ):
            return None
        return _EntryMetadata(signal_at, direction, gap, stop)

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any]
    ) -> pd.DataFrame:
        result = dataframe.copy(deep=True)
        result["lead_lag_side"] = 0
        result["lead_lag_entry_tag"] = None
        result["lead_lag_signal_at"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        result["lead_lag_btc_z"] = float("nan")
        if result.empty:
            return result

        pair = metadata.get("pair")
        panel = self._closed_panel(result, pair)
        frames = build_signal_frames_v1(panel, self._spec)
        intents = build_signal_intents_v1(panel, self._spec)
        dates = pd.to_datetime(result["date"], utc=True, errors="raise")
        result["lead_lag_btc_z"] = dates.map(
            frames[BTC_REFERENCE_PAIR]["z_return"]
        )
        latest_signal_at = pd.Timestamp(dates.max())
        for position, intent in enumerate(intents):
            if intent.pair != pair:
                continue
            mask = dates == intent.signal_at
            if intent.signal_at == latest_signal_at:
                rank = sum(
                    previous.signal_at == intent.signal_at
                    for previous in intents[: position + 1]
                )
                if not self._audit_signal_once(intent, rank):
                    continue
            result.loc[mask, "lead_lag_side"] = 1 if intent.side == "LONG" else -1
            result.loc[mask, "lead_lag_entry_tag"] = self._entry_tag(intent)
            result.loc[mask, "lead_lag_signal_at"] = intent.signal_at
        return result

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any]
    ) -> pd.DataFrame:
        del metadata
        result = dataframe.copy(deep=True)
        result["enter_long"] = 0
        result["enter_short"] = 0
        result["enter_tag"] = None
        result.loc[result["lead_lag_side"] == 1, "enter_long"] = 1
        result.loc[result["lead_lag_side"] == -1, "enter_short"] = 1
        selected = result["lead_lag_side"].isin((1, -1))
        result.loc[selected, "enter_tag"] = result.loc[
            selected, "lead_lag_entry_tag"
        ]
        return result

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any]
    ) -> pd.DataFrame:
        del metadata
        result = dataframe.copy(deep=True)
        result["exit_long"] = 0
        result["exit_short"] = 0
        return result

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        del pair, current_time, current_rate, proposed_leverage, max_leverage
        del entry_tag, side, kwargs
        return 1.0

    def _persistent_runtime(self) -> bool:
        return str(self.config.get("runmode", "")).lower() == "dry_run"

    def _risk_path(self) -> Path:
        configured = os.environ.get("LEAD_LAG_RISK_STATE_PATH") or self.config.get(
            "lead_lag_risk_state_path",
            "/freqtrade/user_data/data/lead_lag_risk_state.json",
        )
        return Path(str(configured))

    def _audit_path(self) -> Path:
        configured = os.environ.get("LEAD_LAG_AUDIT_PATH") or self.config.get(
            "lead_lag_audit_path",
            "/freqtrade/user_data/logs/lead_lag_execution.jsonl",
        )
        return Path(str(configured))

    def _initial_equity(self, equity: float) -> float:
        wallet = self.config.get("dry_run_wallet", equity)
        try:
            configured = float(wallet)
        except (TypeError, ValueError, OverflowError):
            configured = equity
        return max(equity, configured) if isfinite(configured) else equity

    def _load_risk_state(self, equity: float) -> _RiskState:
        existing = getattr(self, "_risk_state", None)
        if isinstance(existing, _RiskState):
            return existing
        initial = _RiskState(self._initial_equity(equity), False)
        self._risk_state_error = False
        if not self._persistent_runtime() or not self._risk_path().exists():
            self._risk_state = initial
            return initial
        try:
            payload = json.loads(self._risk_path().read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version",
                "peak_equity",
                "halted",
            }:
                raise ValueError("risk state fields differ from v1 schema")
            peak = float(payload["peak_equity"])
            halted = payload["halted"]
            if (
                payload["schema_version"] != _RISK_SCHEMA
                or not isfinite(peak)
                or peak <= 0.0
                or type(halted) is not bool
            ):
                raise ValueError("risk state values are invalid")
            state = _RiskState(peak, halted)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.exception("risk state is invalid; failing closed")
            self._risk_state_error = True
            state = _RiskState(initial.peak_equity, True)
        self._risk_state = state
        return state

    def _persist_risk_state(self, state: _RiskState) -> bool:
        if not self._persistent_runtime() or getattr(self, "_risk_state_error", False):
            return not getattr(self, "_risk_state_error", False)
        path = self._risk_path()
        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    {
                        "halted": state.halted,
                        "peak_equity": state.peak_equity,
                        "schema_version": _RISK_SCHEMA,
                    },
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return True
        except OSError:
            logger.exception("cannot persist risk state; failing closed")
            self._risk_state_error = True
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            return False

    def _update_risk(self, equity: float) -> tuple[_RiskState, float]:
        if not isfinite(equity) or equity <= 0.0:
            state = _RiskState(max(self._initial_equity(1.0), 1.0), True)
            self._risk_state = state
            self._risk_state_error = True
            return state, 1.0
        state = self._load_risk_state(equity)
        if state.halted or getattr(self, "_risk_state_error", False):
            drawdown = max(0.0, 1.0 - equity / state.peak_equity)
            return _RiskState(state.peak_equity, True), drawdown
        peak = max(state.peak_equity, equity)
        drawdown = max(0.0, 1.0 - equity / peak)
        updated = _RiskState(peak, drawdown >= _RISK_HALT_DRAWDOWN)
        path_missing = self._persistent_runtime() and not self._risk_path().exists()
        if updated != state or path_missing:
            if not self._persist_risk_state(updated):
                updated = _RiskState(updated.peak_equity, True)
            self._risk_state = updated
        return updated, drawdown

    def _latest_closed_rate(self, pair: str, current_time: datetime) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        eligible = dataframe.loc[dates < pd.Timestamp(current_time)]
        if eligible.empty:
            raise ValueError(f"no closed mark available for {pair}")
        rate = float(eligible.iloc[-1]["close"])
        if not isfinite(rate) or rate <= 0.0:
            raise ValueError(f"invalid closed mark for {pair}")
        return rate

    def _current_equity(self, current_time: datetime) -> float:
        try:
            equity = float(self.wallets.get_total_stake_amount())
            for trade in Trade.get_open_trades():
                rate = self._latest_closed_rate(trade.pair, current_time)
                result = trade.calc_profit(rate)
                profit_abs = (
                    result.profit_abs
                    if hasattr(result, "profit_abs")
                    else result["profit_abs"]
                )
                equity += float(profit_abs)
            return equity
        except (
            ArithmeticError,
            AttributeError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ):
            logger.exception("cannot mark account equity; failing closed")
            return float("nan")

    @staticmethod
    def _iso(value: datetime | pd.Timestamp | None) -> str | None:
        if value is None:
            return None
        stamp = pd.Timestamp(value)
        if stamp.tz is None:
            raise ValueError("audit timestamps must be timezone-aware")
        return stamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    def _append_audit(self, **values: Any) -> bool:
        if not self._persistent_runtime():
            return True
        record = {field: values.get(field) for field in _AUDIT_FIELDS}
        missing = set(values) - set(_AUDIT_FIELDS)
        if missing:
            raise ValueError(f"unknown audit fields: {sorted(missing)}")
        path = self._audit_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            return True
        except (OSError, TypeError, ValueError):
            logger.exception("cannot append execution audit")
            return False

    def _load_signal_audit_keys(self) -> frozenset[tuple[str, str]]:
        cached = getattr(self, "_signal_audit_keys", None)
        if isinstance(cached, frozenset):
            return cached
        keys: frozenset[tuple[str, str]] = frozenset()
        path = self._audit_path()
        if self._persistent_runtime() and path.exists():
            try:
                records = tuple(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                keys = frozenset(
                    (str(record["pair"]), str(record["signal_time"]))
                    for record in records
                    if isinstance(record, dict)
                    and record.get("event") == "SIGNAL_INTENT"
                    and record.get("pair") is not None
                    and record.get("signal_time") is not None
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                logger.exception("cannot read signal audit; failing closed")
                self._signal_audit_error = True
        self._signal_audit_keys = keys
        return keys

    def _audit_signal_once(self, intent: SignalIntentV1, rank: int) -> bool:
        if not self._persistent_runtime():
            return True
        keys = self._load_signal_audit_keys()
        if getattr(self, "_signal_audit_error", False):
            return False
        signal_time = self._iso(intent.signal_at)
        assert signal_time is not None
        key = (intent.pair, signal_time)
        if key in keys:
            return True
        metadata = _EntryMetadata(
            intent.signal_at,
            "L" if intent.side == "LONG" else "S",
            intent.initial_gap_return,
            intent.stop_distance,
        )
        appended = self._append_audit(
            **self._audit_values(
                event="SIGNAL_INTENT",
                pair=intent.pair,
                recorded_at=datetime.now(timezone.utc),
                metadata=metadata,
                side=intent.side,
                entry_tag=self._entry_tag(intent),
                initial_gap_return=float(intent.initial_gap_return),
                stop_distance=float(intent.stop_distance),
                rank=rank,
            )
        )
        if appended:
            self._signal_audit_keys = keys | {key}
        return appended

    def _audit_values(
        self,
        *,
        event: str,
        pair: str,
        recorded_at: datetime,
        metadata: _EntryMetadata | None = None,
        **values: Any,
    ) -> dict[str, Any]:
        result = {
            "event": event,
            "recorded_at": self._iso(recorded_at),
            "pair": pair,
            "signal_time": self._iso(metadata.signal_at) if metadata else None,
            "order_time": None,
            "fill_price": None,
            "latency_ms": None,
            "slippage_bps": None,
            "fee": None,
            "funding": None,
            "rejection_reason": None,
            "exit_reason": None,
            "side": None,
            "entry_tag": None,
            "initial_gap_return": None,
            "stop_distance": None,
            "rank": None,
        }
        return result | values

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        del current_rate, proposed_stake, kwargs
        metadata = self._parse_entry_tag(entry_tag)
        expected_direction = "S" if side == "short" else "L"
        reason: str | None = None
        if pair not in ALT_TARGET_PAIRS or metadata is None:
            reason = "INVALID_ENTRY_TAG"
        elif metadata.direction != expected_direction:
            reason = "ENTRY_SIDE_MISMATCH"

        equity = self._current_equity(current_time)
        state, drawdown = self._update_risk(equity)
        if state.halted:
            reason = reason or "DRAWDOWN_HALT"
        if reason is not None:
            self._append_audit(
                **self._audit_values(
                    event="ENTRY_REJECTED",
                    pair=pair,
                    recorded_at=current_time,
                    metadata=metadata,
                    rejection_reason=reason,
                )
            )
            return 0.0

        assert metadata is not None
        risk_fraction = (
            _REDUCED_RISK if drawdown >= _RISK_REDUCTION_DRAWDOWN else _NORMAL_RISK
        )
        effective_leverage = float(leverage)
        if not isfinite(effective_leverage) or effective_leverage <= 0.0:
            return 0.0
        stake = equity * risk_fraction / (
            metadata.stop_distance * effective_leverage
        )
        stake = min(float(max_stake), stake)
        if not isfinite(stake) or stake <= 0.0 or (
            min_stake is not None and stake < float(min_stake)
        ):
            self._append_audit(
                **self._audit_values(
                    event="ENTRY_REJECTED",
                    pair=pair,
                    recorded_at=current_time,
                    metadata=metadata,
                    rejection_reason="BELOW_MIN_STAKE",
                )
            )
            return 0.0
        return stake

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> Any:
        del pair, current_time, current_profit, after_fill, kwargs
        metadata = self._parse_entry_tag(getattr(trade, "enter_tag", None))
        if metadata is None:
            return self.stoploss
        stop_rate = float(trade.open_rate) * (
            1.0 + metadata.stop_distance
            if bool(trade.is_short)
            else 1.0 - metadata.stop_distance
        )
        return stoploss_from_absolute(
            stop_rate,
            current_rate,
            is_short=bool(trade.is_short),
            leverage=float(trade.leverage),
        )

    def _latest_closed_row(
        self, pair: str, current_time: datetime
    ) -> pd.Series | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        eligible = dataframe.loc[dates < pd.Timestamp(current_time)]
        return None if eligible.empty else eligible.iloc[-1]

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        del current_rate, current_profit, kwargs
        metadata = self._parse_entry_tag(getattr(trade, "enter_tag", None))
        if metadata is None or metadata.direction != ("S" if trade.is_short else "L"):
            return "INVALID_ENTRY_METADATA"
        state, _drawdown = self._update_risk(self._current_equity(current_time))
        reason: str | None = "DRAWDOWN_HALT" if state.halted else None
        row = self._latest_closed_row(pair, current_time)
        if reason is None and row is not None:
            side_sign = -1.0 if trade.is_short else 1.0
            close = float(row["close"])
            btc_z = float(row["lead_lag_btc_z"])
            convergence = side_sign * log(close / float(trade.open_rate)) >= (
                0.8 * side_sign * metadata.initial_gap_return
            )
            opposite = (
                isfinite(btc_z)
                and side_sign * btc_z <= -self._spec.shock_z_threshold
            )
            held = current_time - trade.open_date_utc >= _TIMEOUT
            reason = (
                "CONVERGENCE"
                if convergence
                else "OPPOSITE_SHOCK"
                if opposite
                else "TIMEOUT"
                if held
                else None
            )
        if reason is not None:
            self._append_audit(
                **self._audit_values(
                    event="EXIT_SIGNAL",
                    pair=pair,
                    recorded_at=current_time,
                    metadata=metadata,
                    exit_reason=reason,
                )
            )
        return reason

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> bool:
        del order_type, amount, time_in_force, kwargs
        metadata = self._parse_entry_tag(entry_tag)
        expected_direction = "S" if side == "short" else "L"
        rejection: str | None = None
        if pair not in ALT_TARGET_PAIRS or metadata is None:
            rejection = "INVALID_ENTRY_TAG"
        elif metadata.direction != expected_direction:
            rejection = "ENTRY_SIDE_MISMATCH"
        elif not isfinite(rate) or rate <= 0.0:
            rejection = "INVALID_ORDER_RATE"
        ready = (
            metadata.signal_at.to_pydatetime() + timedelta(hours=4)
            if metadata is not None
            else None
        )
        if rejection is None and ready is not None and current_time < ready:
            rejection = "EARLY_ORDER_TIME"
        elif rejection is None and ready is not None and current_time > ready + _ENTRY_VALIDITY:
            rejection = "LATE_ORDER_TIME"
        elif rejection is None:
            state, _ = self._update_risk(self._current_equity(current_time))
            if state.halted:
                rejection = "DRAWDOWN_HALT"

        values = self._audit_values(
            event="ENTRY_REJECTED" if rejection else "ENTRY_ORDER",
            pair=pair,
            recorded_at=current_time,
            metadata=metadata,
            order_time=self._iso(current_time),
            rejection_reason=rejection,
        )
        if ready is not None:
            values["latency_ms"] = int((current_time - ready).total_seconds() * 1000)
        if not self._append_audit(**values):
            return False
        if rejection is not None:
            return False
        pending = dict(getattr(self, "_pending_entries", {}))
        pending[pair] = {
            "expected_rate": float(rate),
            "order_time": current_time,
            "side": side,
        }
        self._pending_entries = pending
        return True

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        del kwargs
        metadata = self._parse_entry_tag(getattr(trade, "enter_tag", None))
        is_entry = getattr(order, "ft_order_side", None) == getattr(
            trade, "entry_side", None
        )
        pending = dict(getattr(self, "_pending_entries", {}))
        details = pending.pop(pair, None) if is_entry else None
        self._pending_entries = pending
        fill_price_raw = getattr(order, "safe_price", None)
        fill_price = float(fill_price_raw) if fill_price_raw is not None else None
        slippage: float | None = None
        if details is not None and fill_price is not None:
            expected_rate = float(details["expected_rate"])
            direction = -1.0 if details["side"] == "short" else 1.0
            slippage = direction * (fill_price / expected_rate - 1.0) * 10_000.0
        latency: int | None = None
        if metadata is not None:
            ready = metadata.signal_at.to_pydatetime() + timedelta(hours=4)
            latency = int((current_time - ready).total_seconds() * 1000)
        fee_raw = (
            getattr(trade, "fee_open_cost", None)
            if is_entry
            else getattr(trade, "fee_close_cost", None)
        )
        funding_raw = getattr(trade, "funding_fees", None)
        self._append_audit(
            **self._audit_values(
                event="ENTRY_FILL" if is_entry else "EXIT_FILL",
                pair=pair,
                recorded_at=current_time,
                metadata=metadata,
                order_time=self._iso(details["order_time"]) if details else None,
                fill_price=fill_price,
                latency_ms=latency,
                slippage_bps=slippage,
                fee=float(fee_raw) if fee_raw is not None else None,
                funding=float(funding_raw) if funding_raw is not None else None,
            )
        )
