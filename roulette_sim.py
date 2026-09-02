#!/usr/bin/env python3
"""Monte Carlo model of a 3-stage double-zero roulette strategy.

Simulates sessions until the table stack can no longer cover a $10 Stage 1 bet,
and reports how many spins a given budget typically lasts.

Pocket rule (default on): the first time the table stack reaches $150, lock the
original $100 buy-in in your pocket and keep playing with the $50 profit. If
the table stack hits $150 again, lock another $100 and leave $50 in play.
Stage thresholds use total session profit (pocket + table − buy-in), so
pocketing does not reset the ladder.

American double-zero wheel only (0, 00, 1–36). Triple-zero is not modelled.
Every stage has the same 5.26% house edge; bigger stages just lose faster.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass

# --- Wheel -----------------------------------------------------------------

RED = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
GREEN = {0, 37}  # 37 stands in for 00

# Five $2 corners used in Stage 2.
CORNERS = [
    {1, 2, 4, 5},
    {8, 9, 11, 12},
    {16, 17, 19, 20},
    {23, 24, 26, 27},
    {31, 32, 34, 35},
]
CORNER_NUMBERS = set.union(*CORNERS)
RED_INSIDE = RED & CORNER_NUMBERS
RED_OUTSIDE = RED - CORNER_NUMBERS
BLACK_INSIDE = BLACK & CORNER_NUMBERS
BLACK_OUTSIDE_OR_GREEN = (BLACK - CORNER_NUMBERS) | GREEN

WHEEL = list(range(0, 37)) + [37]  # 0..36 plus 00

# --- Stage payouts ---------------------------------------------------------
# Stage 1: $10 on Red. 18/38 win +$10, else -$10.
# Stage 2: $10 Red + five $2 corners ($20 total).
#   Red inside corner:     +$18
#   Red outside corner:     $0
#   Black inside corner:   -$2
#   Black outside / green: -$20
# Stage 3: $10 Red + $10 1st dozen + $10 2nd dozen ($30 total).
#   Red in 1–24:           +$20
#   Black in 1–24:          $0
#   Red in 25–36:          -$10  (soft loss)
#   Black in 25–36 / green:-$30  (wipeout)


def spin_wheel(rng: random.Random) -> int:
    return rng.choice(WHEEL)


def stage1_pnl(n: int) -> int:
    return 10 if n in RED else -10


def stage2_pnl(n: int) -> int:
    if n in RED_INSIDE:
        return 18
    if n in RED_OUTSIDE:
        return 0
    if n in BLACK_INSIDE:
        return -2
    return -20


def stage3_pnl(n: int) -> int:
    if n in GREEN:
        return -30
    if 1 <= n <= 24:
        return 20 if n in RED else 0
    # 25–36
    return -10 if n in RED else -30


PNL_FN = {1: stage1_pnl, 2: stage2_pnl, 3: stage3_pnl}
STAGE_BET = {1: 10, 2: 20, 3: 30}

STAGE2_ENTRY = 20  # enter / remain eligible when session profit > $20
STAGE3_ENTRY = 40  # enter Stage 3 when session profit > $40
PROFIT_FLOOR = 20  # any drop below +$20 resets to Stage 1
DEFAULT_PLAY_WITH = 50  # chips left on the table after a pocket


@dataclass
class SessionResult:
    spins: int
    table_left: int
    pocketed: int
    walked_with: int
    peak_table: int
    peak_wealth: int
    pocket_count: int
    stage_spins: tuple[int, int, int]  # counts at stages 1, 2, 3
    hit_spin_cap: bool
    stop_reason: str  # bust, time, reserve, pocket


def maybe_pocket(
    table: int, pocket: int, pocket_at: int | None, play_with: int
) -> tuple[int, int, int]:
    """If the table stack hits the threshold, lock everything but play_with.

    Returns (table, pocket, amount_just_moved). No-op when pocket_at is None.
    """
    if pocket_at is None or table < pocket_at:
        return table, pocket, 0
    moved = table - play_with
    return play_with, pocket + moved, moved


def affordable_stage(stage: int, bankroll: int, profit: int) -> int:
    """Clamp a decided stage to the profit floor and remaining cash. No promotion."""
    if profit < PROFIT_FLOOR:
        stage = 1
    while stage > 1 and bankroll < STAGE_BET[stage]:
        stage -= 1
    if bankroll < STAGE_BET[1]:
        return 0
    return stage


def apply_stage_rules(
    stage: int, pnl: int, profit: int, consecutive_soft: int
) -> tuple[int, int]:
    """Next stage and soft-loss streak after a spin resolves."""
    if stage == 3:
        if pnl == -30:
            return 1, 0  # full wipeout: instant Stage 1
        if pnl == -10:
            consecutive_soft += 1
            if profit < PROFIT_FLOOR:
                return 1, 0
            if consecutive_soft >= 2:
                return (2 if profit > STAGE2_ENTRY else 1), 0
            return 3, consecutive_soft  # single soft loss: stay
        if profit < PROFIT_FLOOR:
            return 1, 0
        return 3, 0
    if profit < PROFIT_FLOOR:
        return 1, 0
    if profit > STAGE3_ENTRY:
        return 3, 0
    if profit > STAGE2_ENTRY:
        return 2, 0
    return 1, 0


def play_session(
    rng: random.Random,
    bankroll: int,
    max_spins: int,
    pocket_at: int | None = None,
    play_with: int = DEFAULT_PLAY_WITH,
    walk_on_pocket: bool = False,
    stage1_only: bool = False,
    keep: int = 0,
) -> SessionResult:
    start = bankroll
    table = bankroll
    pocket = 0
    peak_table = table
    peak_wealth = start
    pocket_count = 0
    stage = 1
    consecutive_soft = 0
    spins = 0
    stage_counts = [0, 0, 0, 0]
    hit_cap = False
    stop_reason = "bust"

    while spins < max_spins:
        profit = pocket + table - start
        if stage1_only:
            stage = 1 if table >= STAGE_BET[1] else 0
        else:
            stage = affordable_stage(stage, table, profit)
        if stage == 0:
            stop_reason = "bust"
            break

        n = spin_wheel(rng)
        pnl = PNL_FN[stage](n)
        if table + pnl < 0:
            pnl = -table
        table += pnl
        spins += 1
        stage_counts[stage] += 1
        if table > peak_table:
            peak_table = table

        table, pocket, moved = maybe_pocket(table, pocket, pocket_at, play_with)
        if moved:
            pocket_count += 1

        wealth = pocket + table
        if wealth > peak_wealth:
            peak_wealth = wealth

        if moved and walk_on_pocket:
            stop_reason = "pocket"
            break
        if keep and wealth <= keep:
            stop_reason = "reserve"
            break

        if stage1_only:
            consecutive_soft = 0
            stage = 1
        else:
            stage, consecutive_soft = apply_stage_rules(
                stage, pnl, pocket + table - start, consecutive_soft
            )
    else:
        hit_cap = True
        stop_reason = "time"

    return SessionResult(
        spins=spins,
        table_left=table,
        pocketed=pocket,
        walked_with=pocket + table,
        peak_table=peak_table,
        peak_wealth=peak_wealth,
        pocket_count=pocket_count,
        stage_spins=(stage_counts[1], stage_counts[2], stage_counts[3]),
        hit_spin_cap=hit_cap,
        stop_reason=stop_reason,
    )


def play_flat(rng: random.Random, bankroll: int, max_spins: int) -> int:
    """Stage 1 only: $10 on red until you can't."""
    spins = 0
    while spins < max_spins and bankroll >= STAGE_BET[1]:
        bankroll += stage1_pnl(spin_wheel(rng))
        spins += 1
    return spins


def percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def ascii_hist(spins: list[int], bins: list[int]) -> str:
    """Right-open bins; last bin is 'bins[-1]+'."""
    counts = [0] * len(bins)
    for s in spins:
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    width = max(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        if i < len(bins) - 1:
            label = f"{bins[i]:>4}–{bins[i + 1] - 1:<4}"
        else:
            label = f"{bins[i]:>4}+    "
        bar = "█" * int(40 * count / width)
        pct = 100.0 * count / len(spins)
        lines.append(f"  {label}  {bar:<40} {count:6}  ({pct:5.1f}%)")
    return "\n".join(lines)


def format_number(n: int) -> str:
    return "00" if n == 37 else str(n)


def color_of(n: int) -> str:
    if n in RED:
        return "red"
    if n in BLACK:
        return "black"
    return "green"


def trace_session(
    rng: random.Random,
    bankroll: int,
    max_spins: int,
    pocket_at: int | None,
    play_with: int,
) -> None:
    start = bankroll
    table = bankroll
    pocket = 0
    stage = 1
    consecutive_soft = 0
    spin_no = 0
    print(
        f"{'spin':>4}  {'n':>4}  color   stage  pnl   table  pocket  profit  next"
    )
    print("-" * 72)
    while spin_no < max_spins:
        profit = pocket + table - start
        stage = affordable_stage(stage, table, profit)
        if stage == 0:
            print(
                f"\nStopped after {spin_no} spins. "
                f"Walked with ${pocket + table} "
                f"(${pocket} pocketed, ${table} on the table)."
            )
            return
        n = spin_wheel(rng)
        pnl = PNL_FN[stage](n)
        table += pnl
        spin_no += 1
        table, pocket, moved = maybe_pocket(table, pocket, pocket_at, play_with)
        next_stage, consecutive_soft = apply_stage_rules(
            stage, pnl, pocket + table - start, consecutive_soft
        )
        print(
            f"{spin_no:4d}  {format_number(n):>4}  {color_of(n):<5}  "
            f"S{stage}   {pnl:+4d}   ${table:4d}   ${pocket:4d}  "
            f"{pocket + table - start:+6d}  S{next_stage}"
            + (f"  POCKET ${moved}" if moved else "")
        )
        stage = next_stage
    print(
        f"\nHit spin cap ({max_spins}). "
        f"Walked with ${pocket + table} "
        f"(${pocket} pocketed, ${table} on the table)."
    )


def verify_payout_tables() -> None:
    """Sanity-check coverage counts and house edge (2/38 of amount wagered)."""
    assert len(WHEEL) == 38
    assert len(RED) == 18 and len(BLACK) == 18
    assert len(RED_INSIDE) == 10 and len(RED_OUTSIDE) == 8
    assert len(BLACK_INSIDE) == 10
    assert len(BLACK_OUTSIDE_OR_GREEN) == 10
    assert len(CORNER_NUMBERS) == 20

    def ev(fn) -> float:
        return sum(fn(n) for n in WHEEL) / 38

    e1, e2, e3 = ev(stage1_pnl), ev(stage2_pnl), ev(stage3_pnl)
    # House edge 2/38 ≈ 5.263% of the total bet.
    assert abs(e1 - (-10 * 2 / 38)) < 1e-9, e1
    assert abs(e2 - (-20 * 2 / 38)) < 1e-9, e2
    assert abs(e3 - (-30 * 2 / 38)) < 1e-9, e3

    # Stage machine
    assert apply_stage_rules(1, 10, 10, 0) == (1, 0)  # still ≤ $20
    assert apply_stage_rules(1, 10, 30, 0) == (2, 0)  # > $20 → Stage 2
    assert apply_stage_rules(2, 18, 48, 0) == (3, 0)  # > $40 → Stage 3
    assert apply_stage_rules(3, -30, 50, 0) == (1, 0)  # wipeout → Stage 1
    assert apply_stage_rules(3, -10, 35, 0) == (3, 1)  # one soft loss: stay
    assert apply_stage_rules(3, -10, 25, 1) == (2, 0)  # two soft losses → 2
    assert apply_stage_rules(3, -10, 15, 0) == (1, 0)  # floor breach
    assert apply_stage_rules(2, -20, 5, 0) == (1, 0)

    assert maybe_pocket(149, 0, 150, 50) == (149, 0, 0)
    assert maybe_pocket(150, 0, 150, 50) == (50, 100, 100)
    assert maybe_pocket(168, 0, 150, 50) == (50, 118, 118)
    assert maybe_pocket(150, 100, 150, 50) == (50, 200, 100)
    assert maybe_pocket(150, 0, None, 50) == (150, 0, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bankroll", type=int, default=100, help="Starting cash (default 100)")
    p.add_argument("--sessions", type=int, default=100_000, help="Simulated sessions")
    p.add_argument("--max-spins", type=int, default=10_000, help="Per-session spin cap")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--trace", action="store_true", help="Print one sample session and exit")
    p.add_argument(
        "--pocket-at",
        type=int,
        default=None,
        help="Lock the buy-in when the table stack reaches this (default: bankroll+50)",
    )
    p.add_argument(
        "--play-with",
        type=int,
        default=DEFAULT_PLAY_WITH,
        help="Chips left on the table after pocketing (default 50)",
    )
    p.add_argument(
        "--no-pocket",
        action="store_true",
        help="Disable the pocket rule (play the whole stack to ruin)",
    )
    p.add_argument(
        "--night",
        action="store_true",
        help="Compare night-out strategies: time, leftover cash, and drinks",
    )
    p.add_argument(
        "--minutes",
        type=int,
        default=75,
        help="Night-out time cap in minutes (default 75)",
    )
    p.add_argument(
        "--spins-per-hour",
        type=int,
        default=45,
        help="Table pace (default 45; packed ~35, empty ~55)",
    )
    p.add_argument(
        "--keep",
        type=int,
        default=50,
        help="Night-out reserve: stand up when total cash hits this (default 50)",
    )
    return p.parse_args()


def resolve_pocket_at(args: argparse.Namespace) -> int | None:
    if args.no_pocket:
        return None
    pocket_at = args.pocket_at if args.pocket_at is not None else args.bankroll + 50
    if args.play_with >= pocket_at:
        raise SystemExit("--play-with must be less than --pocket-at")
    return pocket_at


def minutes_of(spins: int, sph: int) -> float:
    return 60.0 * spins / sph


def drink_count(minutes: float) -> int:
    """Sit-down drink, then about one every 20 minutes if you keep ordering."""
    if minutes < 8:
        return 0
    return 1 + int((minutes - 8) / 20)


def summarize_night(results: list[SessionResult], sph: int, bankroll: int) -> dict:
    n = len(results)
    mins = [minutes_of(r.spins, sph) for r in results]
    walked = [r.walked_with for r in results]
    drinks = [drink_count(m) for m in mins]
    reasons = Counter(r.stop_reason for r in results)
    return {
        "median_min": statistics.median(mins),
        "mean_min": statistics.mean(mins),
        "p60": 100.0 * sum(m >= 60 for m in mins) / n,
        "p75": 100.0 * sum(m >= 75 for m in mins) / n,
        "p90": 100.0 * sum(m >= 90 for m in mins) / n,
        "median_cash": statistics.median(walked),
        "mean_cash": statistics.mean(walked),
        "p50": 100.0 * sum(w >= 50 for w in walked) / n,
        "p100": 100.0 * sum(w >= bankroll for w in walked) / n,
        "mean_drinks": statistics.mean(drinks),
        "p_bust": 100.0 * reasons.get("bust", 0) / n,
        "p_lost": 100.0 * sum(r.walked_with < 10 for r in results) / n,
        "p_time": 100.0 * reasons.get("time", 0) / n,
        "p_reserve": 100.0 * reasons.get("reserve", 0) / n,
        "p_pocket_walk": 100.0 * reasons.get("pocket", 0) / n,
        "p_ever_pocket": 100.0 * sum(r.pocket_count >= 1 for r in results) / n,
    }


def run_named(n, bankroll, max_spins, seed, **kwargs) -> list[SessionResult]:
    rng = random.Random(seed)
    return [play_session(rng, bankroll, max_spins, **kwargs) for _ in range(n)]


def print_night_report(args: argparse.Namespace, pocket_at: int | None) -> None:
    sph = args.spins_per_hour
    cap = max(1, math.ceil(args.minutes * sph / 60.0))
    packed_cap = max(1, math.ceil(args.minutes * 35 / 60.0))
    n = args.sessions
    bankroll = args.bankroll
    keep = args.keep
    minutes = args.minutes

    scenarios = [
        (
            "A  Opener ($10, stand at $50)",
            dict(
                pocket_at=pocket_at,
                walk_on_pocket=True,
                stage1_only=True,
                keep=keep,
            ),
            cap,
            sph,
            args.seed,
        ),
        (
            f"B  Night block ($10, {minutes}m timer)",
            dict(
                pocket_at=pocket_at,
                walk_on_pocket=False,
                stage1_only=True,
                keep=0,
            ),
            cap,
            sph,
            None if args.seed is None else args.seed + 17,
        ),
        (
            f"C  Packed table ($10, {minutes}m)",
            dict(
                pocket_at=pocket_at,
                walk_on_pocket=False,
                stage1_only=True,
                keep=0,
            ),
            packed_cap,
            35,
            None if args.seed is None else args.seed + 34,
        ),
        (
            "D  Ladder to ruin (old plan)",
            dict(
                pocket_at=pocket_at,
                walk_on_pocket=False,
                stage1_only=False,
                keep=0,
            ),
            10_000,
            sph,
            None if args.seed is None else args.seed + 51,
        ),
    ]

    print(f"Vegas night-out  |  ${bankroll} on the roulette table  |  {n:,} sessions")
    print(f"Default pace {sph} spins/hour ({cap} spins in {minutes} min).")
    print("Packed ~35/hr — more minutes and drink passes per chip.")
    print("Empty ~55/hr — burns the $100 faster in clock time.")
    print()
    print(
        f"{'plan':<36} {'med min':>7} {'60m+':>6} {'timer':>6} "
        f"{'med $':>6} {'≥$50':>6} {'≥$100':>6} {'drinks':>6} {'lost $':>6}"
    )
    print("-" * 96)

    stats = {}
    for name, kwargs, max_spins, pace, seed in scenarios:
        results = run_named(n, bankroll, max_spins, seed, **kwargs)
        s = summarize_night(results, pace, bankroll)
        stats[name] = s
        print(
            f"{name:<36} {s['median_min']:7.0f} {s['p60']:5.0f}% {s['p_time']:5.0f}% "
            f"${s['median_cash']:5.0f} {s['p50']:5.0f}% {s['p100']:5.0f}% "
            f"{s['mean_drinks']:6.1f} {s['p_lost']:5.0f}%"
        )

    rec = stats[f"B  Night block ($10, {minutes}m timer)"]
    packed = stats[f"C  Packed table ($10, {minutes}m)"]
    opener = stats["A  Opener ($10, stand at $50)"]
    print()
    print(f"Plan B — what happens in the {minutes}-minute $10-flat block")
    print(f"  still sitting when the timer rings:   {rec['p_time']:5.1f}%")
    print(f"  last at least an hour:                {rec['p60']:5.1f}%")
    print(f"  lock the $100 (hit $150 at some point): {rec['p_ever_pocket']:4.1f}%")
    print(f"  lose the entire $100:                 {rec['p_lost']:5.1f}%")
    print(f"  walk with ≥ $50 still:                {rec['p50']:5.1f}%")
    print(f"  mean cash leaving the table:          ${rec['mean_cash']:.0f}")
    print(f"  mean drinks if you keep ordering:     {rec['mean_drinks']:.1f}")
    print()
    print("The constraint: $100 cannot buy 75 minutes AND a $50 safety net")
    print("at $10 a spin. Five losing chips and a tight reserve stands you up")
    print(f"around {opener['median_min']:.0f} minutes (Plan A). To actually sit 60–90 minutes")
    print("you have to risk the roulette $100, and keep other-game money in a")
    print("different pocket.")
    print()
    print("Playbook")
    print("  1. Split the wallet before you sit.")
    print("     Left pocket: $100 for roulette. Right pocket: BJ / UTH / slots.")
    print("     The right pocket does not come out at this table. Ever.")
    print("  2. $10 on red or black only. Skip Stage 2/3 — they buy fewer")
    print("     spins and fewer cocktail-server loops.")
    print("  3. Double-zero wheel. Walk past 000. Sit at a busy table so the")
    print("     dealer is slow (Plan C: "
          f"{packed['p60']:.0f}% last an hour vs "
          f"{rec['p60']:.0f}% at a normal pace).")
    print("  4. Phone timer: 75 minutes. Order as you sit, tip $1–2/drink,")
    print("     order every time they come by. 2–4 drinks is a realistic night.")
    print("  5. If the stack hits $150: pocket the original $100 (that can join")
    print("     the other-games stash) and play the $50 until the timer or it")
    print("     is gone. You are now drinking on house money.")
    print("  6. Timer rings → stand up, even if it is fun. That is how you still")
    print("     play the rest of the night. If the $100 dies early (~29% of")
    print("     nights), do not rebuy. Other-games money is still intact.")
    print("  7. Strip $15–25 mins at peak: same $100 lasts ~2/3 as long. Hunt a")
    print("     $10 table (early evening, downtown, locals) or shorten the block.")
    print()
    ev = 10 * (2 / 38) * sph * minutes / 60
    print(f"Expected cost of Plan B: about ${ev:.0f} in house edge")
    print(f"({sph} × $10 bets/hour × 5.26% × {minutes} min). That is the price of")
    print("the seat, the drinks, and the show — not a system that gets paid.")


def main() -> None:
    verify_payout_tables()
    args = parse_args()
    rng = random.Random(args.seed)
    pocket_at = resolve_pocket_at(args)

    if args.trace:
        trace_session(rng, args.bankroll, args.max_spins, pocket_at, args.play_with)
        return

    if args.night:
        print_night_report(args, pocket_at)
        return

    results = [
        play_session(rng, args.bankroll, args.max_spins, pocket_at, args.play_with)
        for _ in range(args.sessions)
    ]
    flat_rng = random.Random(None if args.seed is None else args.seed + 1)
    flat_spins = sorted(
        play_flat(flat_rng, args.bankroll, args.max_spins) for _ in range(args.sessions)
    )
    n = args.sessions
    spins = sorted(r.spins for r in results)
    walked = sorted(r.walked_with for r in results)
    peaks = [r.peak_table for r in results]
    peak_wealth = [r.peak_wealth for r in results]
    caps = sum(r.hit_spin_cap for r in results)
    s1 = sum(r.stage_spins[0] for r in results)
    s2 = sum(r.stage_spins[1] for r in results)
    s3 = sum(r.stage_spins[2] for r in results)
    total_spins = s1 + s2 + s3
    pocketed_once = sum(r.pocket_count >= 1 for r in results)
    pocketed_twice = sum(r.pocket_count >= 2 for r in results)
    kept_buyin = sum(r.walked_with >= args.bankroll for r in results)
    left_ahead = sum(r.walked_with > args.bankroll for r in results)
    lost_all = sum(r.walked_with < 10 for r in results)

    thresholds = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]
    still_alive = Counter()
    for s in spins:
        for t in thresholds:
            if s >= t:
                still_alive[t] += 1

    pocket_label = (
        "off"
        if pocket_at is None
        else f"on (lock buy-in at ${pocket_at}, leave ${args.play_with} in play)"
    )
    print(f"Double-zero roulette  |  ${args.bankroll} bankroll  |  {n:,} sessions")
    print(f"Pocket rule: {pocket_label}")
    print()
    print("House edge is 5.26% of the amount bet at every stage:")
    print("  Stage 1  ($10):  EV -$0.53/spin")
    print("  Stage 2  ($20):  EV -$1.05/spin")
    print("  Stage 3  ($30):  EV -$1.58/spin")
    print()
    print("Spins until the table stack can't cover a $10 bet")
    print(f"  mean     {statistics.mean(spins):.1f}")
    print(f"  median   {statistics.median(spins):.0f}")
    print(f"  stdev    {statistics.pstdev(spins):.1f}")
    print(f"  min      {spins[0]}")
    print(f"  p10      {percentile(spins, 10):.0f}")
    print(f"  p25      {percentile(spins, 25):.0f}")
    print(f"  p75      {percentile(spins, 75):.0f}")
    print(f"  p90      {percentile(spins, 90):.0f}")
    print(f"  p99      {percentile(spins, 99):.0f}")
    print(f"  max      {spins[-1]}")
    if caps:
        print(f"  still going at {args.max_spins} spins: {caps} sessions")
    print()
    print("Chance a session lasts at least N spins")
    for t in thresholds:
        print(f"  {t:3d}+   {100.0 * still_alive[t] / n:5.1f}%")
    print()
    print("Spin count distribution")
    print(ascii_hist(spins, [0, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]))
    print()
    if pocket_at is not None:
        print("Cash you walk away with (pocket + leftover table chips)")
        print(f"  mean     ${statistics.mean(walked):.0f}")
        print(f"  median   ${statistics.median(walked):.0f}")
        print(f"  p10      ${percentile(walked, 10):.0f}")
        print(f"  p90      ${percentile(walked, 90):.0f}")
        print(f"  max      ${walked[-1]}")
        print(f"  lost the buy-in:                 {100.0 * lost_all / n:5.1f}%")
        print(f"  left with ≥ ${args.bankroll} (buy-in safe):    {100.0 * kept_buyin / n:5.1f}%")
        print(f"  left with more than ${args.bankroll}:     {100.0 * left_ahead / n:5.1f}%")
        print(f"  hit ${pocket_at} and pocketed once:      {100.0 * pocketed_once / n:5.1f}%")
        print(f"  ran it up and pocketed again:    {100.0 * pocketed_twice / n:5.1f}%")
        print()
        print("Walk-away cash distribution")
        print(ascii_hist(walked, [0, 10, 50, 100, 110, 150, 200, 250, 300, 500]))
        print()
    print("Peak chips on the table")
    print(f"  mean     ${statistics.mean(peaks):.0f}")
    print(f"  median   ${statistics.median(peaks):.0f}")
    print(f"  p90      ${percentile(sorted(peaks), 90):.0f}")
    print(f"  max      ${max(peaks)}")
    print(
        f"  ever ahead of buy-in: "
        f"{100.0 * sum(w > args.bankroll for w in peak_wealth) / n:.1f}%"
    )
    print()
    print("Where the spins were spent")
    print(f"  Stage 1  {100.0 * s1 / total_spins:5.1f}%")
    print(f"  Stage 2  {100.0 * s2 / total_spins:5.1f}%")
    print(f"  Stage 3  {100.0 * s3 / total_spins:5.1f}%")
    print()
    print("Same budget, $10 flat on red/black (no ladder, no pocket)")
    print(f"  mean     {statistics.mean(flat_spins):.1f}")
    print(f"  median   {statistics.median(flat_spins):.0f}")
    print(f"  p10      {percentile(flat_spins, 10):.0f}")
    print(f"  p90      {percentile(flat_spins, 90):.0f}")
    print()
    if pocket_at is None:
        print("Without the pocket rule you play the whole stack to ruin, so you")
        print("almost always leave with $0–$9. The ladder still does not stretch")
        print("the session versus flat $10 bets.")
    else:
        print("The pocket rule does not beat the house edge on chips in play. It")
        print("caps the damage on lucky sessions: once you have locked the buy-in,")
        print("the worst case is walking even. Sessions that never reach")
        print(f"${pocket_at} still go to zero. Use --no-pocket to compare.")


if __name__ == "__main__":
    main()
