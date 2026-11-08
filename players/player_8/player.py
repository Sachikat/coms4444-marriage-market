"""Group 8's improved strategy: skeptical, adaptive, and deception-aware."""

from math import ceil

from models.player import (
    Advertisement,
    Candidate,
    DateContext,
    DateOutcome,
    Decision,
    GameContext,
    PlayerSnapshot,
    RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import Profile, decrement


class Player8(BasePlayer):
    """Skeptical player that learns the pool, detects liars, and adapts over time."""

    def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
        super().__init__(snapshot, ctx)
        # Core tracking
        self.estimates: dict[int, float] = {}
        self.proposed_to_me: set[int] = set()
        self.we_proposed_to: set[int] = set()
        self.unanswered_proposals: dict[int, int] = {}
        
        # Observation history for lie detection and estimation
        self.obs: dict[int, list[tuple]] = {}  # partner -> list of observed profiles
        self.stated: dict[int, list[tuple]] = {}  # partner -> list of stated ideals
        self.sent: dict[int, tuple] = {}  # what we projected to them
        
        # Population learning
        self.pop: list[tuple] = []  # all first observations
        self.est_hist: list[float] = []
        
        # Exploration phase settings
        self.exploration_fraction = 0.20
        self.exploration_rounds = max(2, ceil(self.rounds * self.exploration_fraction))
        
        # Consistency tracking for lie detection
        self.flagged_liars: set[int] = set()
        
        # Configuration
        self.max_unanswered = 3  # stop chasing after this many ignored proposals

    def _g(self, dim: int, want: float, have: float) -> float:
        """Single dimension decrement calculation."""
        gap = want - have
        if dim < self.monotonic and gap <= 0:
            return 0.0
        return gap * gap

    def _pop_mean(self, dim: int) -> float:
        """Population mean for a dimension, with fallback to 5."""
        if len(self.pop) < 3:
            return 5.0
        return sum(p[dim] for p in self.pop) / len(self.pop)

    def _prior_gap(self, dim: int, have: float) -> float:
        """Expected partner decrement on dim if their ideal ~ population."""
        samples = [p[dim] for p in self.pop] if len(self.pop) >= 3 else list(range(11))
        return sum(self._g(dim, w, have) for w in samples) / len(samples)

    def _quantile(self, xs: list[float], q: float) -> float:
        """Get quantile from list of values."""
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]

    def _ad_cost(self, ad: Advertisement) -> float:
        """Estimate cost from advertised dimensions."""
        mine = self._g(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
        theirs = self._g(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
        return mine + theirs / 2

    def _is_consistent(self, partner: int, ad: Advertisement) -> bool:
        """Check if partner has been consistent (honest) across dates.
        
        Liars may change their projected profile or stated ideal between dates,
        or project values that contradict their advertisement.
        """
        obs = self.obs.get(partner, [])
        stated = self.stated.get(partner, [])
        
        if len(obs) < 2:
            return True  # Not enough data to judge
            
        # Check if projected profile is consistent across dates
        if any(o != obs[0] for o in obs):
            return False
            
        # Check if stated ideal is consistent
        if any(s != stated[0] for s in stated):
            return False
            
        # Check if their projection matches their ad
        if obs[0][ad.own_dim] != ad.own_value:
            return False
            
        return True

    def _partner_profile(self, partner: int, ad: Advertisement) -> list[float]:
        """Estimate partner's true profile from observations."""
        obs = self.obs[partner]
        n = len(obs)
        est = []
        
        for i in range(self.dims):
            vals = sorted(o[i] for o in obs)
            v = float(vals[len(vals) // 2])  # median
            
            # On monotonic dimensions, be conservative about boastful claims
            if i < self.monotonic and ad.own_dim == i:
                v = min(v, ad.own_value)
                
            # Shrink toward pool mean, less with more evidence
            lam = 0.3 / n
            est.append((1 - lam) * v + lam * self._pop_mean(i))
            
        return est

    def _their_cost(self, partner: int, ad: Advertisement) -> float:
        """Expected decrement partner suffers from our TRUE profile.
        
        Be skeptical of stated ideals, especially if they match what we projected.
        """
        stated = self.stated[partner]
        sent = self.sent.get(partner)
        
        # If their stated ideal closely matches what we projected, weight it less
        w_stated = 0.5
        if sent is not None:
            d = sum(abs(stated[-1][i] - sent[i]) for i in range(self.dims)) / self.dims
            if d < 1.0:
                w_stated = 0.05  # They might be flattering us
                
        total = 0.0
        for i in range(self.dims):
            have = self.profile[i]
            from_stated = sum(self._g(i, s[i], have) for s in stated) / len(stated)
            prior = self._prior_gap(i, have)
            
            w = w_stated
            # The fixed ad is somewhat more credible
            if i == ad.seek_dim:
                from_stated = (from_stated + self._g(i, ad.seek_value, have)) / 2
                w = max(w, 0.6)
                
            total += w * from_stated + (1 - w) * prior
            
        return total

    def _mutual_happiness(self, observed: Profile, stated_ideal: Profile) -> tuple[float, float]:
        """Calculate both parties' estimated happiness if married."""
        mine = sum(self._g(i, self.ideal[i], observed[i]) for i in range(self.dims))
        theirs = sum(self._g(i, stated_ideal[i], self.profile[i]) for i in range(self.dims))
        our_happiness = self.bonus - mine - theirs / 2
        their_happiness = self.bonus - theirs - mine / 2
        return our_happiness, their_happiness

    def advertise(self) -> Advertisement:
        """Advertise honestly, but avoid suspicious perfect 10s on monotonic dims."""
        showable = range(self.monotonic) if self.monotonic else range(self.dims)
        
        # Prefer dimensions where we're good but not suspiciously perfect
        good_dims = [i for i in showable if 6 <= self.profile[i] <= 9]
        if good_dims:
            own = max(good_dims, key=lambda i: self.profile[i])
        else:
            own = max(showable, key=lambda i: self.profile[i])
            
        # For seek: show dimension where our ideal is most extreme
        seek = max(
            range(self.dims),
            key=lambda i: (abs(self.ideal[i] - 5), i >= self.monotonic),
        )
        
        return Advertisement(
            own_dim=own,
            own_value=self.profile[own],
            seek_dim=seek,
            seek_value=self.ideal[seek],
        )

    def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
        """Prioritize known good matches, then explore new candidates."""
        # People we've dated and estimated positively (excluding flagged liars)
        chosen = [
            c.index for c in candidates
            if self.estimates.get(c.index, 0.0) > 0
            and c.index not in self.flagged_liars
            and self.unanswered_proposals.get(c.index, 0) < self.max_unanswered
        ]
        
        # Fresh candidates sorted by ad fit
        fresh = sorted(
            (c for c in candidates if c.index not in self.estimates),
            key=lambda c: (self._ad_cost(c.ad), c.index),
        )
        
        # Explore more early, less later
        frac_left = turn.rounds_left / max(1, self.rounds)
        k = max(1, int(len(fresh) * (0.3 + 0.5 * frac_left)))
        chosen += [c.index for c in fresh[:k]]
        
        return chosen[: self.list_limit]

    def project(self, date: DateContext) -> Profile:
        """Project a profile slightly shifted toward partner's preference.
        
        Shift up to 2 points on any dimension toward what they advertised seeking.
        """
        projected = list(self.profile)
        dim, value = date.partner_ad.seek_dim, date.partner_ad.seek_value
        
        if dim < self.monotonic:
            # On monotonic: slightly inflate if they want more
            projected[dim] = min(10, max(projected[dim], min(value, projected[dim] + 2)))
        else:
            # On non-monotonic: shift toward their preference
            step = max(-2, min(2, value - projected[dim]))
            projected[dim] = projected[dim] + step
            
        out = tuple(projected)
        self.sent[date.partner] = out
        return out

    def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
        """Describe ideal shifted slightly toward observed, but not exact copy.
        
        Exact copy is detectable as flattery by sophisticated players (like Player1).
        """
        result = []
        for i, (want, saw) in enumerate(zip(self.ideal, observed)):
            # Move at most 1-2 points toward what they showed
            shift = max(-2, min(2, saw - want))
            # Only shift partway to avoid looking like obvious flattery
            partial_shift = shift // 2 if abs(shift) > 1 else shift
            result.append(max(0, min(10, want + partial_shift)))
        return tuple(result)

    def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
        """Decide whether to continue, propose, or mark incompatible."""
        p = date.partner
        ad = date.partner_ad
        
        # Record observations
        self.obs.setdefault(p, []).append(tuple(observed))
        self.stated.setdefault(p, []).append(tuple(stated_ideal))
        if len(self.obs[p]) == 1:
            self.pop.append(tuple(observed))
            
        # Check for lying
        if not self._is_consistent(p, ad):
            self.flagged_liars.add(p)
            # Don't permanently block - they might still be a good match
            # but be very skeptical
            
        # Calculate estimate
        if p in self.flagged_liars:
            # For liars, use population-based estimate (pessimistic)
            prof = [self._pop_mean(i) for i in range(self.dims)]
        else:
            prof = self._partner_profile(p, ad)
            
        mine = sum(self._g(i, self.ideal[i], prof[i]) for i in range(self.dims))
        theirs = self._their_cost(p, ad)
        estimate = self.bonus - mine - theirs / 2
        self.estimates[p] = estimate
        self.est_hist.append(estimate)
        
        # Calculate mutual happiness for better decision making
        our_h, their_h = self._mutual_happiness(observed, stated_ideal)
        mutual_positive = our_h > 0 and their_h > 0
        
        frac_left = date.rounds_left / max(1, self.rounds)
        margin = (0.02 + 0.08 * frac_left) * self.bonus
        
        # During exploration phase, just observe
        if date.round <= self.exploration_rounds:
            # Only mark incompatible if extremely bad
            if estimate < -0.5 * self.bonus:
                return Decision(incompatible=True)
            return Decision()
            
        # Check if we've proposed too many times without response
        if self.unanswered_proposals.get(p, 0) >= self.max_unanswered:
            return Decision(incompatible=True)
            
        # Hopeless match
        hopeless = estimate < -0.25 * self.bonus or (
            date.date_number >= 3 and estimate <= 0 and frac_left > 0.3
        )
        if hopeless:
            return Decision(incompatible=True)
            
        # Determine if we should ask again
        again = estimate > (-0.1 * self.bonus if frac_left > 0.3 else 0)
        
        # Determine proposal threshold
        if len(self.est_hist) >= 5:
            bar = self._quantile(self.est_hist, 0.75 * frac_left)
        else:
            bar = 0.5 * self.bonus * frac_left
        bar = max(bar, margin)
        
        # Lower bar if they proposed to us
        if p in self.proposed_to_me:
            bar = margin
            
        # Boost confidence if mutual positive happiness
        if mutual_positive:
            bar *= 0.8
            
        propose = date.date_number >= 2 and estimate > bar
        
        if propose:
            self.we_proposed_to.add(p)
            
        return Decision(again=again or propose, propose=propose)

    def on_outcome(self, outcome: DateOutcome) -> None:
        """Track proposals received and unanswered."""
        if outcome.partner_proposed:
            self.proposed_to_me.add(outcome.partner)
            
        # Track unanswered proposals
        if outcome.partner in self.we_proposed_to and not outcome.married:
            if not outcome.partner_proposed:
                self.unanswered_proposals[outcome.partner] = \
                    self.unanswered_proposals.get(outcome.partner, 0) + 1