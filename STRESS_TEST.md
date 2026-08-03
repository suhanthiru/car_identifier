# Stress test: drive the operator console

No project knowledge needed. You need ~10 minutes and a browser.

> This walkthrough assumes CityFlow is downloaded (see DATASETS.md). Without
> it `python start.py` runs the synthetic world instead — the same workflow
> and the same controls, with simulated vehicles rather than real footage,
> and the banner says so.

## What this is

Real traffic-camera footage from a public research dataset (CityFlow), five cameras
at one intersection cluster, 95 real vehicles. You pick a car. The system then
watches the other cameras and tells you where it thinks that car turned up again —
and, importantly, tells you when it *isn't sure*.

It is **not** a product. It is a demo of a reasoning layer: the interesting part is
what it refuses to claim, not what it claims.

## Start it

```bash
python start.py
```

A browser opens at `http://127.0.0.1:8010`. The first launch spends ~15s building
vehicle thumbnails; after that it's cached and starts immediately.

Use the **⏸ PAUSE** button in the top-right whenever you want to stop and look at
something. It freezes the replay clock but leaves the review queue fully workable —
so you can read a card, compare clips, and accept or reject while nothing new
arrives. It turns amber while paused. Resume picks up exactly where it left off.

Add `--time-scale 2` if things still move too fast.

## What you're looking at

- **Left — map.** The five real cameras. Lines are routes vehicles have actually
  been observed taking between them.
- **Middle — review queue.** Where the system puts things it wants you to look at.
- **Right — targets.** Cars you've flagged.
- **Bottom strip.** The reasoning stages: PLATE → CLASS ATTRS → GEOMETRY → REID.

## Do this

1. Scroll the **browse panel** and click any car thumbnail. That flags it — you've
   told the system "watch for this one."
2. Wait. As the replay runs, cards appear in the review queue when the system thinks
   it may have seen your car again.
3. Open a card. Read the plain-English reasons — colour match, travel time, appearance
   similarity. Each line is a real signal with a real weight, not decoration.
4. Click **Accept** or **Reject**. Accept teaches it; reject pushes belief down.
5. Click a target on the right to open its dossier — a looping clip of the actual
   footage where it was spotted.

## The thing that will surprise you

Cards usually say **"candidate set — cannot assert individual"** instead of naming
your car. That is deliberate, and it's the whole point.

On low-resolution traffic cameras, two silver sedans are often genuinely
indistinguishable — not "hard", *impossible* from pixels alone. Measured on this
data, the appearance model does no better than a coin flip at telling apart two
same-colour cars across different cameras. A system that confidently picked one
would be guessing and hiding it.

So it narrows 95 cars to about 4 and hands you the set. A licence-plate read or a
distinctive feature (roof rack, damage) is what lets it name one — and then it
will say so outright.

## Try to break it

1. **Pick the most boring car you can find** — a plain silver or grey sedan. Watch it
   struggle honestly, and see how many candidates it offers.
2. **Pick two similar-looking cars and flag both.** Do their review cards get confused
   with each other? Do they list each other as candidates?
3. **Accept a match you can see is wrong** (compare the clips yourself). Does the system
   recover afterwards, or does one bad confirmation poison it?
4. **Turn off plate reading** — the toggle in the pipeline strip. Notice the PLATE node
   dim. Does the reasoning visibly lean harder on the other signals?
5. **Flag a car late in the replay**, after it has already driven past most cameras.
   Does it behave sensibly with almost no runway?
6. **Just wait without touching anything.** Does it ever fire an alert on its own? It
   shouldn't — nothing auto-confirms on appearance alone.

## What counts as a bug

- It states a car **is** a specific vehicle with only colour + appearance to go on.
- A card's stated reasons don't match what you see in the clips.
- It proposes a car that was physically nowhere near that camera at that time.
- The UI hangs, a panel goes blank, or a clip fails to load.
- It silently does nothing for a very obvious, distinctive vehicle.

## Known, already-measured limits

- About **half** of real camera-to-camera passages get surfaced. It misses things.
- Roughly **1 in 23** suggestions involving a same-colour car is wrong on the
  default backbone (1 in 38 on FastReID) — the same-colour impostor column
  in RESULTS.md.
- Colour is estimated from pixels and gets fooled by lighting; silver/grey/white
  are treated as interchangeable on purpose.
- Numbers behind all of this are in `RESULTS.md`, including the unflattering ones.
