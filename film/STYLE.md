# 《一盏灯，一颗星》 A Lamp and a Star — Style Bible

Read this before drawing anything. The film must look like ONE film, not eight.

## Look in one sentence
A moonlit picture-book: soft rounded shapes, smooth gradients instead of outlines,
deep indigo nights lit by warm gold lamplight and cool starlight, everything that
glows really *glows* (additive light + the engine's bloom), gentle film grain.

## Rules
- **No black outlines.** Separate shapes by value and colour. If an edge needs help,
  use a slightly darker or lighter rim of the *same hue* (1–3 units), or a soft rim light.
- **Rounded, chunky, friendly shapes.** No sharp spiky details except the stars' twinkles.
- **Light logic.** Warm light (lamp, heart core, Guang) = `#ffd36b` / `#ffb347`.
  Cool light (moon/sky/stars/whale) = `#9fc4ff` / `#6ff0ff`. Objects near a light source get
  a rim/bounce of that light's colour. Characters standing in front of the lamp get a warm
  rim light; in open night they get a cool rim on top.
- **Depth.** Far things are lower contrast and bluer (atmospheric perspective). Use parallax
  layers via `Camera.apply(c, depth)`.
- **Glow.** Draw light with `blend=ADD` (engine.core.ADD) radial gradients. The engine adds a
  bloom pass on top of anything brighter than ~0.55 luminance, so bright cores bloom by
  themselves — don't over-blur.
- **Motion.** Everything alive moves a little all the time (idle breathing, twinkle, water,
  scarf, hair-like glow). Use easing — never linear starts/stops. Overshoot and settle
  (`ease_out_back`, `spring`) for character actions. Anticipation before big moves.
- **Determinism.** Frames are rendered in parallel, out of order. Every visual must be a pure
  function of time. Use seeded RNG (`engine.core.nprng(seed)`) created at module level or
  inside functions with fixed seeds. Never `random.random()` without a seed.

## Palette
| Role | Colours |
|---|---|
| Night sky (top → horizon) | `#070b1f` → `#16224d` → `#2c3f7a` (+ faint teal band `#2e6f8e` at horizon) |
| Stars | warm `#fff6e0`, blue-white `#cfe3ff` |
| Sea | deep `#081631`, mid `#123060`, highlight `#5c8fd6`, foam `#bfd8ff` |
| Rocks / island | `#1d2440`, `#2a3358`, moss `#2f4a4a` |
| Lighthouse | white `#e8e4d8` (moonlit `#aab4d4`), red stripes `#b8413a` (moonlit `#7a3a4a`), iron `#2a2f45`, lamp `#ffd36b` |
| Deng (robot) | body teal `#4f8a9a` / shade `#3b6e7e` / light `#7fb5c2`, brass `#c9a15b`, rust `#9a5b3c`, red stripe + scarf `#c8453c`, screen `#10202a`, eyes `#ffe9a8`, heart core `#ffb347` → `#fff0c8` |
| Guang (star) | core `#fffbea`, body `#ffe58a`, edge `#ffc94a`, glow `#ffd86b`, cheeks `#ff9fb2`, eyes `#2a2140` |
| Guang dimmed | body `#8d97b8`, edge `#6b7599` (desaturated blue-grey) |
| Whale | body `#3d7bd9` → `#8a5bd9` translucent, lines `#9fd8ff`, star nodes `#ffffff` |
| Dawn | top `#27306a`, mid `#8a6fb0`, horizon `#f5a878` |
| Sunrise | sun `#fff1c4`, glow `#ffcf80`, sea gold `#e0a86a` |

## Characters
**阿灯 Deng** — ~360 units tall at scale 1. A little robot that looks like a tiny lighthouse:
tapered barrel body in teal with a red stripe, a round brass porthole on the chest showing a
glowing amber heart core, short stubby legs with round boots, long bendy tube arms with brass
joints and three-fingered mitten hands. Head: rounded-rectangle old TV/diving helmet with a
dark screen face on which his two big glowing eyes and simple glowing mouth are drawn (so
emotions are shown by eye shapes: arcs for happy, droops for sad, etc.). Small antenna with a
red bulb that blinks. A hand-knitted red scarf that flutters in the wind. Rust spots and a
riveted patch = 300 years old. Moves slowly, a bit creaky, but gentle and warm.

**小光 Guang** — ~110 units across at scale 1 (fits in Deng's two hands). A plump five-pointed
star with rounded points, glowing warm gold, big shiny eyes with highlights, tiny mouth, pink
cheeks. Upper side points work as arms; she floats and bobs, leaves sparkles. When dying she
fades to blue-grey, halo shrinks, light flickers.

**星鲸 Star whale** — Guang's mother. Enormous (~1400 units long at scale 1): a whale drawn as
a translucent nebula body with a constellation "skeleton" (bright star nodes joined by thin
glowing lines), a bright star for an eye, flowing fins, a tail that leaves stardust. Slow,
majestic, kind.

## Typography
Chinese: Noto Serif CJK SC (titles), Noto Sans CJK SC (subtitles). Use `engine.core.font()`.

## Framing
16:9, 1920x1080 design units. Subtitles occupy y > 930 — keep faces and key action above
y ≈ 900 when dialogue is on screen.
