# 《一盏灯，一颗星》 A Lamp and a Star — Screenplay & Shot List

Runtime 242 s (4:02) at 24 fps. **All timing comes from `build/timeline.json`** — in scene
code use `f.cue('name')`, `f.since('name')`, `f.line('ID')` (local scene seconds), never
hard-coded global times. Local times below are for orientation (they match the timeline
at time of writing). Lines are listed as `ID start→end`.

Emotional arc: quiet loneliness → wonder → warmth & friendship → play & laughter →
fear (running out of time) → sacrifice → miracle → joy → quiet happy ending.

---

## s01 · 开场 Opening — 24.0 s
Mood: vast, hushed, magical.
1. **0–6.5 Title in the sky.** Black; stars fade in from 0.3 s (`s01_stars_in`). At
   `title_in` (1.5) the title **一盏灯，一颗星** (Noto Serif, large, warm-white) assembles from
   drifting star particles in the upper-middle of the sky, with the English
   *A Lamp and a Star* small beneath. Gentle glow. At `title_out` (6.5) the letters dissolve
   back into stardust that drifts up and becomes part of the Milky Way.
2. **6.0–13.5 Tilt down.** Camera starts high in a rich starfield with the Milky Way arcing
   across and slowly tilts down (`tilt_start`→`tilt_end`, ease in-out) until the horizon of a
   dark calm sea sits at ~2/3 frame height. N01 7.2→12.8 plays over it. A distant meteor
   streak or two.
3. **13.5–20.2 Find the lighthouse.** Slow push-in towards a tiny rocky island with a
   lighthouse whose beam slowly sweeps across the water (beam reflections on the sea).
   N02 13.6→20.2.
4. **20.2–24 Meet Deng.** Push closer: on the lighthouse gallery (balcony) a tiny figure with
   a glowing chest — Deng. His antenna bulb blinks. N03 20.8→23.2. Push towards the lamp
   room glass for the crossfade into s02 (s02 crossfades in over its first 1.0 s).

## s02 · 阿灯的夜晚 Deng's evening — 27.0 s
Mood: cosy routine, then loneliness.
1. **0–4 Climb.** (Crossfade in.) Inside the tower: Deng climbs the spiral staircase with a
   little oil can, creaky steps (`stairs_interior` + robot walk cycle). Warm light from his
   chest porthole lights the walls around him.
2. **4–10.8 Lamp room.** `s02_arrive` 4.0: he steps into the lamp room (`lamp_room`), the big
   Fresnel lens dark. He faces the glass and the night sea. D01 4.4→8.9 ("晚上好，大海…") —
   he gives the lens a loving polish, then pulls a lever. `lamp_ignite` 9.4: the lamp
   blooms to life (flash, warm light floods the room, lens begins rotating, dust motes).
3. **10.8–19.3 The beam & 300 years.** `s02_exterior` 10.8: wide exterior, the beam sweeping
   the empty sea. N04 11.2→19.0. `timelapse_start` 11.5 → `timelapse_end` 18.5: a
   time-lapse — the sky spins into star trails around the pole star, clouds streak, the
   beam becomes a blurred disc of light; nights pass (sky flickers subtly). Empty sea, no
   ships ever. Ease back to real time at 18.5.
4. **19.3–27 Alone.** `s02_sit` 19.3: medium shot — Deng sits on the gallery edge, legs
   dangling, chin in his hands, the beam passing behind him periodically, scarf in the wind.
   D02 20.3→24.4 ("要是有人，能看见这束光，就好了。") sad eyes. A small sigh (shoulders
   drop) after. Hold on the lonely image.

## s03 · 流星 The falling star — 20.0 s
Mood: wonder → urgency.
1. **0–7 Meteor.** Over-the-shoulder / behind Deng on the gallery, huge sky. One star
   twinkles oddly. `meteor_start` 2.5: it shoots across the sky, curving *down*, growing
   brighter and bigger, sparks shedding. Deng's head follows. D03 3.3→5.2 ("咦？那是，什么？").
2. **7–10 Impact.** `impact` 7.0: it plunges into the sea some distance from the island —
   a burst of light, a splash, and a ring-wave of **bioluminescent plankton** lighting up
   across the water in expanding rings (the sea sparkles cyan-gold). Lighthouse and Deng
   rim-lit by the flash. D04 8.2→9.8 ("掉进海里了！") — Deng startled, springs up.
3. **10.2–13.5 Hurry.** `run_down` 10.2: comic quick cut — Deng clatters down the spiral
   stairs super fast (motion smears), bursts out of the lighthouse door on the rocks.
4. **13.5–20 Boat.** `boat_out` 13.5: wide, low angle: Deng rows the little boat (lantern
   at the bow) across the glowing sea towards a faint underwater glow. Gentle, mysterious.

## s04 · 相遇 The meeting — 29.0 s
Mood: tender, intimate. Close shots.
1. **0–5 Rescue.** Close on the water beside the boat: faint glow under the surface.
   `lift` 0.8: Deng's mitten hands dip in and lift out a small dim star, dripping (wet=1),
   curled up, shivering, eyes shut. G01 2.6→4.9 ("呜呜，好冷，好黑。") teary.
2. **5–11 Warmth.** Deng holds her close to his chest porthole; the heart glow warms her and
   her own glow returns a little. D05 5.8→11.0 ("别怕，别怕。我叫阿灯。你是从天上掉下来的吗？")
   gentle, head tilted.
3. **11–16.4 Guang.** She opens big shiny eyes, sniffles, then a small embarrassed smile.
   G02 11.7→16.1 ("我叫小光…一不小心，就掉下来了。") — little gestures (points at sky,
   covers face with her arm-points).
4. **16.4–21 The sky is far.** `look_sky` 16.4: both look up; camera tilts to the sky — so
   far away. G03 16.9→20.8 ("天亮以前，我一定要回家。不然，我就会熄灭的。") — her glow
   flickers on "熄灭".
5. **21–25.5 Promise.** `determined` 21.2: Deng looks at her, eyes turn determined, he
   raises a fist. D06 21.9→25.1 ("别担心。我们一起想办法！"). She beams (glow blooms).
6. **25.5–29 Row back.** `row_back` 25.5: wide — the little boat with two lights (lantern +
   Guang on Deng's head) rowing back to the lighthouse across the glowing water.

## s05 · 想办法 Trying — 38.0 s
Mood: playful, funny, warm; then worry creeps in. (Crossfade in 0.8 s.)
1. **0–5 Throw.** On the island rocks at the lighthouse base. Deng crouches with Guang in his
   hands. D07 1.2→2.9 ("一、二、三，飞！"). `throw1` 3.0: he flings her up — she soars
   (sparkle trail), slows at the top, spins, falls, and `catch` 5.0 lands *on his head*
   (squash on both).
2. **5–10.3 Again.** Sitting on his head she laughs: G04 5.4→7.3 ("哈哈，再高一点！").
   Deng: D08 7.9→10.0 ("那就，再高一点！") — rolls up imaginary sleeves.
3. **10.3–16.5 The tower.** `stack` 10.3: quick 3-step montage (cuts or fast motion) of Deng
   stacking crates and barrels into a tall wobbly tower beside the lighthouse. `climb` 13.5:
   he stands on top on tiptoes, arms fully up, Guang held high towards the stars — the stars
   are still impossibly far. The tower wobbles more and more.
4. **16.5–21.8 Crash.** `tower_fall` 16.5: the tower topples; `splash` 17.3: they fall into
   the shallow water — big splash. Deng's head pops up with seaweed on it, Guang pops out
   laughing, sparkling. G05 18.4→21.0 ("哈哈哈！阿灯，你好笨呀！"). Deng sheepish, then he
   laughs too (antenna bulb blinking happily, happy arc eyes).
5. **21.8–31 Together.** `sit_together` 21.8: calm wide-ish shot, they sit on a rock at the
   water's edge, Guang on his knee, looking at the stars; reflections in the water.
   G06 23.0→25.5 ("阿灯，你一直都是一个人吗？"). D10 26.2→29.7 ("是啊。不过今天晚上，不是了。")
   — he looks at her; she glows brighter with happiness.
6. **31–38 Dawn is coming.** The horizon starts turning pale violet/peach (dawn ≈ 0 → 0.35).
   `dawn_warning` 31.3: Guang's light flickers and dims (desaturates). G07 32.2→35.4
   ("阿灯，我好像，越来越暗了。") sleepy eyes. Deng alarmed. `idea` 36.4: the lighthouse beam
   sweeps over them — Deng looks up at it; antenna bulb flashes: an idea!

## s06 · 灯塔的光 The beam — 48.0 s  ★ CLIMAX
Mood: urgency → heartbreak → awe.
1. **0–6.3 Idea.** Close on Deng's face lit by the passing beam, looking up. D11 0.6→6.1
   ("灯塔的光，能照得很远很远。也许，也能照到天上！").
2. **6.3–9.2 Run.** `run_up` 6.3: fast climb up the stairs holding dim Guang.
3. **9.2–13.6 Aim up.** `aim_up` 9.2: lamp room — Deng heaves a big iron wheel; gears grind;
   the lens assembly tilts to point straight up; the roof hatch above opens to the sky.
   `beam_up` 11.8: the beam shoots upward — cut to wide exterior: a vertical beam rising
   from the lighthouse… but it fades out long before it reaches the stars.
4. **13.6–20.4 Not enough.** D12 13.6→16.2 ("还不够，还差一点点。"). Guang lies in his palm,
   almost grey, flickering (glow ≈ 0.15). `touch_heart` 16.6: Deng looks down at his own chest —
   his heart core glows in the porthole. He touches the glass. G08 18.0→20.4
   ("阿灯，那是你的心。不要！") weak, reaching out.
5. **20.4–30.8 The choice.** Deng smiles softly. D13 21.2→25.9 ("三百年了，我一直在等一艘船。")
   — maybe the lamp room glass shows the empty sea. D14 26.9→30.2 ("原来，我等的，是你。") —
   close-up, warm, his eyes on her.
6. **30.8–33.8 The heart.** `heart_out` 30.8: he opens the porthole (creak), lifts out the
   glowing heart core (embers); his eyes flicker and dim, his body sags. `heart_in` 33.0: he
   places it into the lamp.
7. **33.8–40 IGNITION.** `ignite` 33.8: white flash (grade white≈0.8 fading), shockwave,
   and a colossal golden pillar of light (`fx.light_pillar`) blasts from the lighthouse into
   the sky — wide shot, camera shake, clouds blown open in a ring, the pillar reaching the
   stars. `rise` 35.3: Guang, bathed in light, regains her colour and glow (glow → 1.5+) and
   floats up the beam, spinning slowly, sparkle trail. G09 36.8→39.3 ("阿灯，阿灯！") looking
   back down, reaching.
8. **39.5–48 Mother.** `whale_appear` 39.5: high above, the Milky Way swirls and the
   constellation whale emerges, swimming down around the top of the pillar. D15 40.2→42.5
   ("回家吧，小光。") — cut to Deng in the lamp room, dimming, a weak wave. `beam_fade` 44.0:
   the pillar thins and fades; the whale gathers Guang up. `deng_dark` 45.0: Deng's eyes go
   dark, he slumps against the lamp. Silence. Hold on the dark lighthouse.

## s07 · 回答 The answer — 34.0 s
Mood: grief → miracle → joy → golden morning.
1. **0–3 Darkness.** The dark lighthouse, the dark robot slumped in the lamp room (or on the
   gallery). Only waves. Stars quiet.
2. **3–11 The whale comes.** `whale_song` 3.0: a low whale song. The constellation whale
   descends from the sky, enormous and gentle, towards the lighthouse; tiny bright Guang
   rides on her head. `stardust` 6.5: the whale breathes out / sweeps her tail: a river of
   stardust flows down (`fx.stardust_stream`) into the lamp room and swirls around Deng,
   pouring into his empty chest.
3. **11–16.4 Reboot.** `reboot` 11.0: a new **star-shaped heart** lights in his chest
   (heart_star → 1), his eyes flicker on — blink, blink — he looks down at his chest in
   wonder. God rays. W01 13.0→15.8 ("谢谢你，守灯人。") — the whale's huge kind eye close
   to the gallery, her mouth gently moving.
4. **16.4–25.8 Goodbye.** `nuzzle` 16.4: Guang flies down and hugs Deng's face.
   G10 17.2→20.8 ("阿灯！以后每天晚上，我都会在天上，看着你！"). D16 21.5→25.4
   ("好。我也会一直，为你发光。") — happy arc eyes.
5. **25.8–34 Sunrise.** `farewell` 25.8: Guang flies up to the whale, who rises and swims
   up into the sky. `sunrise` 27.0: dawn breaks — sky warms to peach and gold, the sun rises
   over the sea, the sea turns gold, the whale dissolves into the brightening sky as a trail
   of sparkles. Deng waves from the gallery in golden light, scarf fluttering.

## s08 · 尾声 Epilogue — 22.0 s
Mood: quiet happy ending. (Crossfade in 1.5 s.)
1. **0–9.8 Another night.** Night again. Wide: lighthouse lit, beam sweeping, Deng on the
   gallery with his star-shaped heart glowing. N05 1.2→9.0. Exactly at `blink1` and
   `blink2` (synced to the words "一闪，一闪") one particular star right above the lighthouse
   flares twice — a little sparkle burst with cross-shaped glints.
2. **9.2–15 Wave.** `wave` 9.2: Deng waves up at the star. N06 9.8→14.2 ("原来，每一束光，
   都会被看见。"). Camera slowly cranes up and back: lighthouse becomes tiny, the sky fills
   the frame, the answering star twinkles.
3. **15–22 End.** `end_title` 15.0: title **一盏灯，一颗星** returns in the sky (same style as
   s01), with "完 · The End". `credits` 18.0: small elegant credit text:
   - 画面、配音、音乐、音效 —— 全部由代码生成
   - Every frame, voice, note and sound in this film was generated by code.
   - Python · Skia · Kokoro TTS · FluidSynth · Claude
   `fade_out` 21.0 → fade to black by 22.0.

---
## Conventions shared by scenes, ROBOT and SFX
- **Walk cadence.** While Deng walks/climbs, scenes set
  `walk_phase = (f.t - segment_start) * cadence` with cadence **1.0 cycle/s** for normal
  walking/climbing and **2.2 cycles/s** for hurrying (s03 `run_down`, s06 `run_up`).
  A foot touches down at every half cycle (phase 0.0, 0.5, 1.0, …). SFX places footstep
  sounds on exactly those instants, so keep segment starts on the cue times given here:
  s02 climb 0.0→`s02_arrive` (normal); s03 `run_down`→`boat_out` (hurry);
  s06 `run_up`→`aim_up` (hurry).
- **Rowing.** One oar stroke every 1.6 s starting at `boat_out` (s03) and `row_back` (s04);
  the oar blade enters the water at the start of each stroke.
- **Lighthouse beam sweep** (exterior shots): beam angle = `f.T * 0.9` radians (global
  time, so it is continuous across cuts) unless the story needs it stopped/aimed.
