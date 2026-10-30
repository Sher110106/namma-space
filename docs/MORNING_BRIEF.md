# Morning brief for Sher: Friday 30 Oct, from 09:00

You film. We build. This page lists what you do, in order. About one hour of your time before 10:00, then short check-ins.

Open `docs/SOP_card.pdf` on your phone as well. It is the one-page version of steps 3 to 6.

**Do this first: confirm your Team ID. Without a Team ID, nothing we build can be submitted.**

## The day at a glance (IST)

| When | You | We |
|---|---|---|
| 09:00-09:15 | Check registration and Team ID (step 0). Read this page. | `./ns doctor` |
| 09:15-09:30 | Choose the space, set the room and the phone, answer step 2, film the 30 s test clip, AirDrop it | `./ns check` (about 1 min), then `./ns run --quick` on the test clip |
| 09:30-10:00 | Take A, take B, sketch photo, height, optional tape distances, 30 s clip of you capturing | Check each video as it arrives |
| 10:00 | Stay reachable for 30 min after the last upload | `./ns run` on A and B |
| about 10:25 | Answer if we say "reshoot now" | SfM verdict. If take A registers under 75%, you reshoot |
| about 11:30 | | Final bundle, verify, report |
| about 12:00 | 5 min: look at your room in the browser and say if it looks like your room | |
| 14:00 | Last time a reshoot can start | |
| 21:30 | 30 min: review with us, record the voice-over (15 min) or accept the computer voice | |
| 31 Oct 09:00 | Sign off publishing (step 7) | Model freeze, `./ns publish` |
| 31 Oct about 10:00 | 2 min: open every link on your phone, including inside WhatsApp | |
| 31 Oct 10:00-15:00 | Upload the Drive folder, set "anyone with the link can view", test it logged out | Package, PDFs with live links |
| 31 Oct 16:00-18:00 | Submit the Drive link on the Techfest site. Screenshot the confirmation. | Latest 21:00. Deadline 23:59. |

If your capture arrives late: a capture at 18:00 on 30 Oct still fits. We run it overnight and finish the documents in the morning. With no usable capture by 31 Oct 12:00, we ship a fallback built from a public room, and label it as such.

## Step 0. Team ID (09:00, 15 min)

1. Open https://techfest.org/competitions/nammaspace and log in.
2. Check three things: your Techfest account works, your team exists, you can see the Team ID.
3. Write the Team ID in your reply to us. We print it on the abstract and the master PDF.
4. If anything is missing, ask at once:
   - the Doubt section in the portal;
   - Aryan Tamboli in the WhatsApp group (he said on 29 Oct: "Once you register on website, you can make the submission once you form a team").
   - Do not use email. The page says email queries are ignored.
5. The site listed 15 Oct as the registration deadline. The chat suggests teams can still form. We do not know. Ask today.

Tell us the result before you film. If you have no Team ID, we change the plan with you before you spend time on the capture.

## Step 1. Choose the space (2 min)

You said you can film your own room, or a meeting room with a table, four walls and one door. Both can work. Pick the one with the most to look at.

| Space | Verdict |
|---|---|
| A room with texture: shelves, books, posters, pictures, a rug, plants, mixed furniture | Best. Choose this. |
| A meeting room with some texture: writing on the whiteboard, posters, a patterned carpet, chairs, bags | Good |
| A bare meeting room: white walls, plain floor, plain table | Accepted, but risky. See below. |
| Any room with glass walls, big mirrors, wall-sized windows or a large screen | Avoid if you have a choice |

Also prefer a space with a door into a second space (a corridor or another room). Take B uses it.

**Why a bare room is risky.**
1. We find where the camera stood in every frame by matching small marks between frames: a corner, a poster edge, a book spine. A blank wall has no marks. The camera path then breaks, or splits into two models that do not join.
2. A blank wall gives the model nothing to fix its depth on. Bare walls come out smooth and smeared, with stray blobs in the blank areas.
3. Glass, mirrors, screens and glossy tables show a picture that moves as you move. The model places that picture at the wrong depth.
4. A narrow room leaves no space for the inner loop (step 5). We cut it down, and the walls have fewer views.

**If a meeting room is what you have, do this (5 min):**
- Pull the chairs out at angles. Put a few books, bags, coats or papers on the table and chairs. Nothing personal or private.
- Leave the whiteboard as it is, or write something harmless on it.
- Turn off wall screens and projectors. Close blinds on glass walls.
- Turn all lights on.
- Tell us what you added. We state it in the abstract. We never claim a bare room worked if you changed it.

**The model and video will be public.** Clear personal papers, photos, name plates and screens. If the room is not yours, ask the owner before you film, and tell them the model and the video go online. Nobody may appear in the video.

## Step 2. Tell us three things about your phone (before you film)

1. Model (for example "iPhone 15 Pro" or "Pixel 8").
2. Can you turn HDR video off, and lock exposure? If you cannot find the setting, say so. Setting names change with the phone and the software version, and the card lists the usual names. We tell you the nearest option.
3. Free storage (keep at least 5 GB free) and battery (above 50%).

## Step 3. Set the room (5 min)

- Nobody in the shot. Pets out.
- Ceiling fan off and stopped. Curtains still.
- Door fully open and wedged.
- All lights on. No bright patches of direct sun on the floor or walls.
- Cover mirrors.
- Wipe the camera lens.

## Step 4. Set the phone, then film the 30 s test clip (5 min)

Settings are on the card. In short: 4K at 30 fps, HDR video off, camera lock on, stabilisation off, 1x lens, landscape, exposure and focus locked.

- If the lights flicker on screen, use 25 fps (iPhone: Settings > Camera > Record Video > Show PAL Formats).
- Film 30 s: stand in the doorway facing in, hold still for 3 s, then walk slowly along two walls with the phone at chest height.
- AirDrop the original file to the Mac. Do not send it through WhatsApp or Messages. They shrink the video and spoil it.
- Wait about 1 minute for our check result (PASS, WARN or FAIL, with what to change). Do not wait for the quick build. It runs while you film take A.

## Step 5. Take A and take B (about 20 min)

The card has the walk. In short:

- **Take A: the room, 3-4 min, one video.** Doorway, outer loop along the walls, inner loop facing the walls with tilt-ups at the corners, low pass around the table, finish where you started.
- **Take B: the room and the next space, 4-5 min, a separate video.** The outer loop, out through the door slowly, both loops in the next space, back through the same door slowly, end inside the room.
- Hold the phone at chest height, landscape, both hands. Turn your body, not your wrist. Walk about one step per second. Do not spin on the spot, zoom or switch lens.

AirDrop both files when you finish each one.

## Step 6. Also give us (about 10 min)

1. **A photo of your hand sketch:** the outline, the door, any windows, big furniture, and where you started.
2. **Your height in cm.**
3. **Optional tape distances.** With a tape or the phone's Measure app: the room's length and width wall to wall at floor level, and the door's width. Write how you measured. Skip this if it costs time. The scale works without it, and the tape only checks it.
4. **A 30 s clip of the capture process,** filmed by someone else or re-enacted. It goes in the video.

Send this note with the files (copy and fill it in):

```
team_id:
phone:
height_cm:
space: (own room / meeting room / other)
what_I_added_to_the_room: (nothing / list)
length_m:      how measured:
width_m:       how measured:
door_width_m:  how measured:
```

## Step 7. Sign-offs (answer yes or no; nothing goes public before 31 Oct 09:00 and your typed yes)

| Question | What it means |
|---|---|
| Public GitHub repo and GitHub Pages hosting, under Sher110106 | The code and the viewer become public. The viewer shows your room. |
| Publish the room model and the demo video | Anyone with the link can see your room. |
| Put a selected set of your frames in `test/` under CC BY 4.0 | The organizers can re-run the reconstruction from your inputs. If you say no, we use a public sample instead. |
| Optional Hugging Face mirror | A second host for the model file. It needs your account. You log in. We never see your password. |
| Your voice in the video (15 min) | If not, a computer voice reads the script. |
| The Drive folder with "anyone with the link can view" | You upload it. You own the account. |

## Step 8. Time you give

| What | How long |
|---|---|
| Before 10:00: steps 0 to 6 | about 60 min |
| After the last upload | stay reachable for 30 min |
| About 12:00: "does it look like your room?" | 5 min |
| 21:30 on 30 Oct: review and voice-over | 30 min |
| 31 Oct: sign-off at 09:00, link check, Drive upload, portal submission | about 90 min across the day |
| A friend with a Windows laptop or an Android phone, after publishing | 2 min of theirs (frame-rate check) |

## What we do not ask for

No LiDAR, no tripod, no markers, no tape (tape is optional), no special app. One phone.

## Tick list

- [ ] Team ID confirmed and sent to us
- [ ] Space chosen; if bare, texture added and logged
- [ ] Phone answers sent (model, HDR off, storage)
- [ ] Room set; lens clean
- [ ] 30 s test clip sent; check result read
- [ ] Take A sent
- [ ] Take B sent
- [ ] Sketch photo, height, optional tape distances, process clip sent
- [ ] Sign-off answers sent
- [ ] Calendar: 12:00 (5 min), 21:30 (30 min), 31 Oct 09:00
